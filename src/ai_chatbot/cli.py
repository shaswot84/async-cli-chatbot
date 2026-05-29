"""Typer CLI application providing the interactive chat REPL and slash commands."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ai_chatbot.config import AppConfig, load_config, sanitized_config
from ai_chatbot.db import ChatStore, request_to_rows
from ai_chatbot.failure_simulator import FailureSimulator, supported_failure_kinds
from ai_chatbot.llm_client import LLMClient, LLMClientError
from ai_chatbot.logging_setup import setup_logging
from ai_chatbot.session import ChatSession
from ai_chatbot.themes import available_themes, get_theme, resolve_theme_name

app = typer.Typer(add_completion=False, help="Asynchronous command-line AI chatbot.")
console: Console = Console()
logger = logging.getLogger(__name__)


def apply_theme(name: str) -> None:
    """Swap the global console for one using the named theme."""
    global console
    resolved = resolve_theme_name(name)
    console = Console(theme=get_theme(resolved))


import base64
import os
from pathlib import Path


def _save_images(images: tuple[dict[str, str], ...]) -> list[str]:
    """Save base64-encoded images to disk. Returns list of clickable file:// links or URLs."""
    saved: list[str] = []
    output_dir = Path("data/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        b64 = img.get("b64_json")
        if b64:
            path = output_dir / f"generated_{i + 1}.png"
            path.write_bytes(base64.b64decode(b64))
            saved.append(path.resolve().as_uri())  # file:///... link
            continue
        url = img.get("url")
        if url and not url.startswith("data:"):
            saved.append(url)  # remote URL, already clickable
    return saved


def _prompt_markup(session: ChatSession) -> str:
    """Build a theme-coloured prompt string for the current model."""
    styles: dict[str, str] = {
        "dark": "bold cyan",
        "light": "bold blue",
        "monokai": "bold #a6e22e",
        "dracula": "bold #bd93f9",
        "nord": "bold #88c0d0",
        "gruvbox": "bold #b8bb26",
    }
    color = styles.get(session.theme_name, "bold cyan")
    mc = session.config.models.get(session.active_model)
    if mc is not None and mc.image_capable:
        return f"[{color}]🎨 {session.active_model}>[/] "
    return f"[{color}]{session.active_model}>[/] "


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Show sanitized runtime config on start."),
) -> None:
    """Entry point: start the interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(run_repl(debug=debug))


async def run_repl(debug: bool = False) -> None:
    """Set up dependencies and run the read-eval-print loop."""
    try:
        config = load_config()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(2) from exc

    setup_logging(config.log_level, config.log_json)
    apply_theme(config.theme)

    store = ChatStore(config.sqlite_path)
    await store.initialize()
    conversation_id = await store.start_conversation(config.default_model)
    session = ChatSession(config, conversation_id=conversation_id)
    failure_simulator = FailureSimulator(
        enabled=config.simulate_failures,
        rate=config.simulate_failure_rate,
        kind=config.simulate_failure_kind,
    )
    client = LLMClient(config, failure_simulator)
    commands = command_handlers(config, session, store, failure_simulator, client)
    logger.info(
        "chat_session_started",
        extra={
            "conversation_id": conversation_id,
            "default_model": config.default_model,
            "sqlite_path": str(config.sqlite_path),
            "debug": debug,
        },
    )

    console.print(
        Panel.fit(
            f"Async CLI AI Chatbot\nConversation: {conversation_id}\nType /help for commands.",
            title="Ready",
        )
    )
    if debug:
        print_config(config)

    try:
        while True:
            try:
                user_input = console.input(_prompt_markup(session)).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/]")
                logger.info(
                    "chat_session_interrupted",
                    extra={"conversation_id": conversation_id},
                )
                break

            if not user_input:
                continue
            if user_input.startswith("/"):
                should_continue = await commands.get(command_name(user_input), unknown_command)(
                    user_input
                )
                if not should_continue:
                    break
                continue

            use_streaming = session.effective_streaming()
            try:
                if use_streaming:
                    response = await _stream_chat_turn(session, client, store, user_input)
                else:
                    with console.status("Thinking...", spinner="dots"):
                        response = await session.send(client, store, user_input)
            except LLMClientError as exc:
                logger.warning(
                    "chat_turn_failed",
                    extra={
                        "request_id": exc.request_id,
                        "conversation_id": conversation_id,
                        "model": exc.model,
                        "latency_ms": exc.latency_ms,
                        "status_code": exc.status_code,
                        "error_type": exc.error_type,
                    },
                )
                console.print(f"[red]Request failed:[/] {exc}")
                continue

            if not use_streaming:
                console.print(
                    Panel(
                        response.content or "(generated image)",
                        title=f"{response.model} · {response.latency_ms} ms",
                        style="response",
                        title_align="left",
                    )
                )
            if response.images:
                saved = _save_images(response.images)
                for path in saved:
                    console.print(f"[success]Image:[/] [link={path}]{path}[/link]")
            if debug:
                token_text = (
                    f"tokens in/out/total: "
                    f"{response.input_tokens}/{response.output_tokens}/{response.total_tokens}; "
                    f"retries: {response.retry_attempts}"
                )
                effective = session.effective_thinking_budget()
                if effective is not None:
                    token_text += f"; thinking: ON (budget: {effective})"
                else:
                    token_text += "; thinking: OFF"
                if use_streaming:
                    token_text += "; streaming: ON"
                console.print(f"[dim]{response.request_id} · {token_text}[/]")
    finally:
        await client.close()
        await store.end_conversation(conversation_id)
        logger.info(
            "chat_session_ended",
            extra={"conversation_id": conversation_id},
        )


async def _stream_chat_turn(
    session: ChatSession,
    client: LLMClient,
    store: ChatStore,
    user_input: str,
) -> ChatResponse:
    """Run a chat turn with streaming, updating a Live panel in real-time."""
    accumulated = ""

    with Live(
        Panel("...", title=f"{session.active_model} · streaming", style="response"),
        refresh_per_second=15,
        console=console,
    ) as live:

        def on_chunk(delta: str) -> None:
            nonlocal accumulated
            accumulated += delta
            live.update(
                Panel(accumulated, title=f"{session.active_model} · streaming", style="response")
            )

        try:
            response = await session.send(
                client, store, user_input, on_chunk=on_chunk
            )
        except LLMClientError as exc:
            live.update(
                Panel(str(exc), title=f"{session.active_model} · error", style="error")
            )
            raise

        title = (
            f"{response.model} · {response.latency_ms:.0f} ms"
            if response.latency_ms
            else f"{response.model} · streaming"
        )
        live.update(Panel(accumulated or "(generated image)", title=title, style="response"))
        if response.images:
            saved = _save_images(response.images)
            for path in saved:
                console.print(f"[success]Image:[/] [link={path}]{path}[/link]")
        return response


def command_name(raw: str) -> str:
    """Parse a slash command and optional sub-command from raw user input."""
    parts = raw.split()
    if len(parts) >= 2 and parts[0] == "/model":
        return " ".join(parts[:2])
    if len(parts) >= 2 and parts[0] == "/fail" and parts[1] in {"rate", "kind"}:
        return " ".join(parts[:2])
    if parts and parts[0] == "/fail":
        return "/fail"
    if len(parts) >= 2 and parts[0] == "/think" and parts[1] in {"on", "off", "budget"}:
        return " ".join(parts[:2])
    if parts and parts[0] == "/think":
        return "/think"
    if len(parts) >= 2 and parts[0] == "/stream" and parts[1] in {"on", "off"}:
        return " ".join(parts[:2])
    if parts and parts[0] == "/stream":
        return "/stream"
    if len(parts) >= 2 and parts[0] == "/theme" and parts[1] in {"list", "set"}:
        return " ".join(parts[:2])
    if parts and parts[0] == "/theme":
        return "/theme"
    return parts[0]


def command_handlers(
    config: AppConfig,
    session: ChatSession,
    store: ChatStore,
    failure_simulator: FailureSimulator,
    client: LLMClient,
) -> dict[str, Callable[[str], Awaitable[bool]]]:
    """Build a dispatch table mapping command strings to async handlers."""
    return {
        "/help": lambda _: async_value(print_help()),
        "/exit": lambda _: async_value(False),
        "/quit": lambda _: async_value(False),
        "/model list": lambda _: async_value(print_models(config, session)),
        "/model current": lambda _: async_value(print_current_model(session)),
        "/model set": lambda raw: async_value(set_model(raw, session)),
        "/history": lambda _: print_history(session, store),
        "/request": lambda raw: print_request(raw, store),
        "/fail": lambda raw: async_value(set_failure_enabled(raw, failure_simulator)),
        "/fail rate": lambda raw: async_value(set_failure_rate(raw, failure_simulator)),
        "/fail kind": lambda raw: async_value(set_failure_kind(raw, failure_simulator)),
        "/clear": lambda _: async_value(clear_history(session)),
        "/config": lambda _: async_value(print_config(config)),
        "/think": lambda _: async_value(show_thinking_status(session)),
        "/think on": lambda _: async_value(enable_thinking(session)),
        "/think off": lambda _: async_value(disable_thinking(session)),
        "/think budget": lambda raw: async_value(set_thinking_budget(raw, session)),
        "/stream": lambda _: async_value(show_streaming_status(session)),
        "/stream on": lambda _: async_value(enable_streaming(session)),
        "/stream off": lambda _: async_value(disable_streaming(session)),
        "/theme": lambda _: async_value(show_theme_status(session)),
        "/theme list": lambda _: async_value(list_themes()),
        "/theme set": lambda raw: async_value(set_theme_cmd(raw, session)),
        "/image": lambda raw: generate_image(raw, session, client, store),
    }


async def async_value(value: bool) -> bool:
    """Wrap a synchronous bool return value into an awaitable."""
    return value


def print_help() -> bool:
    """Render the help table of available slash commands."""
    table = Table(title="Commands", style="help.border", header_style="table.header")
    table.add_column("Command", style="highlight")
    table.add_column("Action")
    rows = [
        ("/help", "Show commands"),
        ("/model list", "Show configured models"),
        ("/model current", "Show active model"),
        ("/model set <model_id>", "Switch active model"),
        ("/config", "Show sanitized runtime config"),
        ("/history", "Show persisted conversation messages"),
        ("/request <request_id>", "Show SQLite request metadata"),
        ("/fail on", "Enable simulated failures"),
        ("/fail off", "Disable simulated failures"),
        ("/fail rate <0.0-1.0>", "Set simulated failure probability"),
        ("/fail kind <kind>", "Set simulated failure kind"),
        ("/think", "Show thinking status"),
        ("/think on", "Enable extended reasoning"),
        ("/think off", "Disable extended reasoning"),
        ("/think budget <tokens>", "Set thinking token budget (min 1024)"),
        ("/stream", "Show streaming status"),
        ("/stream on", "Enable response streaming"),
        ("/stream off", "Disable response streaming"),
        ("/theme", "Show current theme"),
        ("/theme list", "List available themes"),
        ("/theme set <name>", "Switch to a different theme"),
        ("/image <prompt>", "Generate an image using the image model"),
        ("/clear", "Clear in-memory conversation"),
        ("/exit", "Quit"),
    ]
    for command, action in rows:
        table.add_row(command, action)
    console.print(table)
    return True


def print_models(config: AppConfig, session: ChatSession) -> bool:
    """Render a table of configured models with the active one marked."""
    table = Table(title="Configured Models", style="table.border", header_style="table.header")
    table.add_column("Active")
    table.add_column("Model ID", style="highlight")
    table.add_column("Family")
    table.add_column("Use case")
    table.add_column("Thinking", style="success")
    table.add_column("Streaming", style="info")
    table.add_column("Image", style="warning")
    for model in config.models.values():
        thinking = (
            f"{model.thinking_budget_tokens:,}" if model.thinking_budget_tokens > 0 else "-"
        )
        streaming = "Yes" if model.streaming_capable else "No"
        image = "Yes" if model.image_capable else "-"
        table.add_row(
            "*" if model.model_id == session.active_model else "",
            model.model_id,
            model.family,
            model.use_case,
            thinking,
            streaming,
            image,
        )
    console.print(table)
    return True


def print_current_model(session: ChatSession) -> bool:
    """Print the currently active model ID and its capabilities."""
    mc = session.config.models.get(session.active_model)
    console.print("Current model: ", end="")
    console.print(session.active_model, style="highlight")
    if mc is not None:
        tags = []
        if mc.thinking_budget_tokens > 0:
            tags.append(f"thinking ({mc.thinking_budget_tokens:,})")
        if mc.image_capable:
            tags.append("image generation")
        if not mc.streaming_capable:
            tags.append("no streaming")
        if tags:
            console.print(f"  Capabilities: {', '.join(tags)}", style="dim")
    return True


def set_model(raw: str, session: ChatSession) -> bool:
    """Parse '/model set <id>' and switch the active model. Supports partial IDs."""
    parts = raw.split(maxsplit=2)
    if len(parts) < 3:
        console.print("[yellow]Usage:[/] /model set <model_id>")
        return True
    try:
        resolved = session.config.resolve_model(parts[2])
        if resolved != parts[2]:
            console.print(f"[dim]Resolved `{parts[2]}` → `{resolved}`[/]")
        session.set_model(resolved)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return True
    console.print(f"Switched to [highlight]{session.active_model}[/]")
    mc = session.config.models.get(session.active_model)
    if mc is not None and mc.image_capable:
        console.print(
            "  [dim]This is an image-generation model. Type a prompt to generate images.[/]"
        )
    return True


def set_failure_enabled(raw: str, failure_simulator: FailureSimulator) -> bool:
    """Parse '/fail on|off' and toggle failure simulation."""
    parts = raw.split(maxsplit=1)
    if len(parts) < 2 or parts[1] not in {"on", "off"}:
        console.print("[yellow]Usage:[/] /fail on | /fail off")
        return True
    if parts[1] == "on":
        failure_simulator.enable()
        console.print(
            f"Failure simulation on: rate={failure_simulator.rate}, kind={failure_simulator.kind}"
        )
    else:
        failure_simulator.disable()
        console.print("[dim]Failure simulation off.[/]")
    return True


def set_failure_rate(raw: str, failure_simulator: FailureSimulator) -> bool:
    """Parse '/fail rate <0.0-1.0>' and update the failure probability."""
    parts = raw.split(maxsplit=2)
    if len(parts) < 3:
        console.print("[yellow]Usage:[/] /fail rate <0.0-1.0>")
        return True
    try:
        failure_simulator.set_rate(float(parts[2]))
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return True
    console.print(f"Failure simulation rate: {failure_simulator.rate}")
    return True


def set_failure_kind(raw: str, failure_simulator: FailureSimulator) -> bool:
    """Parse '/fail kind <kind>' and update the failure kind."""
    parts = raw.split(maxsplit=2)
    if len(parts) < 3:
        console.print(
            f"[yellow]Usage:[/] /fail kind <{', '.join(sorted(supported_failure_kinds()))}>"
        )
        return True
    try:
        failure_simulator.set_kind(parts[2])
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return True
    console.print(f"Failure simulation kind: {failure_simulator.kind}")
    return True


def show_thinking_status(session: ChatSession) -> bool:
    """Display current thinking status, budgets, and model support."""
    model_config = session.config.models.get(session.active_model)
    model_supports = model_config is not None and model_config.thinking_budget_tokens > 0

    if session.thinking_enabled and model_supports:
        model_budget = model_config.thinking_budget_tokens
        effective = min(session.thinking_budget_tokens, model_budget)
        console.print(
            f"[green]Thinking: ON[/]\n"
            f"  Session budget: {session.thinking_budget_tokens}\n"
            f"  Model max budget ({session.active_model}): {model_budget}\n"
            f"  Effective budget: {effective}"
        )
    elif session.thinking_enabled and not model_supports:
        console.print(
            f"[yellow]Thinking: ON (inactive)[/]\n"
            f"  {session.active_model} does not configure a thinking budget.\n"
            f"  Session budget: {session.thinking_budget_tokens} (not used)"
        )
    else:
        console.print("[dim]Thinking: OFF[/]")
    return True


def enable_thinking(session: ChatSession) -> bool:
    """Turn thinking on. Warns if the active model does not support it."""
    session.set_thinking_enabled(True)
    model_config = session.config.models.get(session.active_model)
    model_supports = model_config is not None and model_config.thinking_budget_tokens > 0
    if not model_supports:
        console.print(
            f"[yellow]Thinking enabled, but {session.active_model} does not configure a "
            f"thinking budget. The thinking field will not be sent in requests.[/]"
        )
    else:
        console.print("[green]Thinking enabled.[/]")
    return True


def disable_thinking(session: ChatSession) -> bool:
    """Turn thinking off."""
    session.set_thinking_enabled(False)
    console.print("[dim]Thinking disabled.[/]")
    return True


def set_thinking_budget(raw: str, session: ChatSession) -> bool:
    """Parse '/think budget <N>' and update the session thinking token budget."""
    parts = raw.split(maxsplit=2)
    if len(parts) < 3:
        console.print("[yellow]Usage:[/] /think budget <tokens> (min 1024)")
        return True
    try:
        budget = int(parts[2])
        session.set_thinking_budget(budget)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return True
    console.print(f"Thinking budget set to [cyan]{session.thinking_budget_tokens}[/]")
    return True


def show_streaming_status(session: ChatSession) -> bool:
    """Display current streaming status."""
    model_config = session.config.models.get(session.active_model)
    model_supports = model_config is not None and model_config.streaming_capable
    if session.streaming_enabled and model_supports:
        console.print("[green]Streaming: ON[/]")
    elif session.streaming_enabled and not model_supports:
        console.print(
            f"[yellow]Streaming: ON (inactive)[/]\n"
            f"  {session.active_model} does not support streaming."
        )
    else:
        console.print("[dim]Streaming: OFF[/]")
    return True


def enable_streaming(session: ChatSession) -> bool:
    """Turn streaming on. Warns if the active model does not support it."""
    session.set_streaming(True)
    model_config = session.config.models.get(session.active_model)
    model_supports = model_config is not None and model_config.streaming_capable
    if not model_supports:
        console.print(
            f"[yellow]Streaming enabled, but {session.active_model} does not support streaming.[/]"
        )
    else:
        console.print("[green]Streaming enabled.[/]")
    return True


def disable_streaming(session: ChatSession) -> bool:
    """Turn streaming off."""
    session.set_streaming(False)
    console.print("[dim]Streaming disabled.[/]")
    return True


def show_theme_status(session: ChatSession) -> bool:
    """Display the current theme name."""
    console.print(f"Current theme: [highlight]{session.theme_name}[/]")
    return True


def list_themes() -> bool:
    """List all available themes, marking the current one."""
    current = console.width  # not used, placeholder
    table = Table(title="Available Themes", style="table.border", header_style="table.header")
    table.add_column("Theme", style="highlight")
    for name in available_themes():
        table.add_row(name)
    console.print(table)
    console.print("\n[dim]Use /theme set <name> to switch.[/]")
    return True


def set_theme_cmd(raw: str, session: ChatSession) -> bool:
    """Parse '/theme set <name>' and switch the UI theme."""
    parts = raw.split(maxsplit=2)
    if len(parts) < 3:
        console.print(
            f"[yellow]Usage:[/] /theme set <{' | '.join(available_themes())}>"
        )
        return True
    name = parts[2].strip().lower()
    try:
        session.set_theme(name)
    except Exception:
        available = ", ".join(available_themes())
        console.print(f"[red]Unknown theme `{name}`. Available: {available}[/]")
        return True
    apply_theme(session.theme_name)
    available = ", ".join(available_themes())
    console.print(
        f"Theme switched to [highlight]{session.theme_name}[/]. "
        f"Available: {available}"
    )
    return True


async def generate_image(
    raw: str, session: ChatSession, client: LLMClient, store: ChatStore
) -> bool:
    """Parse '/image <prompt>' and generate an image using the configured image model."""
    parts = raw.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        console.print("[yellow]Usage:[/] /image <prompt>")
        return True

    image_models = [
        mid for mid, mc in session.config.models.items() if mc.image_capable
    ]
    if not image_models:
        console.print("[red]No image-capable model configured.[/]")
        return True

    image_model = image_models[0]
    prompt = parts[1].strip()

    with console.status(f"Generating with {image_model}...", spinner="dots"):
        try:
            result = await client.generate_image(
                image_model, prompt, n=session.config.image_n
            )
        except LLMClientError as exc:
            console.print(f"[red]Image generation failed:[/] {exc}")
            return True

    if result.images:
        saved = _save_images(result.images)
        links = "\n".join(f"[link={p}]{p}[/link]" for p in saved)
        console.print(
            Panel(
                links,
                title=f"{result.model} · {result.latency_ms:.0f} ms",
                style="response",
            )
        )
    else:
        console.print(
            Panel(
                "[yellow]No image data returned.[/]",
                title=f"{result.model} · {result.latency_ms:.0f} ms",
                style="error",
            )
        )
    return True


async def print_history(session: ChatSession, store: ChatStore) -> bool:
    """Fetch and display all persisted messages for the current conversation."""
    messages = await store.list_messages(session.conversation_id)
    if not messages:
        console.print("[dim]No messages yet.[/]")
        return True
    for index, message in enumerate(messages, start=1):
        role = message["role"]
        console.print(Panel(message["content"] or "", title=f"{index}. {role}"))
    return True


async def print_request(raw: str, store: ChatStore) -> bool:
    """Parse '/request <id>' and display the stored LLM request metadata."""
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        console.print("[yellow]Usage:[/] /request <request_id>")
        return True

    request = await store.get_request(parts[1])
    if request is None:
        console.print(f"[yellow]No request found for:[/] {parts[1]}")
        return True

    table = Table(title="LLM Request", style="table.border", header_style="table.header")
    table.add_column("Field", style="highlight")
    table.add_column("Value")
    for key, value in request_to_rows(request):
        table.add_row(key, "" if value is None else str(value))
    console.print(table)
    return True


def clear_history(session: ChatSession) -> bool:
    """Clear the in-memory conversation history."""
    session.clear()
    console.print("[dim]History cleared.[/]")
    return True


def print_config(config: AppConfig) -> bool:
    """Render a table of the current sanitized runtime configuration."""
    table = Table(title="Runtime Config", style="table.border", header_style="table.header")
    table.add_column("Setting", style="highlight")
    table.add_column("Value")
    for key, value in sanitized_config(config).items():
        table.add_row(key, str(value))
    console.print(table)
    return True


async def unknown_command(raw: str) -> bool:
    """Handle unrecognized slash commands."""
    console.print(f"[yellow]Unknown command:[/] {raw}. Try /help.")
    return True
