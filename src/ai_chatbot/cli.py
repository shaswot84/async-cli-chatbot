from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_chatbot.config import AppConfig, load_config, sanitized_config
from ai_chatbot.db import ChatStore, request_to_rows
from ai_chatbot.failure_simulator import FailureSimulator, supported_failure_kinds
from ai_chatbot.llm_client import LLMClient, LLMClientError
from ai_chatbot.logging_setup import setup_logging
from ai_chatbot.session import ChatSession

app = typer.Typer(add_completion=False, help="Asynchronous command-line AI chatbot.")
console = Console()
logger = logging.getLogger(__name__)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Show sanitized runtime config on start."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(run_repl(debug=debug))


async def run_repl(debug: bool = False) -> None:
    try:
        config = load_config()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(2) from exc

    setup_logging(config.log_level, config.log_json)
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
    commands = command_handlers(config, session, store, failure_simulator)
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
                user_input = console.input(f"[bold cyan]{session.active_model}>[/] ").strip()
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

            try:
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

            console.print(
                Panel(response.content, title=f"{response.model} · {response.latency_ms} ms")
            )
            if debug:
                token_text = (
                    f"tokens in/out/total: "
                    f"{response.input_tokens}/{response.output_tokens}/{response.total_tokens}; "
                    f"retries: {response.retry_attempts}"
                )
                console.print(f"[dim]{response.request_id} · {token_text}[/]")
    finally:
        await client.close()
        await store.end_conversation(conversation_id)
        logger.info(
            "chat_session_ended",
            extra={"conversation_id": conversation_id},
        )


def command_name(raw: str) -> str:
    parts = raw.split()
    if len(parts) >= 2 and parts[0] == "/model":
        return " ".join(parts[:2])
    if len(parts) >= 2 and parts[0] == "/fail" and parts[1] in {"rate", "kind"}:
        return " ".join(parts[:2])
    if parts and parts[0] == "/fail":
        return "/fail"
    return parts[0]


def command_handlers(
    config: AppConfig,
    session: ChatSession,
    store: ChatStore,
    failure_simulator: FailureSimulator,
) -> dict[str, Callable[[str], Awaitable[bool]]]:
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
    }


async def async_value(value: bool) -> bool:
    return value


def print_help() -> bool:
    table = Table(title="Commands")
    table.add_column("Command", style="cyan")
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
        ("/clear", "Clear in-memory conversation"),
        ("/exit", "Quit"),
    ]
    for command, action in rows:
        table.add_row(command, action)
    console.print(table)
    return True


def print_models(config: AppConfig, session: ChatSession) -> bool:
    table = Table(title="Configured Models")
    table.add_column("Active")
    table.add_column("Model ID", style="cyan")
    table.add_column("Family")
    table.add_column("Use case")
    for model in config.models.values():
        table.add_row(
            "*" if model.model_id == session.active_model else "",
            model.model_id,
            model.family,
            model.use_case,
        )
    console.print(table)
    return True


def print_current_model(session: ChatSession) -> bool:
    console.print(f"Current model: [cyan]{session.active_model}[/]")
    return True


def set_model(raw: str, session: ChatSession) -> bool:
    parts = raw.split(maxsplit=2)
    if len(parts) < 3:
        console.print("[yellow]Usage:[/] /model set <model_id>")
        return True
    try:
        session.set_model(parts[2])
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        return True
    console.print(f"Switched to [cyan]{session.active_model}[/]")
    return True


def set_failure_enabled(raw: str, failure_simulator: FailureSimulator) -> bool:
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


async def print_history(session: ChatSession, store: ChatStore) -> bool:
    messages = await store.list_messages(session.conversation_id)
    if not messages:
        console.print("[dim]No messages yet.[/]")
        return True
    for index, message in enumerate(messages, start=1):
        role = message["role"]
        console.print(Panel(message["content"] or "", title=f"{index}. {role}"))
    return True


async def print_request(raw: str, store: ChatStore) -> bool:
    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        console.print("[yellow]Usage:[/] /request <request_id>")
        return True

    request = await store.get_request(parts[1])
    if request is None:
        console.print(f"[yellow]No request found for:[/] {parts[1]}")
        return True

    table = Table(title="LLM Request")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key, value in request_to_rows(request):
        table.add_row(key, "" if value is None else str(value))
    console.print(table)
    return True


def clear_history(session: ChatSession) -> bool:
    session.clear()
    console.print("[dim]History cleared.[/]")
    return True


def print_config(config: AppConfig) -> bool:
    table = Table(title="Runtime Config")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    for key, value in sanitized_config(config).items():
        table.add_row(key, str(value))
    console.print(table)
    return True


async def unknown_command(raw: str) -> bool:
    console.print(f"[yellow]Unknown command:[/] {raw}. Try /help.")
    return True
