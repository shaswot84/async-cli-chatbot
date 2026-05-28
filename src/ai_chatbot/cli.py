from __future__ import annotations

import asyncio
from collections.abc import Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_chatbot.config import AppConfig, load_config, sanitized_config
from ai_chatbot.llm_client import LLMClient, LLMClientError
from ai_chatbot.session import ChatSession

app = typer.Typer(add_completion=False, help="Asynchronous command-line AI chatbot.")
console = Console()


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

    session = ChatSession(config)
    client = LLMClient(config)
    commands = command_handlers(config, session)

    console.print(Panel.fit("Async CLI AI Chatbot\nType /help for commands.", title="Ready"))
    if debug:
        print_config(config)

    try:
        while True:
            try:
                user_input = console.input(f"[bold cyan]{session.active_model}>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/]")
                break

            if not user_input:
                continue
            if user_input.startswith("/"):
                should_continue = commands.get(command_name(user_input), unknown_command)(
                    user_input
                )
                if not should_continue:
                    break
                continue

            try:
                with console.status("Thinking...", spinner="dots"):
                    response = await session.send(client, user_input)
            except LLMClientError as exc:
                console.print(f"[red]Request failed:[/] {exc}")
                continue

            console.print(
                Panel(response.content, title=f"{response.model} · {response.latency_ms} ms")
            )
            if debug:
                token_text = (
                    f"tokens in/out/total: "
                    f"{response.input_tokens}/{response.output_tokens}/{response.total_tokens}"
                )
                console.print(f"[dim]{response.request_id} · {token_text}[/]")
    finally:
        await client.close()


def command_name(raw: str) -> str:
    parts = raw.split()
    if len(parts) >= 2 and parts[0] == "/model":
        return " ".join(parts[:2])
    return parts[0]


def command_handlers(config: AppConfig, session: ChatSession) -> dict[str, Callable[[str], bool]]:
    return {
        "/help": lambda _: print_help(),
        "/exit": lambda _: False,
        "/quit": lambda _: False,
        "/model list": lambda _: print_models(config, session),
        "/model current": lambda _: print_current_model(session),
        "/model set": lambda raw: set_model(raw, session),
        "/history": lambda _: print_history(session),
        "/clear": lambda _: clear_history(session),
        "/config": lambda _: print_config(config),
    }


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
        ("/history", "Show in-memory conversation"),
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


def print_history(session: ChatSession) -> bool:
    if not session.history:
        console.print("[dim]No messages yet.[/]")
        return True
    for index, message in enumerate(session.history, start=1):
        role = message["role"]
        console.print(Panel(message["content"], title=f"{index}. {role}"))
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


def unknown_command(raw: str) -> bool:
    console.print(f"[yellow]Unknown command:[/] {raw}. Try /help.")
    return True
