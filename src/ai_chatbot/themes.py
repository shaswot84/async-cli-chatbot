"""Predefined colour themes for the chat terminal UI."""

from __future__ import annotations

from typing import Any

from rich.style import Style
from rich.theme import Theme

# Each theme defines named styles used throughout the CLI.
# Style names follow a convention:
#   prompt          — the input prompt (model name)
#   response        — the assistant response panel border
#   response.title  — the panel title bar
#   info            — informational messages
#   success         — success / confirmation messages
#   warning         — warning messages
#   error           — error messages
#   dim             — muted / secondary text
#   debug           — debug output line
#   help.border     — help table border
#   table.border    — generic table border
#   table.header    — table header row
#   highlight       — bright accent for active/selected items

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "prompt": "bold cyan",
        "response": "cyan",
        "response.title": "bold cyan",
        "info": "dim white",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "dim": "dim white",
        "debug": "dim white",
        "help.border": "cyan",
        "table.border": "cyan",
        "table.header": "bold cyan",
        "highlight": "bold cyan",
    },
    "light": {
        "prompt": "bold blue",
        "response": "blue",
        "response.title": "bold blue",
        "info": "dim black",
        "success": "green",
        "warning": "dark_orange",
        "error": "red",
        "dim": "dim black",
        "debug": "dim black",
        "help.border": "blue",
        "table.border": "blue",
        "table.header": "bold blue",
        "highlight": "bold blue",
    },
    "monokai": {
        "prompt": "bold #a6e22e",
        "response": "#66d9ef",
        "response.title": "bold #66d9ef",
        "info": "#75715e",
        "success": "#a6e22e",
        "warning": "#e6db74",
        "error": "#f92672",
        "dim": "#75715e",
        "debug": "#75715e",
        "help.border": "#f92672",
        "table.border": "#f92672",
        "table.header": "bold #f92672",
        "highlight": "bold #a6e22e",
    },
    "dracula": {
        "prompt": "bold #bd93f9",
        "response": "#8be9fd",
        "response.title": "bold #8be9fd",
        "info": "#6272a4",
        "success": "#50fa7b",
        "warning": "#f1fa8c",
        "error": "#ff5555",
        "dim": "#6272a4",
        "debug": "#6272a4",
        "help.border": "#ff79c6",
        "table.border": "#ff79c6",
        "table.header": "bold #ff79c6",
        "highlight": "bold #bd93f9",
    },
    "nord": {
        "prompt": "bold #88c0d0",
        "response": "#81a1c1",
        "response.title": "bold #81a1c1",
        "info": "#616e88",
        "success": "#a3be8c",
        "warning": "#ebcb8b",
        "error": "#bf616a",
        "dim": "#616e88",
        "debug": "#616e88",
        "help.border": "#5e81ac",
        "table.border": "#5e81ac",
        "table.header": "bold #5e81ac",
        "highlight": "bold #88c0d0",
    },
    "gruvbox": {
        "prompt": "bold #b8bb26",
        "response": "#83a598",
        "response.title": "bold #83a598",
        "info": "#928374",
        "success": "#b8bb26",
        "warning": "#fabd2f",
        "error": "#fb4934",
        "dim": "#928374",
        "debug": "#928374",
        "help.border": "#d3869b",
        "table.border": "#d3869b",
        "table.header": "bold #d3869b",
        "highlight": "bold #b8bb26",
    },
}


def get_theme(name: str) -> Theme:
    """Return a Rich Theme for the given name. Falls back to 'dark' on unknown names."""
    styles = THEMES.get(name, THEMES["dark"])
    return Theme({key: Style.parse(value) for key, value in styles.items()})


def available_themes() -> list[str]:
    """Return a sorted list of available theme names."""
    return sorted(THEMES.keys())


def resolve_theme_name(name: str) -> str:
    """Return the canonical theme name, falling back to 'dark'."""
    if name in THEMES:
        return name
    return "dark"
