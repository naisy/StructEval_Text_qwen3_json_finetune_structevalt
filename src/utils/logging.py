from __future__ import annotations

from rich.console import Console

console = Console()


def info(msg: str) -> None:
    console.print(f"[bold cyan]INFO[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]WARN[/] {msg}")


def err(msg: str) -> None:
    console.print(f"[bold red]ERR[/] {msg}")
