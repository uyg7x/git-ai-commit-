"""Main CLI entry point for git-ai-commit.

Uses Typer for the CLI framework and Rich for terminal output.
"""

from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .ai_engine import AIEngine
from .config import Config
from .git_utils import get_git_status, commit_changes

# Create Rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="git-ai-commit",
    help="AI-powered conventional commit message generator",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version", is_eager=True),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="AI provider (openai/ollama)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="AI model"),
) -> None:
    """Generate a conventional commit message using AI.
    
    Run this in a git repository with staged changes.
    """
    # Show version and exit
    if version:
        console.print(f"git-ai-commit v{__version__}")
        raise typer.Exit()

    # If a subcommand was invoked, don't run the callback logic
    if ctx.invoked_subcommand is not None:
        return

    # Load configuration
    config = Config.from_env()

    # Override config with CLI options
    if provider is not None and provider != "":
        config.ai_provider = provider
    if model is not None and model != "":
        if config.ai_provider == "openai":
            config.openai_model = model
        else:
            config.ollama_model = model

    # Validate configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        console.print(f"[red]Configuration error:[/red] {error_msg}")
        raise typer.Exit(1)

    # Get git status
    status = get_git_status()

    # Handle errors
    if not status.is_repo:
        console.print(f"[red]Error:[/red] {status.error}")
        raise typer.Exit(1)

    if not status.has_staged_changes:
        console.print(f"[yellow]No staged changes found[/yellow]")
        if status.branch:
            console.print(f"Current branch: [blue]{status.branch}[/blue]")
        console.print("Stage your changes with: [green]git add <files>[/green]")
        raise typer.Exit(1)

    # Show status
    if status.branch:
        console.print(f"Current branch: [blue]{status.branch}[/blue]")
    console.print(f"Analyzing [green]{len(status.staged_diff)}[/green] bytes of staged changes...")

    # Generate suggestion
    engine = AIEngine(config)

    with console.status("[bold green]Generating commit message...") as _:
        success, result = engine.generate_commit_message(status.staged_diff)

    if not success:
        console.print(f"[red]Failed to generate commit message:[/red] {result}")
        raise typer.Exit(1)

    # Display suggestion
    console.print()
    suggestion = result

    panel = Panel(
        Text(suggestion, style="bold cyan"),
        title="[bold green]Suggested Commit Message[/bold green]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)

    # Prompt user for action
    console.print()
    action = Confirm.ask(
        "Would you like to accept this commit message?",
        choices=["y", "n"],
        default=True,
    )

    if action:
        # Accept and commit
        success, msg = commit_changes(suggestion)
        if success:
            console.print(f"[bold green]✓[/bold green] {msg}")
        else:
            console.print(f"[red]Error:[/red] {msg}")
            raise typer.Exit(1)
    else:
        console.print("[yellow]Commit cancelled.[/yellow] You can commit manually with:")
        console.print(f"[blue]git commit -m \"{suggestion}\"[/blue]")


@app.command()
def gac(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="AI provider"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="AI model"),
) -> None:
    """Alias for git-ai-commit command."""
    # Re-run the callback logic
    callback(typer.Context(app), version=False, provider=provider, model=model)


def run() -> None:
    """Run the application."""
    app()


if __name__ == "__main__":
    run()
