"""PaperTrail command-line interface.

Commands are filled in as the pipeline is built. For now this provides the
entry point (`p2c`) and a placeholder `check` command.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    help="PaperTrail — reconcile an ML paper with its implementation and return a cited verdict.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def check() -> None:
    """Verify configuration and credentials. (Implemented in the LLM-layer step.)"""
    typer.echo("check: not implemented yet")


if __name__ == "__main__":
    app()
