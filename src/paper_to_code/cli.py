"""PaperTrail command-line interface.

`check` is implemented here; `ingest` and `ask` are added once the agent loop exists.
Heavy imports are done lazily inside commands so `p2c --help` stays fast.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    help="PaperTrail — reconcile an ML paper with its implementation and return a cited verdict.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """PaperTrail — paper-to-code reproducibility agent."""


def _status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


@app.command()
def check() -> None:
    """Verify configuration and that Bedrock + Voyage + LangSmith are wired up."""
    from paper_to_code.config import get_settings

    s = get_settings()
    typer.echo("PaperTrail check")
    typer.echo("----------------")
    typer.echo(f"Bedrock credentials : {_status(s.has_bedrock)}")
    typer.echo(f"Voyage API key      : {_status(s.has_voyage)}")
    typer.echo(f"LangSmith API key   : {_status(s.has_langsmith)}")
    typer.echo(f"model={s.bedrock_model_id}  region={s.aws_region}  langsmith_project={s.langsmith_project}")
    typer.echo("")

    if s.has_bedrock:
        try:
            from paper_to_code.llm import get_chat_model

            content = str(get_chat_model(s).invoke("Reply with exactly one word: pong").content)
            ok = "pong" in content.lower()
            typer.echo(f"Bedrock live call   : {_status(ok)} -> {content.strip()[:40]!r}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Bedrock live call   : FAILED ({type(exc).__name__}: {str(exc)[:140]})")

    if s.has_voyage:
        try:
            from paper_to_code.embeddings.code_embedder import CodeEmbedder

            dim = len(CodeEmbedder().embed_query("scaled dot product attention"))
            typer.echo(f"Voyage live call    : {_status(dim > 0)} -> embedding dim={dim}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"Voyage live call    : FAILED ({type(exc).__name__}: {str(exc)[:140]})")

    if s.has_langsmith:
        from paper_to_code.llm import validate_langsmith

        ok, detail = validate_langsmith(s)
        typer.echo(f"LangSmith live call : {_status(ok)} -> {detail}")

    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
