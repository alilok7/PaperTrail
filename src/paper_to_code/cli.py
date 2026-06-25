"""PaperTrail command-line interface.

Commands:
    p2c check                         verify credentials are wired up
    p2c ingest-paper PATH [--id ID]   parse + embed + store a paper PDF
    p2c ingest-code SRC [--id ID]     clone/scan + embed + store a repo (URL or local path)
    p2c ask "QUESTION" [--direction]  retrieve both sides, reconcile, print the verdict

Heavy imports are done lazily inside commands so `p2c --help` stays fast.
"""

from __future__ import annotations

import sys

import typer

# Windows consoles default to cp1252 and crash printing math symbols (√, ×, …) that appear
# in verdict explanations. Force UTF-8 output, replacing anything truly unencodable.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

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


def _parse_direction(value: str | None):
    from paper_to_code.models import Direction

    if value is None or value.lower() in ("auto", ""):
        return None
    key = value.lower().replace("_", "-")
    mapping = {
        "paper-to-code": Direction.PAPER_TO_CODE,
        "paper2code": Direction.PAPER_TO_CODE,
        "code-to-paper": Direction.CODE_TO_PAPER,
        "code2paper": Direction.CODE_TO_PAPER,
    }
    if key not in mapping:
        raise typer.BadParameter("direction must be one of: auto, paper-to-code, code-to-paper")
    return mapping[key]


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
            typer.echo(f"Bedrock live call   : {_status('pong' in content.lower())} -> {content.strip()[:40]!r}")
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


@app.command("ingest-paper")
def ingest_paper_cmd(
    pdf_path: str = typer.Argument(..., help="Path to the paper PDF."),
    paper_id: str = typer.Option(None, "--id", help="Identifier for the paper (defaults to file stem)."),
) -> None:
    """Parse, embed, and store a paper PDF."""
    from paper_to_code.service import PaperTrail

    typer.echo(f"Ingesting paper: {pdf_path} (first run downloads Docling models)...")
    result = PaperTrail().ingest_paper(pdf_path, paper_id)
    typer.echo(f"Stored {len(result.chunks)} paper chunks for '{result.paper_id}'.")


@app.command("ingest-code")
def ingest_code_cmd(
    source: str = typer.Argument(..., help="Git URL or local path to the repository."),
    repo_id: str = typer.Option(None, "--id", help="Identifier for the repo (defaults to repo name)."),
) -> None:
    """Clone/scan, embed, and store a code repository."""
    from paper_to_code.service import PaperTrail

    typer.echo(f"Ingesting code: {source} ...")
    result = PaperTrail().ingest_code(source, repo_id)
    typer.echo(f"Stored {len(result.chunks)} code chunks for '{result.repo_id}' at {result.repo_path}.")


def _print_verdict(result) -> None:
    v = result.verdict
    typer.echo("")
    typer.echo(f"VERDICT : {v.verdict.value}   confidence={v.confidence:.2f}   direction={result.direction.value}")
    typer.echo(f"concept : {result.concept}")
    typer.echo(f"code query (reformulated): {result.reformulated_query}")
    typer.echo("")
    typer.echo(v.explanation)
    typer.echo("")
    if v.paper_citation:
        pc = v.paper_citation
        typer.echo(f"PAPER   : section={pc.section} page={pc.page} eq={pc.equations} fig={pc.figures}")
        if pc.quote:
            typer.echo(f'          "{pc.quote[:280]}"')
    else:
        typer.echo("PAPER   : (none)")
    if v.code_citation:
        cc = v.code_citation
        typer.echo(f"CODE    : {cc.file}::{cc.symbol} lines {cc.start_line}-{cc.end_line}")
        if cc.quote:
            typer.echo(f"          {cc.quote[:280]}")
    else:
        typer.echo("CODE    : (none)")
    typer.echo("")
    typer.echo(f"paper evidence ids: {[h.chunk.id for h in result.paper_evidence]}")
    typer.echo(f"code evidence ids : {[h.chunk.id for h in result.code_evidence]}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question about the paper/code."),
    direction: str = typer.Option("auto", "--direction", help="auto | paper-to-code | code-to-paper"),
    k: int = typer.Option(8, "--k", help="Top-k chunks per side."),
    follow_refs: bool = typer.Option(True, "--follow-refs/--no-follow-refs", help="Single-hop reference following."),
) -> None:
    """Ask how a concept maps between the paper and the code, and print the verdict."""
    from paper_to_code.service import PaperTrail

    result = PaperTrail().ask(
        question, direction=_parse_direction(direction), k=k, follow_refs=follow_refs
    )
    _print_verdict(result)


if __name__ == "__main__":
    app()
