"""PaperTrail — Streamlit UI (a thin, resilient layer over paper_to_code.service.PaperTrail).

Design goals for this UI:
- Never show a raw traceback or a half-rendered page: every service call is wrapped and
  failures become clean, actionable messages.
- Always show what's happening: ingestion runs inside a live status panel with a progress
  bar and rate-limit messages, so the user is never left wondering.
- Survive reruns: the last verdict is kept in session state, so using the sidebar (e.g.
  deleting a library item) never wipes the answer on screen.

Run with:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from paper_to_code.config import get_settings
from paper_to_code.errors import PaperTrailError
from paper_to_code.models import Direction, Verdict

st.set_page_config(page_title="PaperTrail", page_icon="📄", layout="wide")

_VERDICT_RENDER = {
    "MATCH": ("✅ MATCH", st.success),
    "MATCH_SIMPLIFIED": ("🟦 MATCH (simplified)", st.info),
    "EXTRA_IN_CODE": ("🟧 EXTRA IN CODE", st.warning),
    "NOT_IMPLEMENTED": ("🟥 NOT IMPLEMENTED", st.error),
    "UNCERTAIN": ("⬜ UNCERTAIN", st.warning),
}

_DIRECTIONS = {
    "Auto-detect": None,
    "Paper → Code": Direction.PAPER_TO_CODE,
    "Code → Paper": Direction.CODE_TO_PAPER,
}

_ALL_SCOPE = "All (no scope)"

# A couple of well-known public repos to make repo selection a one-click affair.
_EXAMPLE_REPOS = {
    "nanoGPT": "https://github.com/karpathy/nanoGPT",
    "vision_transformer": "https://github.com/google-research/vision_transformer",
}


@st.cache_resource(show_spinner=False)
def get_service():
    # Heavy import (torch/Docling/Chroma/LangChain) kept lazy so the page paints first.
    from paper_to_code.service import PaperTrail

    return PaperTrail()


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _run_with_progress(label: str, func):
    """Run ``func(on_progress)`` inside a live status panel with a progress bar.

    Returns ``(result, error)`` — exactly one is None. The status panel reflects the
    final outcome (complete / error) so the user always sees how it ended.
    """
    status = st.status(label, expanded=True)
    bar = status.progress(0.0)
    state = {"frac": 0.0}

    def on_progress(message: str, fraction):
        if fraction is not None:
            state["frac"] = max(0.0, min(1.0, float(fraction)))
        bar.progress(state["frac"], text=message)

    try:
        result = func(on_progress)
    except PaperTrailError as exc:
        status.update(label="Failed", state="error", expanded=True)
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - unexpected: show a clean message, not a stack trace
        status.update(label="Failed", state="error", expanded=True)
        return None, f"Unexpected error ({type(exc).__name__}): {exc}"
    status.update(label="Done", state="complete", expanded=False)
    return result, None


def _credential_banner(settings) -> None:
    """Warn up front if a credential needed for ingest/ask is missing."""
    missing = []
    if not settings.has_voyage:
        missing.append("**Voyage** (needed to ingest & embed)")
    if not settings.has_bedrock:
        missing.append("**Bedrock** (needed to answer questions)")
    if missing:
        st.warning(
            "Some credentials are not configured: "
            + ", ".join(missing)
            + ". Add them to `.env` and restart. Run `uv run p2c check` to verify."
        )


# ──────────────────────────────────────────────────────────────────────────
# Sidebar sections
# ──────────────────────────────────────────────────────────────────────────

def _render_ingest(service, settings) -> None:
    st.header("📥 Ingest")
    counts = service.counts()
    st.caption(
        f"Indexed: {counts['paper_chunks']} paper chunks · {counts['code_chunks']} code chunks"
    )

    can_ingest = settings.has_voyage
    if not can_ingest:
        st.info("Add a Voyage API key to `.env` to enable ingestion.")

    # -- Paper ---------------------------------------------------------------
    st.subheader("Paper (PDF)")
    uploaded = st.file_uploader("Upload a paper PDF", type=["pdf"], key="pdf_uploader")
    if st.button("Ingest paper", disabled=(uploaded is None or not can_ingest), use_container_width=True):
        try:
            dest = Path(settings.papers_dir) / uploaded.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getbuffer())
        except OSError as exc:
            st.error(f"Could not save the upload: {exc}")
        else:
            result, error = _run_with_progress(
                f"Ingesting {uploaded.name} …",
                lambda cb: service.ingest_paper(dest, on_progress=cb),
            )
            if error:
                st.error(error)
            elif not result.chunks:
                st.warning(
                    f"No text could be extracted from '{uploaded.name}'. "
                    "It may be image-only or unsupported."
                )
            else:
                st.success(f"Stored {len(result.chunks)} paper chunks as “{result.paper_id}”.")

    # -- Repo ----------------------------------------------------------------
    st.subheader("Code (repo)")
    st.caption("Add a **new** repo: paste a public GitHub URL or a local folder path.")

    # Quick-fill chips, only for examples you haven't already ingested (avoids the
    # "this is already in my library, why does it want to ingest?" confusion).
    already = set(service.list_library()["repos"])
    examples = {n: u for n, u in _EXAMPLE_REPOS.items() if n not in already}
    if examples:
        st.caption("Or quick-fill an example into the box below:")
        cols = st.columns(len(examples))
        for col, (name, url) in zip(cols, examples.items()):
            if col.button(f"＋ {name}", key=f"ex_{name}", use_container_width=True, help=f"Fill in {url}"):
                st.session_state["repo_input"] = url

    repo = st.text_input(
        "Git URL or local path",
        key="repo_input",
        placeholder="https://github.com/karpathy/nanoGPT",
    )
    if st.button("Ingest repo", disabled=(not repo or not can_ingest), use_container_width=True):
        result, error = _run_with_progress(
            f"Ingesting {repo} …",
            lambda cb: service.ingest_code(repo.strip(), on_progress=cb),
        )
        if error:
            st.error(error)
        elif not result.chunks:
            st.warning(
                "No supported source files were found in that repository "
                "(supported: Python, JS/TS, Java, Go, Rust, C/C++, C#, Ruby, CUDA)."
            )
        else:
            st.success(f"Stored {len(result.chunks)} code chunks as “{result.repo_id}”.")


def _render_library(service) -> dict:
    """Render the library list with delete buttons. Returns the library dict."""
    st.header("📚 Library")
    st.caption("Already ingested — these persist across restarts. 🗑 removes one.")
    lib = service.list_library()
    papers, repos = lib["papers"], lib["repos"]

    st.subheader("Papers")
    if papers:
        for pid, cnt in sorted(papers.items()):
            name_col, btn_col = st.columns([4, 1])
            name_col.markdown(f"**{pid}**  \n<small>{cnt} chunks</small>", unsafe_allow_html=True)
            if btn_col.button("🗑", key=f"del_paper_{pid}", help=f"Remove paper '{pid}'"):
                n = service.remove_paper(pid)
                st.toast(f"Removed paper '{pid}' ({n} chunks).", icon="🗑")
                st.rerun()
    else:
        st.caption("No papers ingested yet.")

    st.subheader("Repos")
    if repos:
        for rid, cnt in sorted(repos.items()):
            name_col, btn_col = st.columns([4, 1])
            name_col.markdown(f"**{rid}**  \n<small>{cnt} chunks</small>", unsafe_allow_html=True)
            if btn_col.button("🗑", key=f"del_repo_{rid}", help=f"Remove repo '{rid}'"):
                n = service.remove_repo(rid)
                st.toast(f"Removed repo '{rid}' ({n} chunks).", icon="🗑")
                st.rerun()
    else:
        st.caption("No repos ingested yet.")

    return lib


def _render_scope(lib: dict) -> tuple[str | None, str | None]:
    """Active paper/repo selectboxes (with a stale-selection guard). Returns resolved ids."""
    st.header("🎯 Active scope")
    st.caption("Limit retrieval to one paper/repo, or search everything.")

    paper_options = [_ALL_SCOPE] + sorted(lib["papers"].keys())
    repo_options = [_ALL_SCOPE] + sorted(lib["repos"].keys())

    # Reset a selection that points at something that was just deleted.
    if st.session_state.get("active_paper") not in paper_options:
        st.session_state["active_paper"] = _ALL_SCOPE
    if st.session_state.get("active_repo") not in repo_options:
        st.session_state["active_repo"] = _ALL_SCOPE

    active_paper = st.selectbox(
        "Active paper", paper_options, index=paper_options.index(st.session_state["active_paper"])
    )
    active_repo = st.selectbox(
        "Active repo", repo_options, index=repo_options.index(st.session_state["active_repo"])
    )
    st.session_state["active_paper"] = active_paper
    st.session_state["active_repo"] = active_repo

    return (
        active_paper if active_paper != _ALL_SCOPE else None,
        active_repo if active_repo != _ALL_SCOPE else None,
    )


# ──────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────

def render_verdict(result, scope_label: str) -> None:
    verdict: Verdict = result.verdict
    label, renderer = _VERDICT_RENDER.get(verdict.verdict.value, (verdict.verdict.value, st.info))
    renderer(f"**{label}**  ·  confidence {verdict.confidence:.0%}  ·  {result.direction.value}")

    if scope_label:
        st.caption(f"🎯 Scope — {scope_label}")

    st.markdown(f"**Explanation**\n\n{verdict.explanation}")
    st.caption(f"concept: {result.concept}  ·  reformulated query: {result.reformulated_query}")

    paper_col, code_col = st.columns(2)
    with paper_col:
        st.subheader("📄 Paper")
        pc = verdict.paper_citation
        if pc:
            st.markdown(f"**section** {pc.section or '—'}  ·  **page** {pc.page or '—'}")
            if pc.equations:
                st.markdown(f"equations: {', '.join(pc.equations)}")
            if pc.figures:
                st.markdown(f"figures: {', '.join(pc.figures)}")
            if pc.quote:
                st.markdown(f"> {pc.quote}")
        else:
            st.markdown("_No matching paper passage._")
    with code_col:
        st.subheader("💻 Code")
        cc = verdict.code_citation
        if cc:
            st.markdown(f"**{cc.file}** · `{cc.symbol or '—'}` · lines {cc.start_line}–{cc.end_line}")
            if cc.quote:
                st.code(cc.quote, language=cc.language or "python")
        else:
            st.markdown("_Not implemented in the code._")

    with st.expander("Retrieved evidence (raw chunks)"):
        st.markdown("**Paper chunks**")
        if result.paper_evidence:
            for h in result.paper_evidence:
                st.caption(f"{h.chunk.id}  ·  score {h.score:.3f}")
                st.text(h.chunk.text[:800])
        else:
            st.caption("— none —")
        st.markdown("**Code chunks**")
        if result.code_evidence:
            for h in result.code_evidence:
                st.caption(f"{h.chunk.id}  ·  score {h.score:.3f}")
                st.code(h.chunk.text[:800], language=h.chunk.metadata.get("language", "python"))
        else:
            st.caption("— none —")


def _render_ask(service, settings, paper_id, repo_id, scope_label: str) -> None:
    if scope_label:
        st.info(f"🎯 Scoped to — {scope_label}")
    else:
        st.caption("Searching across everything ingested. Pick an active paper/repo in the sidebar to scope a pair.")

    with st.form("ask_form"):
        question = st.text_input(
            "Ask a question",
            placeholder="How is multi-head attention from Section 3 implemented in the code?",
        )
        col_a, col_b, col_c = st.columns([2, 1, 1])
        direction_label = col_a.selectbox("Direction", list(_DIRECTIONS.keys()))
        k = col_b.slider("Top-k per side", 3, 15, settings.retrieval_top_k)
        follow_refs = col_c.checkbox("Follow references", value=True)
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Type a question first.")
        elif not settings.has_bedrock:
            st.error("Bedrock is not configured — add credentials to `.env` to answer questions.")
        else:
            with st.spinner("Retrieving both sides and reconciling …"):
                try:
                    result = service.ask(
                        question,
                        direction=_DIRECTIONS[direction_label],
                        k=k,
                        follow_refs=follow_refs,
                        paper_id=paper_id,
                        repo_id=repo_id,
                    )
                except PaperTrailError as exc:
                    st.error(str(exc))
                    result = None
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not answer the question ({type(exc).__name__}): {exc}")
                    result = None
            if result is not None:
                st.session_state["last_result"] = result
                st.session_state["last_scope"] = scope_label

    # Render the most recent verdict (persists across sidebar-triggered reruns).
    if st.session_state.get("last_result") is not None:
        st.divider()
        render_verdict(st.session_state["last_result"], st.session_state.get("last_scope", ""))


# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📄 PaperTrail")
    st.caption("Reconcile an ML paper with the code that implements it — with a cited verdict.")

    settings = get_settings()

    # Start the engine. A startup failure shows a clean card instead of a broken page.
    try:
        with st.spinner("Loading models and index… first load can take ~30s"):
            service = get_service()
    except Exception as exc:  # noqa: BLE001
        st.error("PaperTrail could not start. Check your configuration and try again.")
        with st.expander("Details"):
            st.write(f"{type(exc).__name__}: {exc}")
        st.stop()

    _credential_banner(settings)

    with st.sidebar:
        _render_ingest(service, settings)
        st.divider()
        lib = _render_library(service)
        st.divider()
        paper_id, repo_id = _render_scope(lib)

    scope_parts = []
    if paper_id:
        scope_parts.append(f"paper: {paper_id}")
    if repo_id:
        scope_parts.append(f"repo: {repo_id}")
    scope_label = " · ".join(scope_parts)

    if not lib["papers"] and not lib["repos"]:
        st.info("👋 Start by ingesting a paper (PDF) and the repo that implements it from the sidebar.")

    _render_ask(service, settings, paper_id, repo_id, scope_label)


if __name__ == "__main__":
    main()
