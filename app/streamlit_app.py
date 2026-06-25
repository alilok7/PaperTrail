"""PaperTrail — Streamlit UI (a thin layer over paper_to_code.service.PaperTrail).

Run with:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from paper_to_code.config import get_settings
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


@st.cache_resource(show_spinner=False)
def get_service():
    # Heavy import (torch/Docling/Chroma/LangChain) kept lazy so the page paints first.
    from paper_to_code.service import PaperTrail

    return PaperTrail()


def render_verdict(result) -> None:
    verdict: Verdict = result.verdict
    label, renderer = _VERDICT_RENDER.get(verdict.verdict.value, (verdict.verdict.value, st.info))
    renderer(f"**{label}**  ·  confidence {verdict.confidence:.0%}  ·  {result.direction.value}")

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
        for h in result.paper_evidence:
            st.caption(f"{h.chunk.id}  ·  score {h.score:.3f}")
            st.text(h.chunk.text[:800])
        st.markdown("**Code chunks**")
        for h in result.code_evidence:
            st.caption(f"{h.chunk.id}  ·  score {h.score:.3f}")
            st.code(h.chunk.text[:800], language=h.chunk.metadata.get("language", "python"))


def main() -> None:
    st.title("📄 PaperTrail")
    st.caption("Reconcile an ML paper with the code that implements it — with a cited verdict.")

    settings = get_settings()
    with st.spinner("Loading models and index… first load can take ~30s"):
        service = get_service()

    with st.sidebar:
        st.header("Ingest")
        counts = service.counts()
        st.caption(f"Indexed: {counts['paper_chunks']} paper chunks · {counts['code_chunks']} code chunks")

        st.subheader("Paper (PDF)")
        uploaded = st.file_uploader("Upload a paper PDF", type=["pdf"])
        if st.button("Ingest paper", disabled=uploaded is None):
            dest = Path(settings.papers_dir) / uploaded.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uploaded.getbuffer())
            with st.spinner("Parsing + embedding the paper… (first run downloads Docling models)"):
                result = service.ingest_paper(dest)
            st.success(f"Stored {len(result.chunks)} paper chunks ({result.paper_id}).")

        st.subheader("Code (repo)")
        repo = st.text_input("Git URL or local path", placeholder="https://github.com/karpathy/nanoGPT")
        if st.button("Ingest repo", disabled=not repo):
            with st.spinner("Cloning + embedding the repo…"):
                result = service.ingest_code(repo)
            st.success(f"Stored {len(result.chunks)} code chunks ({result.repo_id}).")

    question = st.text_input(
        "Ask a question",
        placeholder="How is multi-head attention from Section 3 implemented in the code?",
    )
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        direction_label = st.selectbox("Direction", list(_DIRECTIONS.keys()))
    with col_b:
        k = st.slider("Top-k per side", 3, 15, settings.retrieval_top_k)
    with col_c:
        follow_refs = st.checkbox("Follow references", value=True)

    if st.button("Ask", type="primary", disabled=not question):
        with st.spinner("Retrieving both sides and reconciling…"):
            result = service.ask(
                question, direction=_DIRECTIONS[direction_label], k=k, follow_refs=follow_refs
            )
        render_verdict(result)


if __name__ == "__main__":
    main()
