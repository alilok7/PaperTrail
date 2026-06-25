"""Score PaperTrail's verdicts against the hand-labeled answer key.

Usage:
  uv run python scripts/evaluate.py --ingest        # download+ingest paper & repo, then score
  uv run python scripts/evaluate.py                 # score using already-ingested data
  uv run python scripts/evaluate.py --no-follow-refs # skip reference-following (fewer API calls)
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import yaml

from paper_to_code.config import get_settings
from paper_to_code.models import Direction
from paper_to_code.service import PaperTrail

KEY_PATH = Path(__file__).resolve().parent.parent / "eval" / "answer_key.yaml"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "PaperTrail/0.1"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:
        fh.write(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Download + ingest paper and repo first.")
    parser.add_argument("--no-follow-refs", dest="follow_refs", action="store_false")
    args = parser.parse_args()

    key = yaml.safe_load(KEY_PATH.read_text(encoding="utf-8"))
    settings = get_settings()
    service = PaperTrail(settings)

    if args.ingest:
        pdf = Path(settings.papers_dir) / f"{key['paper']['id']}.pdf"
        if not pdf.exists():
            print(f"Downloading paper -> {pdf}")
            _download(key["paper"]["pdf_url"], pdf)
        print("Ingesting paper…")
        pr = service.ingest_paper(pdf, key["paper"]["id"])
        print(f"  {len(pr.chunks)} paper chunks")
        print("Ingesting code…")
        cr = service.ingest_code(key["code"]["repo_url"], key["code"]["id"])
        print(f"  {len(cr.chunks)} code chunks")

    print(f"\nIndexed: {service.counts()}\n")

    passed = 0
    for case in key["cases"]:
        result = service.ask(
            case["question"], direction=Direction(case["direction"]), follow_refs=args.follow_refs
        )
        verdict = result.verdict
        verdict_ok = verdict.verdict.value in case["expected_verdicts"]

        want_symbol = case.get("expect_code_symbol")
        if want_symbol:
            seen = [h.chunk.metadata.get("symbol", "") for h in result.code_evidence]
            if verdict.code_citation:
                seen.append(verdict.code_citation.symbol or "")
            symbol_ok = any(want_symbol in (s or "") for s in seen)
        else:
            symbol_ok = True

        ok = verdict_ok and symbol_ok
        passed += int(ok)
        cited = verdict.code_citation.symbol if verdict.code_citation else None
        print(f"[{'PASS' if ok else 'FAIL'}] {case['question']}")
        print(f"       verdict={verdict.verdict.value} expected={case['expected_verdicts']} (ok={verdict_ok})")
        print(f"       expect_symbol={want_symbol} cited={cited} (ok={symbol_ok}) conf={verdict.confidence:.2f}")
        print(f"       {verdict.explanation[:160]}")
        print()

    print(f"Score: {passed}/{len(key['cases'])}")
    return 0 if passed == len(key["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
