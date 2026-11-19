"""
Assignment 3: evaluation (`main.py evaluate`).

Stage 1, retrieval: hit rate at k = 1, 3, 5 for every chunking design and
search method, on the answerable questions of eval/questions.csv.
Stage 2, end to end: agent answers, code checks, LLM judge, agreement with
eval/human_labels.csv (added in Phase 7).

    uv run python evaluate.py                         # everything
    uv run python evaluate.py --stage retrieval       # only hit rates
    uv run python evaluate.py --questions other.csv   # another test set

A hit for a question = a retrieved chunk from the expected regulation that
covers the expected article/annex (paragraph numbers are ignored: a chunk of
Article 33 counts for "Article 33(1)"). For `multi` questions, the strict hit
needs every expected passage in the top k; the lenient one needs at least one.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

from ingest import DESIGNS
from retrieval import METHODS, search

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
QUESTIONS = EVAL / "questions.csv"
KS = (1, 3, 5)

sys.stdout.reconfigure(encoding="utf-8")


def read_questions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("question") or "").strip()]
    for r in rows:
        r["kind"] = r["kind"].strip().lower()
        sources = [s.strip() for s in r["expected_source"].split(";") if s.strip()]
        locations = [l.strip() for l in r["expected_location"].split(";") if l.strip()]
        if len(sources) == 1 and len(locations) > 1:
            sources = sources * len(locations)
        r["expected"] = [(s, article_key(l)) for s, l in zip(sources, locations)]
    return rows


def article_key(location: str) -> str:
    """'Article 33(1)' -> 'Article 33'; 'Annex III, point 4' -> 'Annex III'."""
    m = re.match(r"\s*(Article\s+\d+[a-z]?|Annex\s+[IVXLC]+)", location, re.I)
    if not m:
        return location.strip()
    word, num = m.group(1).split()
    return f"{word.capitalize()} {num if word.lower() == 'article' else num.upper()}"


def first_rank(hits: list[dict], source: str, key: str) -> int | None:
    """1-based rank of the first hit from `source` that covers `key`, or None."""
    for rank, h in enumerate(hits, 1):
        if h["doc"] == source and key in h["covers"].split("|"):
            return rank
    return None


def retrieval_eval(questions: list[dict]) -> pd.DataFrame:
    answerable = [q for q in questions if q["kind"] != "unanswerable"]
    rows, scores = [], []
    for design in DESIGNS:
        for method in METHODS:
            for q in questions:
                hits = search(q["question"], design=design, method=method, k=max(KS))
                if method == "vector":
                    scores.append({"design": design, "id": q["id"], "kind": q["kind"],
                                   "top1_score": hits[0]["score"] if hits else None,
                                   "top1": f"{hits[0]['doc']} {hits[0]['location']}" if hits else ""})
                if q["kind"] == "unanswerable":
                    continue
                ranks = [first_rank(hits, s, key) for s, key in q["expected"]]
                row = {"design": design, "method": method, "id": q["id"], "kind": q["kind"],
                       "ranks": ";".join("-" if r is None else str(r) for r in ranks),
                       "retrieved": " | ".join(f"{h['doc']} {h['location']}" for h in hits)}
                for k in KS:
                    found = [r is not None and r <= k for r in ranks]
                    row[f"hit@{k}"] = int(all(found))
                    row[f"any@{k}"] = int(any(found))
                    row[f"words@{k}"] = sum(h["words"] for h in hits[:k])
                rows.append(row)
            print(f"  {design:9} {method:7} done", file=sys.stderr, flush=True)

    per_q = pd.DataFrame(rows)
    per_q.to_csv(EVAL / "retrieval_per_question.csv", index=False)
    pd.DataFrame(scores).to_csv(EVAL / "retrieval_scores.csv", index=False)

    cols = [f"hit@{k}" for k in KS] + [f"any@{k}" for k in KS] + [f"words@{k}" for k in KS]
    summary = per_q.groupby(["design", "method"])[cols].mean().round(3).reset_index()
    multi = per_q[per_q.kind == "multi"].groupby(["design", "method"])[[f"hit@{k}" for k in KS]].mean().round(3)
    multi.columns = [f"multi_{c}" for c in multi.columns]
    summary = summary.merge(multi.reset_index(), on=["design", "method"], how="left")
    summary.insert(2, "n", len(answerable))
    summary.to_csv(EVAL / "retrieval_summary.csv", index=False)
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["retrieval", "e2e", "all"], default="all")
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    a = p.parse_args()

    questions = read_questions(a.questions)
    kinds = pd.Series([q["kind"] for q in questions]).value_counts().to_dict()
    print(f"{len(questions)} questions: {kinds}")

    if a.stage in ("retrieval", "all"):
        print("\n== Retrieval: hit rate at k (strict for multi: every expected passage in top k)")
        summary = retrieval_eval(questions)
        print(summary.to_string(index=False))

    if a.stage in ("e2e", "all"):
        print("\n== End-to-end evaluation: not implemented yet (Phase 7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
