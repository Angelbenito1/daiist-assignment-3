# AI use log

Running log of what an AI assistant (Claude Code, model Claude Opus 5.5) generated or helped with, for the AI-use disclosure in REPORT.md. Decisions, the test set, human grades and the report's analysis are written by the student.

| Date | Phase | What the AI did | What the student did |
|---|---|---|---|
| 2026-11-18 | 0. Setup | Ran `uv sync`; inspected the four EUR-Lex HTML files (titles, raw word counts); drafted `corpus/README.md` (source, date, licence, language); created this log; adjusted `.gitignore` so `corpus/README.md` is committed. | Chose English as corpus language; set the deadline; created `.env` with own key; forked the repo; makes all commits. |
| 2026-11-18 | 1. Corpus and budget | Inspected EUR-Lex HTML structure; wrote the parser in `ingest.py` (units with source/location, definition-point splitting, footnote removal, Unicode cleaning); computed word counts per section and chunk/embedding estimates per design and scope. | Chose scope B (articles + annexes, no recitals). |
| 2026-11-18 | 2. Ingestion | Wrote the chunking designs (fixed window, per article, per paragraph), the SQLite embedding cache keyed by content hash, ChromaDB storage and `index.py`; found and fixed a batching bug (gemini-embedding-2 returned one vector per batch). | Chose to compare all three designs. |
| 2026-11-19 | 2–3 | Finished embedding the fixed design; fixed index growth (skip already-stored chunks); wrote `lookup.py` (read-only corpus explorer) and a test-set guide (format, mix, checklist, candidate topics). | Writes all test questions and reference answers. |
