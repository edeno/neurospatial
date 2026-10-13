---
paths:
  - "**/*.md"
---

# Executable Markdown

`tests/docs/test_executable_docs.py` executes Python fences in the documents it
lists: every fence in `RUN_ALL` files, and only marked fences in `OPT_IN` files.
`README.md` and `docs/getting-started/quickstart.md` run cumulatively, so each
document must import what it uses. Marked fences elsewhere run in a fresh
namespace built by `_fixture_namespace`.

Put a marker on the line directly above the fence:
`<!-- docs-test: run -->`, `<!-- docs-test: run setup=<name> -->` (a key of
`SETUPS`; every setup must be used), `<!-- docs-test: skip <reason> -->` or
`<!-- docs-test: raises <Exception> -->`. NWB recipes in
`docs/user-guide/interoperability.md` use `<!-- nwb-docs-test: run -->` and run
in `tests/nwb/test_documented_workflow.py`.

Check one document with `uv run pytest tests/docs/test_executable_docs.py -n 0 -k <name>`.
