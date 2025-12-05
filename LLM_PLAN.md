# LLM Integration Implementation Plan

**Goal:** Make `neurospatial` directly usable by LLM agents (including via MCP), with a small, robust tool surface, good schemas, and “skills”/playbooks that describe higher‑level workflows.

---

## Phase 0 – Scope & Priorities

- Clarify target environments:
  - Which agent runtimes first? (MCP, Claude Code, custom Python agent, etc.)
  - Primary data sources? (NWB only vs NWB + CSV/HDF5, etc.)
- Pick 1–2 “flagship workflows” as MVP:
  - Example: “Analyze an NWB session: build environment, compute occupancy + place fields, summarize results.”
  - Example: “Annotate a video, build environment + regions, then compute reward‑zone stats.”
- Decide layering:
  - In‑library agent façade (`neurospatial.agent_api`) vs separate `neurospatial-mcp-server` repo.
  - Recommended: implement both, with the MCP server as a thin wrapper over `neurospatial.agent_api`.

---

## Phase 1 – LLM/Tool Façade Inside neurospatial

Implement a minimal, JSON‑friendly API that wraps core functionality and hides Python objects behind IDs.

### P1.1 Module and Registry

- Add `src/neurospatial/agent_api/__init__.py` (or `src/neurospatial/agent_api.py`) with:
  - `ENV_REGISTRY: dict[str, Environment]`
  - `FIELD_REGISTRY: dict[str, np.ndarray]`
  - `_new_id(prefix: str) -> str` helper.
- Define registry lifecycle expectations in the module docstring:
  - Registries are in‑process only.
  - Callers are responsible for persistence via IO tools.

### P1.2 Core Agent Functions (MVP)

Provide JSON‑safe, high‑level agent functions that wrap existing APIs:

- **Session / environment creation**
  - `create_environment_from_samples_tool(positions, bin_size, units=None, frame=None) -> {"env_id": str, "summary": {...}}`
  - `create_environment_from_nwb_tool(path, bin_size, position_source=...) -> {"env_id": str, "summary": {...}}` (after NWB M1 is available).
  - `load_environment_tool(path) -> {"env_id": str, "summary": {...}}` (wraps `from_file` / later `Environment.from_nwb`).
- **Basic analysis**
  - `compute_occupancy_tool(env_id, times, positions, **options) -> {"field_id": str, "summary": {...}}`
  - `compute_place_field_tool(env_id, spike_times, times, positions, smoothing_method="diffusion_kde", **options) -> {"field_id": str, "summary": {...}}`
  - Inputs/outputs must be JSON‑safe: lists, dicts, floats, ints, bools.
- **Geometry / regions**
  - `list_regions_tool(env_id) -> {"regions": [..., {"name": str, "type": "point|polygon", ...}]}`
  - `add_region_tool(env_id, name, kind, geometry) -> {"ok": bool, "summary": {...}}`
- **Serialization**
  - `save_environment_tool(env_id, path) -> {"path": str}` (wraps `to_file` / later `.to_nwb`).
  - `save_field_tool(env_id, field_id, path) -> {"path": str}` (NumPy save or NWB writer later).

### P1.3 Introspective Summaries (“Describe” Helpers)

- Implement helpers used by the agent API:
  - `summarize_environment(env) -> dict`
    - Example keys: `n_bins`, `n_dims`, `units`, `extent`, `is_1d`, `n_regions`, etc.
  - `summarize_field(env, field) -> dict`
    - Example keys: `min`, `max`, `mean`, `peak_bin`, `peak_position`, and optional metrics like sparsity/info.
  - Optional: `summarize_field_text(env, field) -> str`
    - 1–3 sentence natural‑language description (for direct LLM use).
- Ensure each summary contains:
  - Machine‑friendly numeric fields.
  - A short textual description that an LLM can quote or expand.

### P1.4 Error Handling and Logging

- Standardize exceptions in agent API functions:
  - Validation errors → `ValueError` with precise messages.
  - Missing env/field IDs → `KeyError` (e.g., `f"env_id {env_id!r} not found"`).
  - Downstream errors (e.g., from NWB IO) are re‑raised or wrapped with clear context.
- Avoid printing in tools:
  - Use `logging.getLogger("neurospatial.tools")` for debug information.

### P1.5 Tests

- Add `tests/agent_api/test_agent_api_core.py`:
  - Happy path for each agent API function.
  - Error paths for invalid IDs or bad inputs.
  - Tests to check `summarize_environment` / `summarize_field` dict shape and key presence.

---

## Phase 2 – MCP Server / Agent Adapter

Expose the Phase 1 agent API through a standard agent protocol (e.g., MCP).

### P2.1 MCP Server Skeleton

- Create a separate project or subdirectory:
  - Example: `neurospatial-mcp-server/` or `servers/mcp/`.
- Basic structure:
  - `pyproject.toml`
  - `src/neurospatial_mcp_server/server.py` (or similar entrypoint).
- Add dependency on `neurospatial` and chosen MCP implementation.

### P2.2 Tool Definitions

- For each `neurospatial.agent_api.*` function:
  - Map to an MCP tool with JSON schema for inputs/outputs.
  - Keep schemas stable and minimal; reuse enums and small structs where possible.
- Name tools and write descriptions at the workflow level:
  - Examples: `create_environment_from_nwb`, `compute_place_field`, *not* low‑level `bin_at`.

### P2.3 Resources and Resource Templates

- Define resources for:
  - On‑disk session artifacts (e.g., NWB files by path or logical ID).
  - Saved environments/fields (e.g., `env://{env_id}` URIs).
- Define resource templates for:
  - “NWB session by ID.”
  - “Environment by analysis run ID.”
- Implement MCP handlers that:
  - Resolve resource URIs to files.
  - Call `neurospatial.agent_api.*` or IO functions to summarize them.

### P2.4 Minimal Integration Test

- Write a small test harness that:
  - Starts the MCP server.
  - Calls a minimal set of tools through an MCP client:
    - `create_environment_from_samples`
    - `compute_occupancy`
    - `compute_place_field`
  - Verifies responses and internal registry state.

---

## Phase 3 – Skills / Playbooks (Claude‑Style)

Provide higher‑level “skills” that orchestrate multiple tools and link to deeper context files.

### P3.1 Skills Directory and Format

- Add `skills/` (or `docs/skills/`) with one skill per file:
  - `skills/analyze_nwb_session.skill.md`
  - `skills/place_field_batch.skill.md`
- Use a simple format: front‑matter + markdown body.
  - Front‑matter (YAML/JSON):
    - `id`
    - `title`
    - `description`
    - `inputs`
    - `outputs`
    - `tools_used`
    - `entrypoint` (module + function)
    - `related_docs`
    - `example_invocations`
  - Markdown body:
    - Narrative description.
    - Edge cases and caveats.
    - Performance notes and recommended parameters.

### P3.2 Define 2–3 Flagship Skills

- **Skill: `analyze_nwb_session`**
  - Orchestrates:
    - `create_environment_from_nwb_tool`
    - `compute_occupancy_tool`
    - `compute_place_field_tool` (looping over units)
    - `summarize_environment` / `summarize_field` / `summarize_field_text`
  - Inputs:
    - `nwb_path`
    - `bin_size`
    - Optional filters, e.g., units or time ranges.
  - Outputs:
    - Summary text.
    - List of per‑cell stats.
    - Paths to saved artifacts.

- **Skill: `annotate_video_and_analyze`**
  - Uses:
    - `annotate_video`
    - `Environment.from_samples`
    - Tool‑layer functions for environment and region‑based stats.
  - Focus:
    - Boundary + region annotations.
    - Reward‑zone or region‑based summary statistics.

- **Skill: `compare_place_fields_between_sessions`**
  - Uses:
    - IO + alignment functions from the main API.
    - Tools to compute and describe differences between sessions/animals.
  - Focus:
    - Cross‑session / cross‑animal alignment.
    - Quantitative and textual comparison.

### P3.3 Link Skills to Code and Docs

- In each skill front‑matter:
  - `entrypoint: neurospatial.agent_api.analyze_nwb_session` (or equivalent).
  - `related_docs` pointing to:
    - User guides, e.g., `docs/user-guide/neuroscience-metrics.md`.
    - Examples, e.g., `docs/examples/11_place_field_analysis.ipynb`.
- This makes it easy for agents (or humans) to:
  - Discover the right workflows.
  - Pull additional context when needed.

---

## Phase 4 – LLM‑Oriented Documentation

Make it easy for any LLM to learn how to use `neurospatial` in tool/agent mode.

### P4.1 LLM Integration Doc

- Add `docs/llm_integration.md` with:
  - Overview of the `neurospatial.agent_api` module:
    - Registry model.
    - Design philosophy (JSON‑safe, ID‑based).
  - JSON examples for tool inputs/outputs.
  - Mapping between tools and MCP (or other agent frameworks).
  - Pointers to:
    - `skills/` directory.
    - Key user‑guide pages (metrics, animation, NWB, etc.).

### P4.2 Update CLAUDE.md

- Add a section: “Using neurospatial with tools / MCP”.
- Include:
  - Recommended import patterns:
    - `from neurospatial import ...` for direct code.
    - `from neurospatial.agent_api import ...` for agent workflows.
  - Example multi‑step workflow for Claude Code:
    - Open skill file.
    - Call tools.
    - Inspect results and summaries.

---

## Phase 5 – Iteration and Expansion

After an MVP vertical slice is working, expand coverage and depth.

### P5.1 Broaden Tool Surface

- Add tools for additional analysis modules as they mature:
  - Head direction analysis.
  - Event‑based analyses (laps, region crossings, reward events).
  - Alignment/transforms (cross‑session or cross‑animal alignment).
- Apply the same patterns:
  - ID‑based registries.
  - JSON‑safe IO.
  - Numeric + textual summaries.

### P5.2 Performance and Robustness

- Stress‑test tools on large sessions:
  - Avoid huge JSON payloads (prefer IDs + persisted artifacts).
  - Add options for downsampling or summarizing large arrays.
- Harden error messages and logging for agent debuggability.

### P5.3 Feedback Loop

- Use real LLM agents (Claude Code, OpenAI, MCP clients) against the server:
  - Collect failure modes:
    - Ambiguous errors.
    - Confusing parameter names.
    - Missing skills or summaries.
  - Adjust:
    - Tool schemas.
    - Summaries and descriptions.
    - Skill docs and examples.
