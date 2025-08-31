# Web Research Subsystem - Breakdown
/web_research
┣ ai_browser_research_agent.py []
┣ deep_research_planner.py     []
┣ report_exporter.py           []
┣ research_cache_engine.py     []
┣ research_engine.py           []
┣ research_intelligence.py     []
┣ research_session.py          []
┣ research_stream_manager.py   []
┣ research_vm_orchestrator.py  []
┗ TODO.md
## Core Functions
1. **Query Handling**
   - Parse user query
   - Apply filters: depth, mode, source types
   - Build structured queries (with QDF, keywords, intent)

2. **Source Discovery**
   - Collect sources from defined pools (news, academic, technical, gov)
   - Apply max source limits (slider)
   - Handle trust/credibility weighting

3. **Analysis Engine**
   - Text summarization
   - Concept/keyphrase extraction
   - Sentiment/topic distribution
   - Contradiction/conflict detection
   - Knowledge graph construction (network/timeline/hierarchy)

4. **Research Progress Pipeline**
   - Query Analysis
   - Source Discovery
   - Content Analysis
   - Report Generation

5. **Quality Metrics**
   - Credibility Score (0-10)
   - Source Count
   - Coverage % (how broad the source net is)
   - Conflict Count (detected contradictions)

6. **Modes of Operation**
   - AI Assisted → approve/disapprove system actions
   - Autonomous → full agent mode
   - Collaborative → multi-agent / multi-user co-op

7. **Depth Settings**
   - Surface (Fast)
   - Moderate (Balanced)
   - Deep (Comprehensive)
   - Expert (Maximum)

8. **Output**
   - Sources panel (individual items w/ metadata)
   - Analysis panel (metrics, graphs, distributions)
   - Knowledge Graph (network, timeline, hierarchy views)
   - Export actions (Report, Website, Other)
   
   
# **HTML TODO**
# Web Research Subsystem - HTML/UI To-Do

## Controls
- [ ] Implement **Research Mode dropdown** (Assisted / Autonomous / Collaborative)
- [ ] Implement **Research Depth dropdown** (Surface / Moderate / Deep / Expert)
- [ ] Implement **Source Types toggle buttons** (Academic, News, Technical, Gov)
- [ ] Max Sources **slider control** with dynamic labels

## Panels
- [x] **Sources panel**: clickable cards (title, summary, source, credibility, views, recency)
- [ ] **Analysis panel**: metrics widgets (words analyzed, key concepts, sentiment pie, topic bars, timeline)
- [ ] **Knowledge Graph panel**: switchable views (network, timeline, hierarchy)
- [ ] **Research Progress tracker**: progress dots (query analysis, source discovery, etc.)
- [ ] **Quality Metrics panel**: credibility score, source count, coverage %, conflict count

## Actions
- [ ] **Start / Pause / Stop buttons**
- [x] **Export options** (Report → PDF/Markdown, Website → static HTML, JSON → raw output)
- [ ] **Generate button** (kickoff deeper synthesis)

## UX/Logic
- [ ] Dynamic update of metrics as analysis runs
- [ ] Smooth transitions between panels (Sources → Analysis → Knowledge Graph)
- [ ] Ensure export buttons map backend results to correct file formats
- [ ] Responsive design → mobile + desktop

*Note: The redundant `<div class="sidebar-sections">` in `research_v3.html` was also removed for structural cleanup.*

# research_vm_orchestrator.py TODOs

- [ ] **VM Configurations**
  - Expose `ResearchVMType`, `BrowserEngine`, `ResearchCapability` to UI for selection.
  - Add ability to save/load custom VM presets from UI.

- [ ] **Browser Integration**
  - Connect research extension’s `/api/research/extracted_data` POST → orchestrator → DB.
  - Add UI buttons for "Extract Data" and "Analyze Credibility" that call Flask endpoints.

- [ ] **Model/Analysis**
  - Extend `ai_analysis.py` with:
    - Fact-check cross-referencing (pull multiple URLs, check contradictions).
    - Citation graph extraction (map relationships between sources).
  - UI hooks for running these functions on highlighted content.

- [ ] **Database Layer**
  - Add migrations for scaling beyond SQLite (option: Postgres).
  - Expose "Recent Research Sessions" in UI using `/api/research/sessions`.

- [ ] **File Handling**
  - UI needs drag-drop zone → sends file to orchestrator via file watcher.
  - Add PDF summarization hook (link to embeddings + bias analysis automatically).

- [ ] **Memory Core**
  - Display session memory log in UI (persisted context across sessions).
  - Add user-facing toggle: “Store this research session” vs “Ephemeral”.

- [ ] **Monitoring**
  - Expose resource monitoring (CPU%, RAM%, Disk%) to UI dashboard.
  - Add "Suspend VM" and "Resume VM" buttons.

- [ ] **Security**
  - Add configurable VPN/proxy support to UI.
  - UI toggles for blocked domains/content filters.

- [ ] **API Integration**
  - Wrap Flask endpoints with authentication (user/session tokens).
  - Align API schema with Somnus Kernel global router.

- [ ] **HTML/UI Alignment**
  - Hook dropdowns (browser engine, capabilities) to orchestrator config.
  - Hook search bar → `/api/research/session` start.
  - Add tabbed views: Session history, Active Sources, Analysis Results.
  
  
# Web Research Subsystem TODO

## research_vm_orchestrator.py
- [ ] Finish orchestration logic for VM lifecycle
- [ ] Add error handling for engine startup/shutdown
- [ ] Implement timeout + watchdog for stuck jobs
- [ ] Add hooks for session persistence + recovery
- [ ] Security: enforce user/session isolation

## research_stream_manager.py
- [ ] Implement real authentication (JWT/OAuth/local tokens)
- [ ] Add authorization: enforce session ownership + roles
- [ ] Integrate Redis pub/sub for scaling (multi-node)
- [ ] Add encryption/compression layer for WebSocket payloads
- [ ] Implement fine-grained event subscriptions (e.g. entity-level, contradiction-level)
- [ ] Add retry/backoff strategies for broadcast failures
- [ ] Harden error handling for Redis + network edge cases
- [ ] Expose monitoring API for metrics (Prometheus endpoint)

## research_session.py
- [ ] Upgrade contradiction detection beyond embeddings (symbolic + temporal reasoning)
- [ ] Add scoring/triage system for ethical concerns
- [ ] Enhance bias detection (statistical + language model classifiers)
- [ ] Allow pluggable embedding models (larger + multimodal)
- [ ] Implement entity prioritization (by credibility, uniqueness, coverage)
- [ ] Persist synthesis results + contradictions into memory manager
- [ ] Smarter pruning/archiving for large sessions
- [ ] Add collaboration-aware updates (multiple users editing plan/entities)
- [ ] Improve memory export (not just top 10 entities)


# projection_registry.py
TODO:
- [ ] Add projection rules for research subsystem capabilities:
    - prefs.research.search_engine -> capabilities.flags.search_engine
    - prefs.research.depth (shallow/deep/exhaustive) -> prefs.research_mode
    - prefs.research.sources (news, academic, forums) -> context.research_sources
- [ ] Ensure new fields are redacted properly (avoid leaking raw queries).
- [ ] Add unit tests for new DSL rules.

# events.py
TODO:
- [ ] Expand ResearchEvent payload fields:
    - queries: List[str]
    - sources: List[str] (news, forums, drive, notion, etc.)
    - status: str ("started", "in_progress", "completed", "failed")
    - results_ref: Optional[str] (pointer to artifact/memory entry)
- [ ] Create convenience publishers:
    - publish_research_started(query, sources, **kwargs)
    - publish_research_completed(research_id, results_ref, **kwargs)
- [ ] Subscribe Research subsystem managers to these events.
- [ ] Wire research subsystem events into memory/artifact subsystems for persistence.

# integration
TODO:
- [ ] In research_vm_orchestrator, hook into projection_registry for user research prefs.
- [ ] In research_session, listen for ResearchEvents and store session metadata.
- [ ] In research_stream_manager, emit ResearchEvents when new queries are launched or completed.


# ===========================
# TODOs for report_exporter.py
# ===========================

- [ ] **Artifact System Tightening**
  - Ensure `ResearchArtifactCreator` always routes through the **Artifact System** when available (currently has a `_create_standalone_artifact` fallback — confirm this isn’t needed in production).
  - Confirm artifact metadata (report_id, scores, contradictions) is standardized with the rest of the ecosystem.

- [ ] **Artifact Linking**
  - Each transformation should not only create a new artifact, but also **link back** to the original (bi-directional reference in metadata).

- [ ] **Security/Isolation**
  - Double-check artifacts created here respect the same sandbox/overlay security levels as the Artifact System (currently uses `SecurityLevel.SANDBOXED`, but needs review).

- [ ] **Session → Artifact Traceability**
  - Verify `ExportResult.metadata['session_id']` ties into **Research VM Orchestrator** sessions.
  - Ensure reports created here can be indexed/reopened later through artifact lookups.

- [ ] **Error Handling**
  - Right now errors in artifact creation/logging are written, but not escalated. Need retry/backoff for artifact writes.

- [ ] **UI Integration**
  - Interactive HTML has a "Transform Report" panel — ensure this calls through **Artifact System APIs** rather than raw fetch.

- [ ] **Model Loader Integration**
  - Ensure `ReportTransformationEngine` uses the **shared model loader** system, not its own loader, for consistent control across subsystems.

- [ ] **Artifact Subprocess Calls**
  - Reports are written as raw HTML/MD/JSON files in `_create_standalone_artifact`. Replace this with Artifact System subprocess calls, so even fallback mode uses a controlled container overlay.
  
# Web Research Subsystem – Rolling TODO

## Deep Research Planner (research_planner.py)
- [ ] Add **Artifact System integration**:
  - Export `ResearchPlan` and `CollaborationPlan` as artifacts (markdown/json).
  - Store in `/artifacts/research/` with session + plan ID.
  - Allow later editing via artifact UI → sync back into planner state.
- [ ] Replace `data/research_vm` persistence with artifact-backed persistence (still mirror to VM for fallback).
- [ ] Add artifact metadata:
  - Complexity, domains, deadlines.
  - Version history → track modifications in artifact.
- [ ] Ensure multi-AI collab plans generate **sub-artifacts** (e.g. `agent_assignments.json`).

## Research Execution Engine (research_engine.py)
- [ ] Confirm **artifact outputs**:
  - Current `ReportExporter` integration is correct (HTML/Markdown artifacts).
  - Add support for **intermediate artifact checkpoints**:
    - Memory search results (`memory_summary.md`).
    - Contradiction maps (`contradictions.json`).
    - Partial syntheses (`draft_synthesis.md`).
- [ ] Add **artifact references


# Web Research ↔ Memory TODO
- [ ] Patch research_session.py to store session metadata & queries into memory_core
- [ ] Patch research_stream_manager.py to log streamed results into memory
- [ ] Ensure deep_research_planner.py persists planning traces into memory
- [ ] Wire research_intelligence.py insights + exports into memory + artifacts


# TODO: Web Research Prompt Integration

# 1. Expand Prompt Templates
# - Finalize `research_mode` template to include artifact export instructions.
# - Add structured blocks like "Output to Artifact System" and "Cache research session context".

# 2. Template Variable Injection
# - Ensure `research_query`, `research_depth`, and `source_types` are always injected by
#   research_session / stream_manager before calling prompt_manager.generate_system_prompt.
# - Add "artifact_path" or "artifact_channel" variable to templates.

# 3. Memory-Aware Prompts
# - Extend _gather_context_data() to explicitly pull prior research session context
#   (via research_cache_engine) in addition to general memory_manager.
# - Pass these snippets into `memory_context` for continuity.

# 4. Research-Specific Analytics
# - Track research prompt usage separately (success rate, session duration, export count).
# - Add counters into prompt_analytics keyed to "research_mode".

# 5. Artifact System Hooks
# - Ensure generated prompts reference artifact export system (report_exporter.py).
# - Add default "Research Artifact" tag to research outputs so they can be indexed and retrieved.

# 6. Multi-Agent Prompt Cohesion
# - Sync with multi-agent collaboration: research_mode templates should accept injected
#   agent roles ("synthesizer", "scout", "critic") when research requires agent division.

# 7. HTML/UI Bindings
# - Update research_v3.html dropdowns so selecting "Research Mode" sends correct prompt_type.
# - Ensure UI passes research_query → prompt_manager as variable payload.

# 8. Safety Layer
# - Add research-specific safety filters (source reliability, disinfo handling) into
#   _apply_safety_filters when prompt_type == RESEARCH_MODE.
# Task: Implement & Integrate Web Research Subsystem (Somnus Sovereign Kernel)

## Objective
Deliver a **refactored, production-ready** `/web_research` subsystem that **keeps the existing scope and behavior** while tightening interfaces, reliability, and security. Do not add features; complete and integrate what already exists.

---

## Inputs
- `/web_research/` (9 files):  
  `ai_browser_research_agent.py`, `deep_research_planner.py`, `report_exporter.py`,  
  `research_cache_engine.py`, `research_engine.py`, `research_intelligence.py`,  
  `research_session.py`, `research_stream_manager.py`, `research_vm_orchestrator.py`
- Shared: `prompt_manager.py`, `research_v3.html`
- Backlog: `TODO.md` (authoritative)

---

## Context & Responsibilities (unchanged)
1. Parse user queries → **structured search requests** (QDF, keywords, intent).
2. **Discover + weight sources** (news, academic, gov, technical) with credibility and max-source limits.
3. Run **analysis** (summaries, key concepts, sentiment/topic, contradictions, knowledge graph).
4. **Export** results as artifacts (reports, graphs, JSON) via Artifact System.
5. Persist **sessions** (queries, contradictions, metrics) to the Memory Core.
6. Operate in modes: **Assisted**, **Autonomous**, **Collaborative**.
7. Drive **UI controls** (depth slider, source toggles, metrics panels).

**Integrations**
- Artifact System (all outputs flow here; no standalone writes in prod).
- Memory Core (sessions, caches, contradictions).
- Prompt Manager (`PromptType.RESEARCH_MODE`).
- Global Router (prefs, events, user registry).

---

## Operating Modes (semantics)
- **Assisted**: confirm plan/fetch/analyze/export steps with user.
- **Autonomous**: run plan → fetch → analyze → export using prefs/depth; stream status.
- **Collaborative**: multi-user/agent contributions; merge + dedupe with session locks.

## Depth Settings (map to engine limits)
- **Surface** (fast), **Moderate** (balanced), **Deep** (comprehensive), **Expert** (maximum).

## Research Pipeline
1. Query Analysis → 2. Source Discovery → 3. Content Analysis → 4. Report Generation

## Quality Metrics
- **Credibility (0–10)**, **Source Count**, **Coverage %**, **Conflict Count**.

---

## Orchestration Flow (contracts)
1. `research_vm_orchestrator.start(job_spec)`  
   Emits `research.job.started(session_id, user_id, mode, depth, sources)`
2. `deep_research_planner.build_plan(query, prefs)`  
   Emits `research.plan.ready(query_plan, qdf, limits)`
3. `ai_browser_research_agent.run(plan)`  
   Streams `research.fetch.progress(source_id, pct, eta)`
4. `research_engine.process(doc_batch)`  
   Emits `research.analysis.update(summaries, contradictions, key_concepts, metrics)`
5. Persist via `research_session.save(...)` and `research_cache_engine.put(...)`
6. Export via `report_exporter.emit(artifact_spec)` → Artifact System
7. `research.job.finished(artifact_ids, metrics)`

---

## UI / HTML Bindings (`research_v3.html`)
**Controls**
- Research Mode dropdown (Assisted / Autonomous / Collaborative)
- Research Depth dropdown (Surface / Moderate / Deep / Expert)
- Source Type toggles (Academic, News, Technical, Gov)
- Max Sources slider (with dynamic label)

**Panels**
- Sources (cards: title, summary, source, credibility, views, recency) ✔
- Analysis (metrics: words analyzed, key concepts, sentiment pie, topic bars, timeline)
- Knowledge Graph (views: network / timeline / hierarchy)
- Progress tracker (dots: query analysis → source discovery → content analysis → report)
- Quality Metrics (credibility score, source count, coverage %, conflict count)

**Actions**
- Start / Pause / Stop
- Export options (Report → PDF/Markdown, Website → static HTML, JSON → raw)
- Generate (kickoff deeper synthesis)

**UX/Logic**
- Live metric updates as analysis runs
- Smooth panel transitions (Sources → Analysis → Graph)
- Export buttons map to Artifact System formats
- Responsive design (mobile + desktop)

**Binding Table**
| UI Control | Endpoint | Payload (min fields) | Event Stream |
|---|---|---|---|
| Start | `POST /api/research/run` | `{query, mode, depth, sources[], prefs{}, max_sources}` | `fetch.progress`, `analysis.update`, `job.finished` |
| Pause/Stop | `POST /api/research/session/{id}/{pause|stop}` | `{}` | `job.updated` |
| Mode/Depth | `PATCH /api/research/session/{id}/settings` | `{mode, depth}` | `plan.ready` |
| Source toggles | `PATCH /api/research/session/{id}/sources` | `{sources[]}` | `plan.ready` |
| Max Sources slider | `PATCH /api/research/session/{id}/limits` | `{max_sources}` | `plan.ready` |
| Metrics panel | `GET /api/research/session/{id}/metrics` | `—` | `analysis.update` |
| Export | `POST /api/research/session/{id}/export` | `{formats[], template?, scope}` | `export.done` |

---

## Stable Schemas
**Structured Query**
```json
{
  "query": "string",
  "mode": "assisted|autonomous|collaborative",
  "depth": "surface|moderate|deep|expert",
  "sources": ["news","academic","gov","technical"],
  "prefs": { "qdf": 3, "lang": "en", "max_sources": 40 }
}

Analysis Update
{
  "session_id": "uuid",
  "stage": "analysis",
  "summaries": [{ "source_id": "string", "summary": "…" }],
  "contradictions": [{ "a":"source_id", "b":"source_id", "note":"…" }],
  "key_concepts": ["…"],
  "metrics": { "coverage": 0.0, "novelty": 0.0, "confidence": 0.0 }
}

Artifact Export Spec
{
  "session_id": "uuid",
  "formats": ["html","pdf","md","jsonl"],
  "include": { "summaries": true, "graphs": true, "raw": false },
  "metadata": { "project": "Somnus", "module": "web_research" }
}

Memory Session Record
{
  "session_id": "uuid",
  "user_id": "uuid",
  "queries": ["…"],
  "mode": "…",
  "timeline": [{ "t": "iso8601", "event": "research.plan.ready" }],
  "contradictions": ["…"],
  "cache_keys": ["…"],
  "artifacts": ["artifact_id…"]
}


Security, Fault-Tolerance, Observability

Authentication/Authorization: JWT/OAuth/local tokens; enforce session ownership and roles.
Isolation: namespace all IO by session_id + user_id; deny cross-tenant reads.
Timeouts: per-job watchdog; cancel stuck tasks.
Retries: exponential backoff + jitter for artifact/memory/stream writes.
Validation: sanitize URLs/MIME; strip scripts; reject unsupported content.
Metrics: Prometheus counters/timers; structured logs (ts, level, session_id, user_id, component, event, ms, size).
Health: /api/research/healthz reports adapter readiness + queue depth.

Error Codes

WR-INPUT-400 invalid payload
WR-TIMEOUT-408 job timeout
WR-NET-502 upstream fetch error
WR-ART-503 artifact service unavailable
WR-MEM-503 memory/caching unavailable
WR-STATE-409 invalid session state/lock


Implementation Rules

Backlog is authoritative: implement every TODO.md item. If deferring, mark DEFERRED: with rationale and impact.
Wire the flow: orchestrator → stream_manager → engine.

All exports through report_exporter.py → Artifact System.
research_session + research_cache_engine persist to Memory Core.
Inject prompt_manager.py templates when RESEARCH_MODE is active.
Align with projection_registry.py + events.py.


HTML/UI alignment: ensure all controls bind to endpoints above and reflect state via streams.


File-Scoped Backlog (from TODO.md, unchanged in scope)
research_vm_orchestrator.py

VM lifecycle; engine start/stop errors; timeout + watchdog; session persistence/recovery; user/session isolation.
Expose ResearchVMType, BrowserEngine, ResearchCapability to UI; save/load VM presets.
Browser integration: accept /api/research/extracted_data POST; UI “Extract Data” & “Analyze Credibility” buttons → Flask endpoints.
Model/analysis: extend ai_analysis.py with fact-check cross-refs + citation graph extraction; UI hooks for highlighted content.
DB layer: migrations for Postgres; /api/research/sessions lists recent sessions.
File handling: drag-drop → file watcher; PDF summarization hook (embeddings + bias analysis).
Memory Core: session memory log in UI; “Store this session” vs “Ephemeral”.
Monitoring: CPU/RAM/Disk; “Suspend VM” / “Resume VM”.
Security: VPN/proxy config; blocked domains/content filters.
API: auth wrapping; align schema with Global Router.
HTML hooks: dropdowns → orchestrator config; search bar → /api/research/session start; tabs (history/sources/results).

research_stream_manager.py

Auth (JWT/OAuth/local); authorization (ownership/roles).
Redis pub/sub for scale; encryption/compression for WS payloads.
Fine-grained subscriptions (entity-level, contradiction-level).
Retry/backoff for broadcasts; robust Redis/network error handling.
Monitoring API (Prometheus).

research_session.py

Contradiction detection beyond embeddings (symbolic + temporal); ethical triage scoring.
Enhanced bias detection (statistical + classifier); pluggable embedding models (incl. multimodal).
Entity prioritization (credibility, uniqueness, coverage).
Persist syntheses + contradictions to Memory; pruning/archiving for large sessions.
Collaboration-aware updates; richer memory export.

projection_registry.py

Projection rules:
prefs.research.search_engine → capabilities.flags.search_engine
prefs.research.depth → prefs.research_mode
prefs.research.sources → context.research_sources
Redaction rules; unit tests for DSL rules.

events.py

Expand payloads: queries: List[str], sources: List[str], status, results_ref.
Convenience publishers: publish_research_started(...), publish_research_completed(...).
Subscribe managers; wire events into Memory + Artifacts.

Integration

Orchestrator reads projection_registry for user research prefs.
Session listens for ResearchEvents; Stream Manager emits them on start/complete.

report_exporter.py

Force Artifact System path (fallback only under controlled overlay).
Standardize metadata (report_id, scores, contradictions); bi-directional artifact linking.
Security parity with Artifact System sandbox levels.
Session ↔ artifact traceability (metadata.session_id); resolvable/indexable.
Retry/backoff on artifact writes; UI “Transform Report” calls Artifact APIs.
Use shared model loader; replace raw file writes in fallback with controlled subprocess.

deep_research_planner.py

Export ResearchPlan/CollaborationPlan as artifacts (md/json) under /artifacts/research/; editable via artifact UI.
Replace data/research_vm persistence with artifact-backed storage (mirror to VM for fallback).
Artifact metadata (complexity, domains, deadlines, version history); sub-artifacts for multi-AI plans.

research_engine.py

Confirm ReportExporter integration (HTML/MD).
Intermediate artifact checkpoints: memory_summary.md, contradictions.json, draft_synthesis.md.
(Keep scope to current analysis features.)

Web Research ↔ Memory

Persist session metadata & queries; stream logs to memory; planner traces into memory; intelligence insights + exports into memory + artifacts.

Prompt Integration (prompt_manager.py)

Finalize research_mode templates (include artifact export & cache instructions).
Inject research_query, research_depth, source_types, artifact_channel.
_gather_context_data() pulls prior research context via research_cache_engine.
Track prompt analytics for research_mode; add default “Research Artifact” tag.
Multi-agent roles supported when collaboration is active.
UI passes prompt_type=RESEARCH_MODE with query vars.


Definition of Done

UI can run any mode/depth and receive live progress/analysis.
Sessions + caches persist to Memory Core; contradictions and key concepts appear in metrics and memory.
All exports route via Artifact System (fallback controlled overlay only).
All TODO.md items implemented or marked DEFERRED: with reason + impact.
No cross-session leakage; negative tests confirm isolation.
Graceful failures with stable error codes; retries visible in logs.
research_v3.html controls map 1:1 to backend; export buttons produce correct files.


Deliverable (unchanged)
A refactored, production-ready /web_research subsystem aligned with TODO.md:

Every TODO resolved or consciously DEFERRED (with rationale)
All exports via Artifact System
All sessions logged to Memory Core
Prompts routed via prompt_manager.py
UI bindings validated against research_v3.html


