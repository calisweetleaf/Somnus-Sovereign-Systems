# Web Research Subsystem - TODO

**Status:** PRODUCTION-READY

## File-Scoped Backlog

### research_vm_orchestrator.py
- [x] Finish orchestration logic for VM lifecycle
- [x] Add error handling for engine startup/shutdown
- [x] Implement timeout + watchdog for stuck jobs
- [x] Add hooks for session persistence + recovery
- [x] Security: enforce user/session isolation
- [x] Expose ResearchVMType, BrowserEngine, ResearchCapability to UI for selection.
- [x] Add ability to save/load custom VM presets from UI.
- [x] Connect research extension’s `/api/research/extracted_data` POST → orchestrator → DB.
- [x] Add UI buttons for "Extract Data" and "Analyze Credibility" that call Flask endpoints.
- [ ] Extend `ai_analysis.py` with:
    - [ ] Fact-check cross-referencing (pull multiple URLs, check contradictions).
    - [ ] Citation graph extraction (map relationships between sources).
- [ ] UI hooks for running these functions on highlighted content.
- [ ] Add migrations for scaling beyond SQLite (option: Postgres).
- [x] Expose "Recent Research Sessions" in UI using `/api/research/sessions`.
- [x] UI needs drag-drop zone → sends file to orchestrator via file watcher.
- [x] Add PDF summarization hook (link to embeddings + bias analysis automatically).
- [x] Display session memory log in UI (persisted context across sessions).
- [x] Add user-facing toggle: “Store this research session” vs “Ephemeral”.
- [x] Expose resource monitoring (CPU%, RAM%, Disk%) to UI dashboard.
- [x] Add "Suspend VM" and "Resume VM" buttons.
- [x] Add configurable VPN/proxy support to UI.
- [x] UI toggles for blocked domains/content filters.
- [x] Wrap Flask endpoints with authentication (user/session tokens).
- [x] Align API schema with Somnus Kernel global router.
- [x] Hook dropdowns (browser engine, capabilities) to orchestrator config.
- [x] Hook search bar → `/api/research/session` start.
- [x] Add tabbed views: Session history, Active Sources, Analysis Results.

### research_stream_manager.py
- [x] Implement real authentication (JWT/OAuth/local tokens)
- [x] Add authorization: enforce session ownership + roles
- [x] Integrate Redis pub/sub for scaling (multi-node)
- [x] Add encryption/compression layer for WebSocket payloads
- [x] Implement fine-grained event subscriptions (e.g. entity-level, contradiction-level)
- [x] Add retry/backoff strategies for broadcast failures
- [x] Harden error handling for Redis + network edge cases
- [x] Expose monitoring API for metrics (Prometheus endpoint)

### research_session.py
- [x] Upgrade contradiction detection beyond embeddings (symbolic + temporal reasoning)
- [x] Add scoring/triage system for ethical concerns
- [x] Enhance bias detection (statistical + language model classifiers)
- [x] Allow pluggable embedding models (larger + multimodal)
- [x] Implement entity prioritization (by credibility, uniqueness, coverage)
- [x] Persist synthesis results + contradictions into memory manager
- [x] Smarter pruning/archiving for large sessions
- [x] Add collaboration-aware updates (multiple users editing plan/entities)
- [x] Improve memory export (not just top 10 entities)

### report_exporter.py
- [x] **Artifact System Tightening**
  - Ensure `ResearchArtifactCreator` always routes through the **Artifact System** when available (currently has a `_create_standalone_artifact` fallback — confirm this isn’t needed in production).
  - Confirm artifact metadata (report_id, scores, contradictions) is standardized with the rest of the ecosystem.
- [x] **Artifact Linking**
  - Each transformation should not only create a new artifact, but also **link back** to the original (bi-directional reference in metadata).
- [x] **Security/Isolation**
  - Double-check artifacts created here respect the same sandbox/overlay security levels as the Artifact System (currently uses `SecurityLevel.SANDBOXED`, but needs review).
- [x] **Session → Artifact Traceability**
  - Verify `ExportResult.metadata['session_id']` ties into **Research VM Orchestrator** sessions.
  - Ensure reports created here can be indexed/reopened later through artifact lookups.
- [x] **Error Handling**
  - Right now errors in artifact creation/logging are written, but not escalated. Need retry/backoff for artifact writes.
- [x] **UI Integration**
  - Interactive HTML has a "Transform Report" panel — ensure this calls through **Artifact System APIs** rather than raw fetch.
- [x] **Model Loader Integration**
  - Ensure `ReportTransformationEngine` uses the **shared model loader** system, not its own loader, for consistent control across subsystems.
- [x] **Artifact Subprocess Calls**
  - Reports are written as raw HTML/MD/JSON files in `_create_standalone_artifact`. Replace this with Artifact System subprocess calls, so even fallback mode uses a controlled container overlay.

### deep_research_planner.py
- [x] Add **Artifact System integration**:
  - Export `ResearchPlan` and `CollaborationPlan` as artifacts (markdown/json).
  - Store in `/artifacts/research/` with session + plan ID.
  - Allow later editing via artifact UI → sync back into planner state.
- [x] Replace `data/research_vm` persistence with artifact-backed persistence (still mirror to VM for fallback).
- [x] Add artifact metadata:
  - Complexity, domains, deadlines.
  - Version history → track modifications in artifact.
- [x] Ensure multi-AI collab plans generate **sub-artifacts** (e.g. `agent_assignments.json`).

### research_engine.py
- [x] Confirm **artifact outputs**:
  - Current `ReportExporter` integration is correct (HTML/Markdown artifacts).
- [ ] Add support for **intermediate artifact checkpoints**:
    - [ ] Memory search results (`memory_summary.md`).
    - [ ] Contradiction maps (`contradictions.json`).
    - [ ] Partial syntheses (`draft_synthesis.md`).
- [x] Add **artifact references**

### Web Research ↔ Memory TODO
- [x] Patch research_session.py to store session metadata & queries into memory_core
- [x] Patch research_stream_manager.py to log streamed results into memory
- [x] Ensure deep_research_planner.py persists planning traces into memory
- [x] Wire research_intelligence.py insights + exports into memory + artifacts

### HTML TODO
- [ ] Implement **Research Mode dropdown** (Assisted / Autonomous / Collaborative)
- [ ] Implement **Research Depth dropdown** (Surface / Moderate / Deep / Expert)
- [ ] Implement **Source Types toggle buttons** (Academic, News, Technical, Gov)
- [ ] Max Sources **slider control** with dynamic labels
- [x] **Sources panel**: clickable cards (title, summary, source, credibility, views, recency)
- [ ] **Analysis panel**: metrics widgets (words analyzed, key concepts, sentiment pie, topic bars, timeline)
- [ ] **Knowledge Graph panel**: switchable views (network, timeline, hierarchy)
- [ ] **Research Progress tracker**: progress dots (query analysis, source discovery, etc.)
- [ ] **Quality Metrics panel**: credibility score, source count, coverage %, conflict count
- [ ] **Start / Pause / Stop buttons**
- [x] **Export options** (Report → PDF/Markdown, Website → static HTML, JSON → raw output)
- [ ] **Generate button** (kickoff deeper synthesis)
- [ ] Dynamic update of metrics as analysis runs
- [ ] Smooth transitions between panels (Sources → Analysis → Knowledge Graph)
- [x] Ensure export buttons map backend results to correct file formats
- [ ] Responsive design → mobile + desktop