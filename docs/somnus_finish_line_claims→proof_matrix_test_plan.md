# Somnus Sovereign Systems — Finish-Line Triage (Claims → Proof Matrix)

**Context**: Based on Claude’s two analysis dumps (local connectors + project-RAG), mapped to Somnus’ stated architecture: persistent VMs per agent; Artifact subsystem via disposable Docker overlays; unified memory/cache; offline-first; ≤6 GB RAM target.

---

## 1) Architecture Snapshot (derived from the dumps + prior Somnus specs)
- **VM Topology**
  - Chat VM (coordination), Projects VM (repos + dev flows), Research VM (autonomous browser research), optional Dev VM variants.
  - **Hot-swap** model: only one foreground VM at a time; memory/cache continuity across swaps.
- **Memory & Cache**
  - Semantic memory store (encrypted), write-behind + crash-safe checkpoints.
  - System cache for model/session artifacts; namespaced + LRU.
- **Artifact System**
  - Ephemeral Docker containers for compute-heavy or code-exec tasks; VMs stay clean.
- **Multi-Agent Core**
  - Agent Collaboration Core; Communication Protocol with message IDs, statuses, checksums.
- **Session-as-Repo**
  - Chat turns emit versioned code blocks; promotion path to full Git repo; integrated repo analysis + semantic search.
- **Security**
  - Offline-first (SaaS kill switch), container escape detection, purple-team hooks, explicit user egress toggles.

---

## 2) Claims → Proof Matrix (what must be true + how to prove fast)

| # | Claude Claim | What Must Be True (Design Invariants) | 15–30 min Proof Procedure | Pass Criteria |
|---|---|---|---|---|
| 1 | **VM hot-swapping with zero state loss** | (a) Handover ledger records last committed agent state; (b) Memory-core flush + WAL; (c) Foreground VM teardown is transactional; (d) Rehydrate on next VM boot. | Run chat → switch to Projects mid-convo → back to Chat; kill Somnus between swaps once. Inspect WAL replay log. | Conversation continuity intact; no duplicate/ghost messages; WAL shows idempotent reapply; <2s rehydrate target (model reload excluded). |
| 2 | **Main VM shuts down, Projects VM launches** | Foreground VM owns no Artifact processes; Artifact subprocesses always in containers with external supervisor; teardown hooks drain IPC. | Start long-running container task; swap VMs mid-run. Verify container continues and logs stream to new VM UI. | No orphaned processes; task completes; UI relinks without manual intervention. |
| 3 | **Memory+cache preserve model “state”** | You aren’t persisting KV-cache; you persist semantic convo DAG + session vars. Rehydration replays context minimally (summaries), not raw tokens. | Measure tokens used on rehydrate vs. previous turn length; check summary compaction. | Rehydrate token cost bounded (e.g., <1k tokens per return). |
| 4 | **AI Personal Dev Environment learns & improves** | Learning ledger keyed by project type; metrics (setup_time, failures, tool_install_delta); policy applies diffs on next boot. | Create two toy projects (web + research), measure second-boot faster; log shows applied policy. | ≥20% setup-time reduction on second run (toy scale). |
| 5 | **Session-as-Repo: code blocks = commits** | Uniform capture schema: (msg_id, file_path, op, diff, rationale); atomic promotion to Git repo. | Generate 3 iterative code blocks; inspect commit metadata; run `git log --stat`. | Commits map 1:1 to messages; diffs reproducible; messages link back to chat IDs. |
| 6 | **Integrated editor + PTY collaboration** | Editor is CRDT-backed (e.g., Yjs); PTY execution runs only inside Artifact container with explicit mounts. | Two clients live-edit a file; run `python -V` in container PTY; confirm no host writes from PTY. | Edit convergence with no conflicts; PTY restricted to container FS; audit log records command. |
| 7 | **Deep repo analysis + semantic search** | Indexer builds embedding store + symbol table; search routes through memory-core with dedupe & TTL. | Clone a medium repo; run 5 queries; measure p95 latency. | p95 ≤ 300 ms on local index; results cite file+line with stable IDs. |
| 8 | **Offline-first, no silent network** | Global egress gate; plugin/Artifact layers inherit; UI banner shows state; attempts are logged and blocked. | Attempt `pip install requests` inside a blocked profile; attempt `fetch` in plugin. | Both blocked with explicit event; UI banner reflects OFFLINE. |
| 9 | **Crash integrity** | WAL + periodic snapshots; idempotent reducers. | SIGKILL somnus mid-write; restart; examine reconciliation. | No lost or duplicated ops; checksum matches. |
|10 | **Deterministic plugin lifecycle** | Install→enable→hot-reload→disable→remove leaves no residue; file diff clean. | Cycle a sample plugin twice; diff config/state dirs. | No orphan files/ports; state directories empty when removed. |

---

## 3) Hot-Swap Handover Blueprint (minimal, testable)
- **Handover Ledger**: `{vm_from, vm_to, session_id, seq_no, ts, state_hash, wal_pointer}`
- **Teardown Steps**: freeze input queue → flush memory-core → sync cache → write ledger → signal supervisor to spawn target VM → confirm ready → release.
- **Rehydrate Steps**: read ledger → validate `state_hash` → replay WAL entries since pointer → compute compaction summary → prime short context into model.
- **Failure Modes**: ledger write fails → keep VM; spawn aborted. Rehydrate validation fails → fallback to last snapshot; flag UI.

---

## 4) Session-as-Repo Pipeline (spec → test)
1) **Capture**: Every code fence annotated: ```lang path=... op=[create|edit|delete] rationale=...```.
2) **Diff**: Compute unified diff vs. working tree; attach `msg_id` and `author=agent_name`.
3) **Commit**: `git commit -m "[Somnus#<msg_id>] <summary>" --signoff`.
4) **Promote**: Create repo if needed; import session history; persist cross-ref index (chat ↔ commits).
5) **Collaborate**: CRDT editor writes into working tree; commits continue to reference chat IDs.
6) **Artifact Runs**: Always execute inside ephemeral container; attach build/run hashes to commit metadata.

**Proof Hooks**: `somnus logs --filter=session-commit`, `somnus demo promote`, `somnus demo link-check`.

---

## 5) Demo Choreography (offline; 8–12 minutes)
- Start in **Chat VM**. Prompt: “Build a tiny Flask API that returns /health=ok.”
- AI emits code blocks (app.py, requirements.txt). Somnus shows commit 1/2.
- Click **Promote to Project** → hot-swap to **Projects VM**. Editor opens with CRDT; concurrent edit tweaks route.
- Run **Artifact**: docker `python:3.11-alpine` to start the API; PTY shows logs.
- Swap to **Research VM** to auto-generate README and an architectural diagram (local-only). Swap back to Projects; continuity preserved.
- Kill Somnus process; relaunch; open project; verify state continuity & commit links.

---

## 6) Exit Criteria (ship/no-ship)
- Local-only parity with ≤6 GB RAM for marquee flows.
- Handover ledger + WAL validated under forced crash.
- Session↔Git cross-ref index accurate for ≥50 commits.
- No silent egress; all network attempts require explicit toggle.
- Plugin lifecycle leaves zero residue.

---

## 7) Risk Register & Mitigations
- **Model reload latency on swap**: consider a host **Model Host** microservice (still local) to keep weights mmapped and contexts multiplexed; or accept reload with visible spinner + progress.
- **Windows file locks**: ensure editor + indexer use shared-read; container bind mounts with `:ro` where possible.
- **Memory pressure (<6 GB)**: enforce small quantized GGUF; tighten cache TTL; avoid concurrent foreground VMs.
- **PTY scope creep**: never attach host FS; PTY restricted to container with explicit mounts.
- **Index drift**: write-after-index updates trigger incremental reindex; debounce filesystem events.

---

## 8) Docs Skeleton (ship with release)
- **README (User)**: install, first-run wizard, offline mode, model selection.
- **ARCHITECTURE**: VMs, Artifacts, Memory/Cache, Session-as-Repo, Security model.
- **OPERATIONS**: Backup/restore, logs, crash recovery, performance tuning.
- **DEVELOPER**: Plugin SDK, message protocol, demo harness.
- **SECURITY**: Threat model, sandboxing, purple-team hooks, no-network guarantees.

---

## 9) 60-Minute Implementation Checklist (today)
- [ ] Implement/write Handover Ledger + WAL pointer.
- [ ] Add `--offline` gate enforcement across plugins/Artifacts.
- [ ] Annotate code fences with `path/op/rationale`; wire commit cross-ref.
- [ ] Add PTY scope check to refuse host paths.
- [ ] Add `somnus demo` commands for the choreography.

---

**Deliverable state**: This doc is your finish-line playbook. Run the proof procedures; if any claim fails, the corresponding invariant tells you exactly what to harden next.

