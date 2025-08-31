# DevSessionManager To-Do
# Somnus Chat App – Core Subsystems (running summary)

## Virtual Machines
- Every subsystem (chat, research, projects, artifacts) = its own VM.
- Only one active VM at a time (6GB RAM budget).
- VM hot-swap handled via memory system + system cache.

## Dev Sessions
- Chat → repo transformation.
- Tracks prompts, AI responses, code blocks as events.
- Supports diffs, versioning, rollback.
- Promotion path to Project VM.
- ✅ Manager + schemas working, needs persistence & bridge to VM.

## Artifacts
- Containerized execution via Docker.
- Local API overlay – doesn’t shut down VM, just overlays front-end.
- Supports multi-language, multi-format, export pipelines.
- Proven working (e.g. Hello World + dashboard demo).

## Research
- Independent VM specialized for web/deep research.
- Configurable depth + AI research modes.
- Dropdown menus for fine-tuning scope & intensity.

## Projects
- One project = one VM (not folders).
- Grows over time, carries memory, cache, and tools.
- Projects menu is gateway into persistent, “real” development VMs.


- [ ] Add/verify `DevSessionStatus` enum in `schemas.dev_session_schemas`.
- [ ] Confirm `MemoryManager` supports `memory_type="dev_session"` and `get_memory_by_id()`.
- [ ] Ensure `DevSession` model implements:
  - `.add_event()`
  - `.update_code_block()` (returns diff string)
  - `.get_code_block()` (returns code block w/ version)
- [ ] Add unit tests:
  - Session creation and persistence
  - Event logging + retrieval
  - Code block update diff correctness
  - Session status transitions (active → paused → completed)
- [ ] Decide memory eviction policy for `active_sessions` (time-based or count-based).
- [ ] Extend logging to include user/session metadata for traceability.

## Prompting Systems Overview

### 1. **Runtime Modifying Prompts** (Main Chat Layer)
- Always active in **normal chat** (base VM).
- Self-modifying system prompt — adapts dynamically during the session.  
- Responsible for:
  - Keeping conversation coherent.
  - Adjusting tone, style, and role inline.
  - Ensuring persistent context flow across turns.
- Lightweight → doesn’t require heavy identity scaffolding.

### 2. **Identity-Stabilized Prompting** (Specialized Subsystems)
- Only engaged when models need **stable roles/identities**:
  - **Deep Research** subsystem.
  - **Multi-Agent Collaboration** subsystem.
- Uses a **persistent identity + context frame**:
  - Explicit model roles (Researcher, Analyst, Synthesizer, etc).
  - Context anchoring to prevent drift/instability.
  - Memory integration to ensure consistent persona across long tasks.
- This is where the “kosha” happens → stability layers (memory, identity, context) fused together.

---

### ⚖️ Division of Labor
- **Normal use (chat, projects, artifacts)** → Runtime modifying prompts only.  
- **Heavy/unstable use (deep research, multi-agent work)** → Identity-stabilized prompts layered in.  

This dual-prompt system is why your app feels both flexible (chat-like) and reliable under load (research/collab). It’s basically *adaptive prompting by subsystem*.

##*repo restructure*
/backend/prompts/
    modifying_prompts.py   # Always active in main chat VM
    prompt_manager.py      # Only invoked in specialized subsystems
    prompt_bridge.py       # (Optional) Integration point if you ever want to sync the two


```python
if subsystem in ["web_research", "multi_agent_collab"]:
    # Use identity prompts from prompt_manager
    prompt = prompt_manager.get_system_prompt(...)
else:
    # Use lightweight adaptive prompting
    prompt = modifying_prompts.modify(...)
```

# TODO: Prompting Integration Map

## Base Runtime (Main Chat VM)
- Uses: modifying_prompts.py
- Behavior: Self-modifying prompt + memory_core + system_cache
- Notes:
  * No static system prompt
  * Personality emerges via memory + adaptive modification
  * Always running alongside memory_integration

## Web Research Subsystem
- Uses: prompt_manager.py
- Behavior: Identity-anchored prompts (e.g. "Research Analyst", "Deep Investigator")
- Notes:
  * Required to prevent model drift in long-horizon analysis
  * Tied to memory_integration so research context persists across sessions

## Multi-Agent Collaboration Subsystem
- Uses: prompt_manager.py
- Behavior: Structured roles + identities for stability
- Notes:
  * Each agent VM gets a stable identity prompt
  * Memory integration ensures cross-agent persistence

## Projects Subsystem
- Uses: modifying_prompts.py primarily
- Behavior: Inherits chat-like adaptive prompting
- Notes:
  * Prompt_manager.py may be used when project VM spawns “specialized agents”
  * Otherwise stays freeform

## Artifacts Subsystem
- Uses: modifying_prompts.py
- Behavior: Lightweight prompts, relies on container isolation + memory
- Notes:
  * Artifacts are chosen/suggested, don’t need identity anchoring
  * Only pulls prompt_manager if artifact explicitly requires role stability

## General Rules
- modifying_prompts.py = DEFAULT (chat, projects, artifacts)
- prompt_manager.py = SPECIALIZED (research + collab)
- Memory (core/integration/cache) = ALWAYS ON
- Future bridge: Create `prompt_bridge.py` to harmonize both systems

[session.py Integration To-Do]

1. Deduplicate Enums & Specs
   - Ensure VMState & VMSpecs aren’t redefined across vm_manager.py and session.py.
   - Consider moving them into a shared enums.py/config.py.

2. Hook SecurityMetrics
   - Tie VMSecurityMetrics into combined_security_system.py so violations raise real events.
   - Add policy enforcement (warn, suspend, terminate).

3. Align ContextWindow
   - Verify AIContextWindow.max_tokens with actual loader/model caps (GGUF, ONNX, ARNE).
   - Decide whether to dynamically set context length per model.

4. Persistence Strategy
   - Where is AIVMInstanceMetadata stored? JSON? SQLite? Redis?
   - Ensure dev_session_manager.py uses this schema consistently.

5. Quantization Handling
   - Wire AIModelConfiguration.quantization_enabled/bits into model_loader.py.
   - Add fallbacks for unsupported formats.

6. Error Reporting
   - Connect add_error() → logs in app_server.py & backend dashboards.
   - Optionally feed into the artifact system for replay/debugging.

7. Extend AIVMListResponse
   - Add richer stats (e.g. avg uptime, security status).
   - Expose in your interactive dashboards.

SYSTEM / IDENTITY:   static rules & principles
MEMORY:              external, persistent, semantic retrieval
CONTEXT:             ephemeral, current task goals + imports
WORKING_MEMORY:      semi-ephemeral scratchpad (task/session scoped)
INTERNAL_SCRATCHPAD: per-response scratchpad (volatile, private)
USER_INPUT:          live query/instruction

NEXT STEPS FOR CLAUDE CODE IMPLEMENTATION
-----------------------------------------
[ ] Wire `AutonomousPromptSystem` → `model_loader.py` (LLM call loop)
[ ] Flesh out memory ops in `memory_core.py` + graft logic in `_identify_semantic_grafts`
[ ] Implement real similarity using embeddings (FAISS / local vector DB)
[ ] Add error handling + sandboxing in Artifact system
[ ] Hook Projects VM → Persistent Memory + Prompt Manager
[ ] Security layer: implement rule enforcement + anomaly logging
[ ] Tests: per subsystem (VM, Cache, Prompt, Security)


## Git Integration
- [ ] Wire `GitHubIntegrationManager` into Project VM creation
- [ ] Ensure repo memories are injected into prompt layers
- [ ] Add artifact hook for repo testing/sandbox execution
- [ ] Implement cleanup triggers for stale repos


