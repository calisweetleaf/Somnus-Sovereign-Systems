# Somnus Diagram Library

This directory contains extensive diagrams (Mermaid) covering core and advanced subsystems. These are additive to the main README.

- Research Streaming Protocol: docs/diagrams/research_streaming.md
- Research VM Orchestrator: docs/diagrams/research_vm_orchestrator.md
- Projection Registry & Global Events: docs/diagrams/projections_and_events.md
- Security & Privacy Flows: docs/diagrams/security_privacy.md
- Cache & Memory Architecture: docs/diagrams/cache_memory.md
- Artifact System & Container Overlays: docs/diagrams/artifact_system.md

## Overview Map

```mermaid
flowchart LR
  A[Somnus Platform] --> B[Persistent AI VMs]
  A --> C[Container Overlays]
  A --> D[Memory & Cache]
  A --> E[Security & Privacy]
  A --> F[Research Subsystem]
  A --> G[Projection Registry & Events]

  B --> B1[Research VM Orchestrator]
  C --> C1[Artifact System]
  D --> D1[Unified Cache]
  D --> D2[Semantic Memory]
  E --> E1[JWT / Capability Controls]
  F --> F1[Streaming Manager]
  F --> F2[Deep Research Engine]
  G --> G1[Projection Rules + Transforms]
  G --> G2[Global Event Bus]

  click B1 "./research_vm_orchestrator.md" "Research VM Orchestrator"
  click C1 "./artifact_system.md" "Artifact System"
  click F1 "./research_streaming.md" "Research Streaming"
  click G1 "./projections_and_events.md" "Projections & Events"
  click E1 "./security_privacy.md" "Security & Privacy"
  click D1 "./cache_memory.md" "Cache & Memory"
```

## PNG generation (fallback images)

If your environment does not render Mermaid natively, generate PNGs:

```
npm install -g @mermaid-js/mermaid-cli@10
python scripts/render_mermaid.py --format png --theme default
```

Images will be written to docs/diagrams/img as <docbase>__blockNN.png.

Make usage:
- make diagrams       # PNG
- make diagrams-svg   # SVG
- make diagrams-clean # cleanup

## Legend
- Solid boxes: Core subsystems
- Hollow boxes: Subcomponents
- Clickable links jump to deep-dive diagrams

