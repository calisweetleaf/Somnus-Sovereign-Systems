# Artifact System & Container Overlays — Diagrams

Unlimited execution via disposable containers per artifact.

## Component Architecture

```mermaid
graph LR
  VM[Persistent AI VM] --> ORCH[Artifact Orchestrator]
  ORCH --> RT[Container Runtime]
  RT --> C1[Artifact Container A]
  RT --> C2[Artifact Container B]
  C1 --> WS1[(Real-time WS Stream)]
  C2 --> WS2[(Real-time WS Stream)]
  ORCH --> FS[(Persistent Workspace Volumes)]
```

## Sequence: Artifact Execution

```mermaid
sequenceDiagram
  participant VM as VM (AI Orchestrator)
  participant OR as Overlay Manager
  participant RT as Container Runtime
  participant WS as WebSocket Stream

  VM->>OR: create_overlay(config)
  OR->>RT: start_container(config)
  RT-->>OR: container_id
  OR-->>VM: overlay_ready
  VM->>RT: exec(command, env)
  RT->>WS: stream stdout/stderr
  WS-->>UI: live updates
  VM->>RT: stop_container()
  RT-->>VM: stopped
```

## Class Diagram: Overlay/Artifact Config

```mermaid
classDiagram
  class ContainerConfig {
    +string base_image
    +bool gpu_enabled
    +string network_mode
    +map env
    +list tools
    +limits resources
  }
  class Artifact {
    +UUID id
    +string name
    +string type
    +string code
    +list capabilities
  }
  class OverlayManager {
    +create_overlay(Artifact, ContainerConfig)
    +execute()
    +stop()
  }
  OverlayManager o-- ContainerConfig
  OverlayManager o-- Artifact
```

## Flowchart: Security Separation

```mermaid
flowchart TD
  A[VM Intelligence Layer] -->|API| B[Overlay Boundary]
  B --> C[Container Execution Layer]
  C --> D[Ephemeral FS]
  C --> E[Network Policies]
  C --> F[GPU Passthrough]
  D -->|cleanup| G[Auto Remove]
```

