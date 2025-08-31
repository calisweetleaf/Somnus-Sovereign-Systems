# Cache & Memory Architecture — Diagrams

Unified memory+cache with app swapping and background persistence.

## Layered Architecture

```mermaid
graph TD
  A[Persistent Memory] -->|semantic links| B[Knowledge Graph]
  A --> C[Vector Store]
  C --> D[Runtime Cache]
  D --> E[VM State Cache]
  D --> F[Model State Cache]
  D --> G[Research Namespace]
  E --> H[App Swapping]
```

## Sequence: App Swapping with Cache

```mermaid
sequenceDiagram
  participant Chat as Chat VM
  participant Cache as Unified Cache
  participant Res as Research VM

  Chat->>Cache: set(chat_vm_state, priority=HIGH)
  Chat->>Cache: set(user_context, priority=CRITICAL)
  Chat-->>User: transitioning...
  Res->>Cache: get(research_session_context)
  alt cache hit
    Cache-->>Res: context
  else miss
    Res->>Memory: reconstruct from persistent
    Memory-->>Res: reconstructed
    Res->>Cache: set(research_session_context)
  end
  Res-->>User: research ready (~50ms)
```

## State Diagram: Cache Entry Lifecycle

```mermaid
stateDiagram-v2
  [*] --> HOT
  HOT --> WARM: LRU decay
  WARM --> COLD: eviction candidate
  COLD --> HOT: re-reference
  COLD --> EVICTED: background cleanup
  EVICTED --> [*]
```

## Gantt: Background Tasks

```mermaid
gantt
  title Cache & Memory Background Processes
  dateFormat  X
  axisFormat  %s
  section Cache
  Eviction Sweep       :done,    e1, 0, 20
  Persistence Flush    :active,  e2, 5, 30
  section Memory
  Consolidation        :         m1, 10, 25
  Importance Decay     :         m2, 15, 10
```

## PNG Fallbacks
- docs/diagrams/img/cache_memory__block01.png (Layered Architecture)
- docs/diagrams/img/cache_memory__block02.png (Sequence: App Swapping)
- docs/diagrams/img/cache_memory__block03.png (State: Cache Entry)
- docs/diagrams/img/cache_memory__block04.png (Gantt: Background Tasks)

```mermaid
gantt
  title Cache & Memory Background Processes
  dateFormat  X
  axisFormat  %s
  section Cache
  Eviction Sweep       :done,    e1, 0, 20
  Persistence Flush    :active,  e2, 5, 30
  section Memory
  Consolidation        :         m1, 10, 25
  Importance Decay     :         m2, 15, 10
```

