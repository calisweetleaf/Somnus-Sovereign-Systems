# Research VM Orchestrator — Diagrams

End-to-end view of research VM provisioning, initialization, services, and runtime loops.

## Flow: VM Initialization & Configuration

```mermaid
flowchart TD
  A[create_research_vm] --> B[Provision base VM]
  B --> C[Setup OS packages & tools]
  C --> D[Create workspace dirs]
  D --> E[Python venv + research packages]
  E --> F[Display server (Xvfb)]
  F --> G[Browser engines (Chromium/Firefox)]
  G --> H[Install research extensions]
  H --> I[Load AI models]
  I --> J[Write analysis scripts]
  J --> K[Init research DB (SQLite)]
  K --> L[Start in-VM Flask API (8888)]
  L --> M[File watcher + processing]
  M --> N[Memory integration]
  N --> O[Prompt configuration]
  O --> P[Capability verification]
  P --> Q[RUNNING]
```

## Class Diagram: Orchestrator Core Types

```mermaid
classDiagram
  class ResearchVMOrchestrator {
    +initialize()
    +create_research_vm(user_id, session_id, cfg)
    +get_vm_status(vm_id)
    +shutdown()
    -_setup_research_environment()
    -_setup_browser_engines()
    -_load_research_models()
    -_setup_research_workspace()
    -_setup_memory_integration()
    -_setup_file_processing()
    -_setup_research_prompts()
    -_verify_capabilities()
  }
  class ResearchVMConfiguration {
    +ResearchVMType vm_type
    +VMSpecs vm_specs
    +BrowserEngine[] browser_engines
    +bool headless_mode
    +ResearchCapability[] enabled_capabilities
    +string[] ai_models
    +bool enable_vpn
    +dict proxy_settings
  }
  class ResearchVMState {
    +UUID vm_id
    +VMState vm_state
    +map browser_sessions
    +map loaded_models
    +set active_research_sessions
    +float cpu_usage
    +float memory_usage
    +float disk_usage
  }
  ResearchVMOrchestrator o-- ResearchVMState
  ResearchVMOrchestrator o-- ResearchVMConfiguration
  ResearchVMOrchestrator ..> MemoryManager
  ResearchVMOrchestrator ..> ModelLoader
  ResearchVMOrchestrator ..> IntelligentFileProcessor
  ResearchVMOrchestrator ..> SystemPromptManager
```

## State Diagram: VM Lifecycle

```mermaid
stateDiagram-v2
  [*] --> CREATING
  CREATING --> RUNNING
  RUNNING --> SUSPENDED: inactive > 2h
  SUSPENDED --> RUNNING: restore
  RUNNING --> ERROR: capability verification failed
  ERROR --> [*]
```

## ER Diagram: Research Database Schema

```mermaid
erDiagram
  research_sessions {
    TEXT session_id PK
    TEXT user_id
    TEXT query
    TIMESTAMP created_at
    TEXT status
    JSON metadata
  }
  research_sources {
    TEXT source_id PK
    TEXT session_id
    TEXT url
    TEXT title
    TEXT content
    TIMESTAMP extracted_at
    REAL quality_score
    REAL credibility_score
    REAL bias_score
    JSON metadata
  }
  research_analysis {
    TEXT analysis_id PK
    TEXT source_id
    TEXT analysis_type
    JSON result
    TIMESTAMP created_at
  }
  research_cache {
    TEXT cache_key PK
    TEXT content
    BLOB embeddings
    TIMESTAMP created_at
    TIMESTAMP expires_at
    INTEGER access_count
  }
  research_sessions ||--o{ research_sources : has
  research_sources ||--o{ research_analysis : analyzed_by
```

## Sequence: Browser Extension Ingestion

```mermaid
sequenceDiagram
    participant EXT as Browser Extension
    participant API as In-VM Flask API :8888
    participant DB as SQLite research.db
    participant EV as Event Bus

    EXT->>API: POST /api/research/extracted_data {title,url,abstract,citations}
    API->>DB: INSERT INTO research_sources(...)
    DB-->>API: rowid
    API-->>EXT: 200 {success:true, source_id}
    API->>EV: research_source_ingested(session_id, source_id)
```

## Gantt: Initialization Timeline (Example)

```mermaid
gantt
  title Research VM Initialization
  dateFormat  HH:mm
  axisFormat  %H:%M
  section System
  Packages & Tools      :done,    des1, 00:00, 00:02
  Workspace            :done,    des2, 00:02, 00:01
  Python/Deps          :active,  des3, 00:03, 00:03
  Playwright/Selenium  :         des4, 00:06, 00:02
  Browsers & Profiles  :         des5, 00:08, 00:02
  Models Load          :         des6, 00:10, 00:04
  DB Init              :         des7, 00:14, 00:01
  API Launch           :         des8, 00:15, 00:01
  Capability Verify    :         des9, 00:16, 00:02
```

