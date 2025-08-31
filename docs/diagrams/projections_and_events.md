# Projection Registry & Global Events — Diagrams

Privacy-preserving capability projections and event bus flows.

## Flow: Capability Projection from User Registry

```mermaid
flowchart LR
  UPR[User Registry (encrypted)] --> PR[Projection Registry]
  PR --> TF[Transform Rules]
  TF --> CAP[Capability Graph]
  CAP --> BROK[Capability Broker]
  BROK --> VM[VM/Research/Artifact Subsystems]
  BROK --> DENY{Denied?}
  DENY -- Yes --> AUDIT[Audit Trail + Reason]
  DENY -- No --> GRANT[Grant Capability]
```

## Sequence: Projection Generation and Enforcement

```mermaid
sequenceDiagram
  participant UI as Settings UI
  participant UR as User Registry
  participant PR as Projection Registry
  participant CB as Capability Broker
  participant VM as VM Subsystem

  UI->>UR: Load Profile
  UR-->>UI: Profile (encrypted-at-rest)
  UI->>PR: Generate projection(scope="research")
  PR->>PR: Apply rules + transforms
  PR-->>UI: Projection {capabilities...}
  UI->>CB: Request capability(gpu_acceleration)
  CB->>PR: Verify projection
  PR-->>CB: allowed: true/false
  alt allowed
    CB->>VM: enable_gpu()
    VM-->>CB: ok
    CB-->>UI: GRANTED
  else denied
    CB-->>UI: DENIED(reason)
  end
```

## Requirement Diagram: Privacy and Capability Constraints

```mermaid
requirementDiagram
  requirement local_only {
    id: PRIV-001
    text: All processing must remain local by default
    risk: low
    verifymethod: test
  }
  requirement capability_abstraction {
    id: PRIV-002
    text: AI sees only capabilities, not raw personal data
    risk: medium
    verifymethod: review
  }
  requirement explicit_consent {
    id: PRIV-003
    text: Any expansion of AI visibility requires explicit user consent
    risk: medium
    verifymethod: test
  }
```

## Class Diagram: ResearchEvent Extensions

```mermaid
classDiagram
  class ResearchEvent {
    +string event_id
    +string session_id
    +string user_id
    +string type
    +string query
    +string[] sources
    +string status
    +object result_ref
    +map metadata
    +datetime ts
  }
```

## Pie: Event Type Mix (Illustrative)

```mermaid
pie title Research Events by Type (Day)
  "research_started": 5
  "progress": 48
  "contradiction": 7
  "user_feedback": 12
  "research_completed": 8
  "system": 20
```

