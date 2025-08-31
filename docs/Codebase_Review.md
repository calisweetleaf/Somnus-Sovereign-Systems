Somnus Sovereign Chat System – Architecture & Core Modules
Overview and Philosophy

Somnus Sovereign Chat is positioned as a sovereign, persistent AI operating system. The design rejects the typical “AI‑as‑a‑service” model in favour of a local‑first architecture in which every agent is given its own persistent virtual machine (VM). These VMs never reset, allowing continuous learning, tool installation and long‑term memory. Only the Artifact System uses disposable container overlays; all other subsystems run natively inside the AI VM. The system emphasises user sovereignty, cost‑free local operation, and a clear separation between the intelligence layer (the VM) and computational workloads (containers)【388067229034279†L112-L121】. A multi‑agent collaboration core enables swarms of LLMs to coordinate using a structured protocol, while a unified caching layer and memory core preserve state across sessions and subsystems.
Documentation Summary
1 – Somnus Co‑Development Roadmap

This roadmap file is a long feature catalogue outlining completed, planned and research‑stage capabilities. Major categories include:

    Core system enhancements – accelerated file processing, GitHub cloning/indexing, dynamic memory management, semantic search, intelligent clipboard history, local automation scripting and automated digests. The accelerated file processor uses optimized parallel algorithms to ingest and embed any local file type, ensuring the VM’s knowledge base is always current【918727156888128†L6-L19】. The GitHub integration module can clone repositories, recursively discover files and index them into the memory core【918727156888128†L80-L105】.

    Developer tools & code intelligence – natural‑language codebase querying, refactoring suggestions, real‑time code linting, and a Git command assistant. These modules are intended to run inside the persistent VM and integrate with the memory core.

    Creative & multimedia features – offline image generation (Stable Diffusion), local audio synthesis, and video analysis. These features are planned to run as artifact overlays; models are loaded inside dedicated containers but results are managed by the VM.

    System & user experience – adaptive UI layouts, accessibility options, and guided onboarding. Many of these are research or planned features.

The roadmap clearly distinguishes between completed, planned and research status, making it a useful checklist of future work.
2 – Somnus Artifact System

This paper proposes a dual‑layer execution model for unlimited AI‑generated content【388067229034279†L125-L156】. Key points include:

    Separation of Concerns – AI agents run in persistent VMs, where their state, learned tools and orchestration logic live【388067229034279†L116-L124】. Computational tasks run in disposable container overlays attached to these VMs【388067229034279†L139-L151】. This prevents tool bloat in the VM while allowing unlimited capabilities.

    Resource Efficiency – the system avoids installing heavy libraries in the VM; instead it spins up specialized overlays (ML/AI training, video processing, research automation) on demand【388067229034279†L139-L151】. Overlays are created and destroyed per task, leaving the VM clean.

    AI‑Container Interface – a bridge class orchestrates overlays from the VM. It analyzes artifact requirements, creates specialized containers, monitors execution and cleans up afterwards.

    On‑Demand Capability Management – a settings/activation framework allows capabilities to be enabled when requested. Capabilities define required tools, resource allocations and network policies.

    Dual‑Trigger Execution – both AI agents and users can initiate overlay execution. The UI exposes triggers (buttons) to activate capabilities, while the AI can start overlays programmatically.

The Artifact System is the only dockerised component. All other subsystems (memory, model loading, multi‑agent orchestration, research) run natively inside the VM. Security is achieved by keeping the AI’s intelligence layer separate from the disposable overlays【388067229034279†L116-L124】.
3 – Somnus VM Architecture

This module defines the configuration and lifecycle for the persistent VMs. The introductory comments state the philosophy: each AI agent runs in a VM that never resets and can progressively accumulate tools, ensuring resource efficiency and user control【659167686334206†L4-L13】. Hardware specifications (vCPUs, memory, storage, network ports) and personality settings (agent name, specialization, creativity, preferred IDE/shell/browser) are defined【659167686334206†L68-L103】. VM instances include backup schedules, auto‑suspend timers, tracking of installed tools and learned capabilities, and metrics such as uptime and efficiency rating【659167686334206†L104-L145】. A VM pool configuration governs the maximum number of concurrent VMs, auto‑scaling and resource limits for the host【659167686334206†L148-L167】.
4 – Somnus Unified Performance Cache (system_cache.py)

The cache module provides a runtime caching layer that complements the memory core. It supports various namespaces – global, session, user, VM, artifact, model, research and system【837333225491083†L41-L49】 – enabling fine‑grained control over cached items. Entries store metadata (priority, TTL, access times) and have LRU‑based eviction scoring【837333225491083†L41-L56】. The cache engine records metrics (hits, misses, evictions) and persists data to disk using a SQLite metadata database and a separate shelf for values. Session‑aware namespacing and automatic cleanup allow context to be swapped or restored across subsystems and sessions【837333225491083†L8-L14】.
5 – Agent Collaboration Core

The collaboration core defines how multiple AI agents coordinate tasks. Important elements are:

    Message Types and Agent Capabilities – the module defines message types such as task proposals, acceptances, delegations, work progress, results and synthesis requests【276281714377717†L24-L33】. Agents advertise capabilities (text analysis, code generation, research synthesis, creative writing, logical reasoning, data processing, technical documentation, problem solving)【276281714377717†L48-L57】 and maintain profiles with collaboration preferences and independence thresholds【276281714377717†L60-L72】.

    Agent Profiles and Decision Making – each AgentProfile includes specialization scores, collaboration preference, independence threshold and communication style【276281714377717†L60-L72】. When processing user input, an agent analyses whether to handle the task alone or collaborate. The decision uses heuristics (keywords, complexity, length, task type) and the agent’s preference.

    Message Bus and Communication Protocol – the companion protocol module defines priority levels, protocol versions and message statuses (pending, delivered, processing, completed, failed, timeout)【528014049555495†L26-L47】. The AgentCommunicationMessage carries routing data, content and metadata, and provides integrity verification via checksums【528014049555495†L61-L115】. The TaskDelegationProtocol offers methods to create task proposals, acceptances, updates and completion messages with explicit deadlines, estimated effort and success criteria【528014049555495†L137-L169】.

    Task Delegation and Synthesis – the collaboration core can decompose complex queries into subtasks, delegate them to other agents based on capabilities, and aggregate results into a final response. It keeps track of active tasks, pending responses and shared workspaces, and integrates with the memory core so that results are stored permanently.

This core operates inside the AI VM; it uses the memory manager, model loader and session manager to maintain context and to load models for each agent.
System Integration & Subsystems

The system can be viewed as layered modules:

    Hardware & Hypervisor – the host machine runs a hypervisor (e.g. libvirt/QEMU) on which persistent AI VMs are created. VM pool settings control resource allocation, auto‑scaling and backup schedules【659167686334206†L148-L167】.

    Persistent AI VM Layer – each VM contains the following subsystems:

        Memory Core & Cache – long‑term semantic memory stores embeddings of all ingested data. The unified cache provides hot‑storage with namespacing for quick retrieval【837333225491083†L41-L56】.

        Model Loader – dynamically loads local models (GGUF) or proxies API models. It supports quantization, hardware acceleration and hot‑swapping.

        Session/VM Manager – creates and tracks VM instances, enforces resource policies and maintains personality settings【659167686334206†L4-L13】.

        File Upload & Processing – accepts files of any type, uses accelerated file processing to embed and index content, and integrates with git_integration_module for cloning and deep indexing of repositories【918727156888128†L6-L19】【918727156888128†L80-L105】.

        Web Research System – performs privacy‑first web searches and browser automation; results are fed into the memory core.

        Security Modules – red‑team toolkit, blue‑team playbook and purple‑team orchestrator provide offensive and defensive security, network isolation, container escape detection and vulnerability scanning.

        Plugin System – supports dynamic discovery, sandboxing and hot‑reloading of plugins, enabling extensibility.

        Agent Collaboration Core – manages swarms of AI agents, task decomposition, inter‑agent messaging and result synthesis【276281714377717†L24-L33】.

    Artifact System Overlay (Docker) – when users or the AI need to run heavy computations or specialized tools (e.g., training large models, video processing, browser automation), the VM orchestrates the creation of disposable container overlays. These overlays have full resource access and can install any package; after execution the overlay is destroyed, keeping the VM clean【388067229034279†L139-L151】. The VM communicates with overlays via a bridge and stores outputs as persistent artifacts.

    User Interface – a local chat UI provides access to all functionality. Capability triggers allow users to activate overlays, run research tasks, manage projects and collaborate with agents. Progressive enhancement means the base interface stays lightweight; overlays slide in when needed.

Text‑Based Architecture Diagram

[Host Machine / Hypervisor]
    └── VM Pool Manager (creates up to N persistent AI VMs)
           └── [Persistent AI VM]
                 ├── Memory Core & Unified Cache (global, session, user, VM, model, research namespaces)【837333225491083†L41-L49】
                 ├── Model Loader & Quantization
                 ├── Session/VM Settings Manager【659167686334206†L4-L13】
                 ├── File Upload & Accelerated Processing【918727156888128†L6-L19】
                 ├── Git/Code Integration & Repository Indexer【918727156888128†L80-L105】
                 ├── Web Research & Browser Automation
                 ├── Security Layer (Red/Blue/Purple Team modules)
                 ├── Plugin System (discovery, hot‑reloading)
                 ├── Agent Collaboration Core & Communication Protocol【276281714377717†L24-L33】
                 ├── Application Logic / UI Server
                 └── Artifact System Orchestrator
                        └── Disposable Container Overlays (Docker)
                              ├── ML/AI training overlay
                              ├── Video processing overlay
                              ├── Research automation overlay
                              └── … (specialized per task)【388067229034279†L139-L151】

Critical Subsystems & Feature Checklist
Subsystem / Feature	Summary	Status
Persistent VM Manager	Creates and configures VMs, including hardware specs, personalities, backup schedules and resource policies【659167686334206†L4-L13】.	Implemented
Memory Core & Unified Cache	Long‑term semantic memory with encryption; unified cache for hot data with namespaces and LRU eviction【837333225491083†L41-L56】.	Implemented
Model Loader	Loads local models (GGUF) and proxies remote APIs; supports quantization, hot‑swapping and hardware acceleration.	Implemented
Accelerated File Processing	Optimized ingestion and embedding of all file types; integrates with file upload system【918727156888128†L6-L19】.	Completed
Git Integration Module	Clones and deeply indexes repositories; classifies files and feeds them into memory【918727156888128†L80-L105】.	Completed
Artifact System	Dual‑layer architecture with container overlays; unlimited tool installation; orchestrated by VM【388067229034279†L116-L124】.	Implemented
Agent Collaboration Core	Profiles agents, decides when to collaborate and uses structured messages to delegate tasks and synthesize results【276281714377717†L24-L33】.	Implemented
Agent Communication Protocol	Defines message priorities, statuses, checksums and a task delegation protocol【528014049555495†L26-L47】.	Implemented
Web Research & Browser Agent	Sandbox browser for AI learning (planned); integrates with memory core for local web reading.	Planned
Dynamic Memory Management & Tiering	Moves cold data to slower storage; compresses or evicts rarely used items.	Planned
Semantic Search & Knowledge Graph	Builds vector index of local files and supports natural‑language queries.	Planned
Clipboard Monitor & Contextual Actions	Tracks clipboard history and suggests context‑aware actions.	Planned
Local Automation Engine	No‑/low‑code rule engine for automating file and system events.	Research
Digest Generator	Summarizes recent activities and content into daily/weekly briefs.	Planned
Developer Tools	Codebase query/refactoring assistant, linting, Git command assistant with visual graphs.	Planned/Research
Creative Tools	Offline image and audio generation; video analysis.	Planned/Research
UI & Accessibility	Adaptive layouts, themes, keyboard navigation and contextual help.	Planned/Research
Security	Red‑team toolkit, blue‑team playbook, purple‑team orchestrator for continuous testing and defence.	Implemented

Observations & Gaps

    Many advanced features (semantic search, dynamic memory tiering, clipboard monitoring, automation, digest generation) are planned or research‑stage; documentation exists but code may be incomplete. These areas represent technical debt or future work.

    The roadmap outlines numerous creative and UI enhancements that are not yet implemented; further design and development is needed.

    The artifact system specification is detailed, but implementation specifics (container orchestration engine, security policies) may need further elaboration.

    There is minimal documentation on the exact API endpoints or UI flows for multi‑agent collaboration; additional design docs could improve clarity.

Getting Started Guide
For a New AI Agent (LLM)

    Prepare the Host – install the hypervisor (libvirt/QEMU) and ensure your machine has enough CPU, memory and storage. Install Python 3.11 and the required dependencies using pip install -r requirements.txt.

    Boot the System – run main.py or the web server entry point. The VM Pool Manager will initialize persistent VM instances based on VMPoolSettings. Default VMs use 4 vCPUs, 8 GB RAM and 100 GB disk【659167686334206†L68-L72】.

    Create a VM for the Agent – use the VMSettingsManager (via API or CLI) to create a new VM instance. Specify a name, user ID, hardware specs and personality (e.g., research‑oriented with preferred IDE = VS Code and research methodology = systematic)【659167686334206†L68-L103】.

    Load an LLM – within the VM, call the ModelLoader to load a local GGUF model or configure an API proxy. The model will be cached for reuse. For multi‑agent setups, load a model per agent and assign each agent a unique profile specifying its capabilities【276281714377717†L60-L72】.

    Ingest Data – upload files via the file upload API or UI. The accelerated file processor will embed and index them【918727156888128†L6-L19】. Use the git integration to clone repositories for code comprehension【918727156888128†L80-L105】. All embeddings and metadata are stored in the memory core.

    Enable Tools & Plugins – install any tools or plugins you need inside the VM (e.g., analysis libraries, notebooks). The VM persists installed tools, unlike disposable overlays.

    Run Tasks and Collaborate – interact through the chat UI. Use the multi‑agent core to spawn additional agents with different capabilities. The system will decide whether to handle tasks solo or delegate. You can manually instruct agents to collaborate or specify swarms.

    Use the Artifact System – when tasks require heavy computation or additional tools, trigger the Artifact System from the UI. The VM will create a specialized container overlay with the requested tools and run the code. Outputs are stored as artifacts accessible from the persistent VM【388067229034279†L116-L124】.

    Monitor and Manage – view VM metrics (uptime, resource usage, installed tools, learned capabilities), manage backups and snapshots via the VM manager. Use the security modules to run periodic red/blue team tests.

For Developers Extending the System

    Understand the architecture – familiarize yourself with the persistent VM philosophy, dual‑layer artifact system, unified cache and multi‑agent collaboration. Review the core modules under /core and /backend.

    Follow modular design – new features should integrate with the VM via well‑defined interfaces. For example, new research capabilities should use the memory core for storage and the agent collaboration core for synthesis.

    Implement new capabilities – to add a tool that requires external libraries, create a capability definition and configure an overlay image. The artifact system will handle lifecycle and resource allocation.

    Add agents – extend AgentProfile to include new capability types, and implement corresponding processing logic. Use the communication protocol to define new message types if needed.

    Respect security – ensure that plugins or overlays do not compromise the VM. Use the red/blue team toolkit to test for vulnerabilities and follow the separation of concerns model.

For LLM Developers (Adding New Models)

    Model Packaging – convert your model to a local format (e.g., GGUF for quantized models) and place it in the models directory.

    Register with ModelLoader – add your model entry to the models configuration file and specify any quantization or hardware preferences.

    Load and Test – use the ModelLoader API to load your model into a VM. Verify that it responds correctly and that embeddings can be generated. Update agent profiles to use the new model when appropriate.

Conclusion

Somnus Sovereign Chat introduces a radical departure from transient, cloud‑based AI services. Its architecture revolves around persistent, user‑owned VMs that host all intelligence and memory, coupled with disposable container overlays for unlimited computational tasks【388067229034279†L116-L124】. A unified cache and memory system preserves context across sessions, and an advanced collaboration core orchestrates swarms of agents. While many planned features remain to be implemented, the foundation already offers a powerful, sovereign AI environment that can be extended and refined into a truly autonomous OS.
Additional Deep Dive – Projects, Research & Plugin Subsystems

The following sections provide a closer look at three pillars of Somnus Sovereign Chat: the projects subsystem, the research subsystem and the plugin system. These analyses drill down into the modules, features and patterns that make each subsystem work and highlight persistent versus containerized components.
Projects Subsystem Overview

Somnus treats each project as an independent AI computer. When a user creates a project via the API, the system spins up a dedicated VM with its own hardware specs, loaded model and persistent file system. The project VM hosts multiple managers that together form the “intelligence” of the project. Core modules include:

    project_api.py – Exposes FastAPI endpoints for lifecycle operations (create, update, delete), file uploads, artifact creation, collaboration tasks and automation rules. A WebSocket manager streams real‑time events such as file uploads, analysis completion and automation triggers to clients【226603923827494†L0-L11】【226603923827494†L125-L176】.

    project_core.py – Defines the ProjectManager and the ProjectSpecs used by the VM manager. This module emphasises that a project is a persistent VM plus a model and intelligence; it handles creation, initialization, monitoring and destruction【471471320384715†L360-L378】【471471320384715†L381-L416】.

    project_vm_manager.py – Allocates resources (disk, CPU, memory), clones a base image, allocates SSH/VNC/Web ports and starts the VM. The manager tracks VM state (creating, running, suspended, maintenance, error) and handles error recovery and cleanup【306324994131137†L4-L13】【306324994131137†L140-L166】.

    project_files.py – Provides unlimited storage and intelligent categorization of files. It monitors the VM’s file system, calculates hashes and metadata, organizes files by category (documents, code, data, images, audio, etc.) and triggers analysis jobs. Each ProjectFile tracks versions, relationships and key concepts【623192288212160†L0-L13】【623192288212160†L76-L150】.

    project_intelligence.py – Runs inside the VM to automatically analyze files, extract key concepts, build a knowledge graph and generate insights. It uses embedding models (sentence transformers) and periodic scheduling to process new files and produce suggestions or pattern detections【59610452569339†L0-L10】【59610452569339†L69-L116】.

    project_knowledge.py – Extracts facts, concepts, procedures and insights from file content and stores them in a semantic vector database (ChromaDB) and a metadata database (SQLite). It builds a dynamic knowledge graph with nodes and edges, offers semantic search and synthesizes information across sources【128503152430992†L111-L135】.

    project_memory.py – Bridges project events to the global memory. It defines memory types (project creation, file upload, knowledge discovery, collaboration session, automation execution, artifact creation, project insight) and stores entries with importance scoring. Context objects maintain project‑scoped namespaces and metadata【851182011924241†L0-L10】【851182011924241†L76-L117】.

    project_artifacts.py – Manages project artifacts such as reports, dashboards, notebooks and apps. It sets up workspace directories inside the VM, integrates with knowledge and memory systems, tracks versions and allows execution of interactive notebooks or dashboards within the VM【790614519062070†L31-L79】【790614519062070†L130-L174】.

    project_collaboration.py – Integrates with the multi‑agent collaboration core. It defines project roles and tasks, computes specialization scores based on project content and coordinates collaborative tasks across agents【366929427464189†L4-L10】【366929427464189†L44-L66】.

    project_automation.py – Defines triggers (schedules, file changes, thresholds, manual commands), actions (analyze files, generate reports, backup the VM, sync knowledge) and dependencies. It monitors metrics and adapts scheduling to optimize performance【471471320384715†L422-L471】.

These modules collectively provide a full AI‑OS environment per project. All services run natively in the project VM; only when heavy computations are needed (e.g., GPU‑intensive training) does the system spin up a disposable container via the Artifact System overlay.
Research Subsystem

The research subsystem provides an orchestrated pipeline for deep information gathering and synthesis, enabling multi‑agent research with persistent state and streaming updates. Core components include:

    deep_research_planner.py – Generates research plans with a detailed structure: complexity level, domain expertise, collaboration mode and research phases. Plans contain sub‑questions, tasks, participants and schedules; the module also defines schedules for each collaboration mode (solo, parallel, hierarchical, democratic)【463113220428428†L0-L18】【463113220428428†L1400-L1491】.

    research_engine.py – Executes plans by coordinating memory, the browser agent, collaboration core and report exporter. It runs in asynchronous loops with state transitions (initializing, running, paused, waiting for input) and tracks metrics (steps completed, memory queries, browser tasks executed, synthesis operations, contradictions resolved)【420084650278688†L0-L18】【420084650278688†L73-L109】.

    research_stream_manager.py – Streams research events via WebSockets. It supports event types like session start/end, search progress, contradiction detection, user intervention and feedback; tracks connection states and enforces rate limits【782266385891420†L0-L68】【782266385891420†L137-L188】.

    ai_browser_research_agent.py – Automates web browsing and content extraction using Playwright/Selenium. It performs searches, follows links, extracts text and media, and uses AI models to summarize and tag content【612563952058091†L0-L18】【612563952058091†L83-L171】.

    research_intelligence.py – Evaluates research quality across dimensions (completeness, diversity, credibility, depth, bias balance) and provides suggestions. It defines insight types (contradiction, gap, bias, confirmation, fact, best practice) and calculates scores for each【127152630838663†L0-L14】【127152630838663†L47-L70】.

    research_cache_engine.py – Implements a cognitive resonance cache that uses embeddings to prefetch relevant data and accelerate recall【729663261335303†L0-L17】【729663261335303†L54-L90】.

    research_session.py – Persists session data: the question, plan, search results, contradictions, biases, entity graphs and final outputs. It enumerates states and depth levels and assigns cognitive scores to research entities【217103768999806†L0-L21】【217103768999806†L49-L112】.

    report_exporter.py – Exports research results in multiple formats, creates artifacts in the project environment and logs high‑quality sources and contradictions back to memory【705944260835398†L0-L11】【705944260835398†L134-L169】.

Collectively, these modules enable collaborative research: the planner decomposes tasks; the engine executes them using agents and browser automation; the stream manager communicates progress; the intelligence module assesses quality; sessions persist for later use; and the exporter produces final reports.
Plugin System

The plugin architecture allows developers to extend Somnus with new capabilities while maintaining security and performance. Core elements include:

    plugin_base.py – Defines fundamental interfaces: PluginCapability enumerates advanced abilities (memory integration, consciousness access, multi‑agent coordination, temporal manipulation)【637635287652795†L28-L37】; PluginState tracks lifecycle states; PluginContext carries session, user and resource info; PluginMetrics records runtime metrics; PluginManifest stores metadata, dependencies and permissions; interfaces define initialization, activation, deactivation, state management and consciousness synchronization【637635287652795†L39-L48】【637635287652795†L50-L75】.

    plugin_manager.py – The orchestrator that discovers, loads and manages plugins. It supports hot reload, secure sandboxing, automatic FastAPI registration, memory persistence via ChromaDB, multi‑agent orchestration and community marketplace integration【646781094433336†L0-L13】. It defines plugin statuses (discovered, loaded, active, error, disabled) and types (API extension, UI component, agent action, memory extension, workflow node, tool integration, research module, creative engine)【646781094433336†L52-L72】.

    discovery.py – Scans the file system and registries, builds a PluginIndex with metadata and dependencies and uses a DependencyResolver to compute safe load orders while handling circular dependencies【331963270571610†L29-L50】【331963270571610†L130-L189】.

    hot_reload.py – Implements reload types (hot, warm, cold, cascade) and defines how much state to preserve. It serializes plugin state (including memory references and UI state) for restoration after reload and tracks dependencies to cascade reload dependents【283181001791637†L39-L90】.

    orchestration.py – Coordinates plugin execution through workflows. Execution modes (sequential, parallel, pipeline, graph, reactive), priorities and workflow states are defined. The ExecutionQueue prioritizes requests, while the ResourceMonitor enforces per‑plugin resource limits【484422261244096†L25-L46】.

    marketplace.py – Implements a plugin registry with listing and rating data. The TrustSystem calculates trust scores based on developer reputation, provider type, rating averages, download counts and security scan results【625857687112005†L83-L111】.

    security.py – Performs static code analysis, signature verification and sandboxed execution. It removes dangerous built‑ins, restricts module imports and ensures that untrusted code runs with strict resource limits【343962154087775†L60-L82】.

Together, these modules provide a secure, extensible ecosystem. Plugins can add new API endpoints, UI components, tools or agent actions without compromising the core system. The marketplace and trust system encourage community contributions while maintaining safety.
Extended Hierarchical Architecture Diagram

The following diagram extends the earlier architecture with details about the projects, research and plugin subsystems:

Somnus Sovereign Chat
└── Persistent AI VM (per user)
    ├── Memory & Unified Cache
    ├── Model Loader & Session Manager
    ├── Projects Subsystem (per project VM)
    │   ├── VM Manager – persistent VM with model & resources【306324994131137†L4-L13】
    │   ├── File Manager – unlimited storage & categorization【623192288212160†L0-L13】
    │   ├── Intelligence Engine – automatic analysis & insights【59610452569339†L69-L116】
    │   ├── Knowledge Base – semantic search & graph【128503152430992†L111-L135】
    │   ├── Memory Manager – logs project events【851182011924241†L76-L117】
    │   ├── Artifact Manager – persistent IDE/artifacts【790614519062070†L31-L79】
    │   ├── Collaboration Manager – connects to agent core【366929427464189†L44-L66】
    │   └── Automation Engine – triggers & actions【471471320384715†L422-L471】
    │
    ├── Artifact System (Docker overlay)
    │   └── Containerized tool execution & persistent IDE【388067229034279†L116-L124】
    │
    ├── Multi‑Agent Collaboration Core
    │   └── Profiles, messaging & consensus (see earlier summary)
    │
    ├── Research Subsystem
    │   ├── Planner – complexity, collaboration modes & schedules【463113220428428†L0-L18】
    │   ├── Engine – execution & metrics【420084650278688†L0-L18】
    │   ├── Browser Agent – search & extraction【612563952058091†L0-L18】
    │   ├── Intelligence – quality assessment【127152630838663†L0-L14】
    │   ├── Cache & Session – cognitive resonance【729663261335303†L0-L17】
    │   └── Report Exporter – artifact generation【705944260835398†L0-L11】
    │
    └── Plugin System
        ├── Base Interfaces & Manifest【637635287652795†L28-L37】
        ├── Manager & Discovery【646781094433336†L0-L13】【331963270571610†L29-L50】
        ├── Hot Reload & State Preservation【283181001791637†L39-L90】
        ├── Orchestration & Resource Monitor【484422261244096†L25-L46】
        ├── Marketplace & Trust System【625857687112005†L83-L111】
        └── Security & Sandbox【343962154087775†L60-L82】

Focused Getting Started Tips

    Projects: Use the project_api endpoints to create a project and specify hardware specs, enable collaboration/automation and upload files. The VM manager will provision a new VM and load the requested model. Once running, the file manager, intelligence engine and knowledge base start automatically; you can monitor progress via WebSockets.

    Research: Invoke the deep_research_planner to generate a plan; choose a collaboration mode depending on whether you want agents to work in parallel, hierarchy or democratic fashion【463113220428428†L1400-L1491】. Run the plan with the research_engine, monitor events via the stream manager and refine as needed. Final reports are saved as artifacts in the project.

    Plugins: To extend the system, create a plugin with a manifest.json specifying its name, type, version and permissions. Implement the interfaces from plugin_base.py and test locally. Register it with the plugin manager; on load it will be discoverable through the API or UI. Use the marketplace for distribution and the trust system for ratings. Observe sandbox restrictions and sign your code for trust.

Additional Observations

    Persistent VMs mean that each project can accumulate its own toolchain and knowledge; however, this may consume significant host resources when many projects run concurrently.

    The research subsystem provides a robust pipeline but may require fine‑tuning for domain‑specific research tasks or bias mitigation.

    The plugin ecosystem is powerful but complex; developers should carefully manage dependencies and test for circular dependencies and resource limits. Graph visualization tools could aid understanding of plugin relationships.

Somnus Sovereign Chat System – Architecture & Core Modules
Overview and Philosophy

Somnus Sovereign Chat is positioned as a sovereign, persistent AI operating system. The design rejects the typical “AI‑as‑a‑service” model in favour of a local‑first architecture in which every agent is given its own persistent virtual machine (VM). These VMs never reset, allowing continuous learning, tool installation and long‑term memory. Only the Artifact System uses disposable container overlays; all other subsystems run natively inside the AI VM. The system emphasises user sovereignty, cost‑free local operation, and a clear separation between the intelligence layer (the VM) and computational workloads (containers)【388067229034279†L112-L121】. A multi‑agent collaboration core enables swarms of LLMs to coordinate using a structured protocol, while a unified caching layer and memory core preserve state across sessions and subsystems.
Documentation Summary
1 – Somnus Co‑Development Roadmap

This roadmap file is a long feature catalogue outlining completed, planned and research‑stage capabilities. Major categories include:

    Core system enhancements – accelerated file processing, GitHub cloning/indexing, dynamic memory management, semantic search, intelligent clipboard history, local automation scripting and automated digests. The accelerated file processor uses optimized parallel algorithms to ingest and embed any local file type, ensuring the VM’s knowledge base is always current【918727156888128†L6-L19】. The GitHub integration module can clone repositories, recursively discover files and index them into the memory core【918727156888128†L80-L105】.

    Developer tools & code intelligence – natural‑language codebase querying, refactoring suggestions, real‑time code linting, and a Git command assistant. These modules are intended to run inside the persistent VM and integrate with the memory core.

    Creative & multimedia features – offline image generation (Stable Diffusion), local audio synthesis, and video analysis. These features are planned to run as artifact overlays; models are loaded inside dedicated containers but results are managed by the VM.

    System & user experience – adaptive UI layouts, accessibility options, and guided onboarding. Many of these are research or planned features.

The roadmap clearly distinguishes between completed, planned and research status, making it a useful checklist of future work.
2 – Somnus Artifact System

This paper proposes a dual‑layer execution model for unlimited AI‑generated content【388067229034279†L125-L156】. Key points include:

    Separation of Concerns – AI agents run in persistent VMs, where their state, learned tools and orchestration logic live【388067229034279†L116-L124】. Computational tasks run in disposable container overlays attached to these VMs【388067229034279†L139-L151】. This prevents tool bloat in the VM while allowing unlimited capabilities.

    Resource Efficiency – the system avoids installing heavy libraries in the VM; instead it spins up specialized overlays (ML/AI training, video processing, research automation) on demand【388067229034279†L139-L151】. Overlays are created and destroyed per task, leaving the VM clean.

    AI‑Container Interface – a bridge class orchestrates overlays from the VM. It analyzes artifact requirements, creates specialized containers, monitors execution and cleans up afterwards.

    On‑Demand Capability Management – a settings/activation framework allows capabilities to be enabled when requested. Capabilities define required tools, resource allocations and network policies.

    Dual‑Trigger Execution – both AI agents and users can initiate overlay execution. The UI exposes triggers (buttons) to activate capabilities, while the AI can start overlays programmatically.

The Artifact System is the only dockerised component. All other subsystems (memory, model loading, multi‑agent orchestration, research) run natively inside the VM. Security is achieved by keeping the AI’s intelligence layer separate from the disposable overlays【388067229034279†L116-L124】.
3 – Somnus VM Architecture

This module defines the configuration and lifecycle for the persistent VMs. The introductory comments state the philosophy: each AI agent runs in a VM that never resets and can progressively accumulate tools, ensuring resource efficiency and user control【659167686334206†L4-L13】. Hardware specifications (vCPUs, memory, storage, network ports) and personality settings (agent name, specialization, creativity, preferred IDE/shell/browser) are defined【659167686334206†L68-L103】. VM instances include backup schedules, auto‑suspend timers, tracking of installed tools and learned capabilities, and metrics such as uptime and efficiency rating【659167686334206†L104-L145】. A VM pool configuration governs the maximum number of concurrent VMs, auto‑scaling and resource limits for the host【659167686334206†L148-L167】.
4 – Somnus Unified Performance Cache (system_cache.py)

The cache module provides a runtime caching layer that complements the memory core. It supports various namespaces – global, session, user, VM, artifact, model, research and system【837333225491083†L41-L49】 – enabling fine‑grained control over cached items. Entries store metadata (priority, TTL, access times) and have LRU‑based eviction scoring【837333225491083†L41-L56】. The cache engine records metrics (hits, misses, evictions) and persists data to disk using a SQLite metadata database and a separate shelf for values. Session‑aware namespacing and automatic cleanup allow context to be swapped or restored across subsystems and sessions【837333225491083†L8-L14】.
5 – Agent Collaboration Core

The collaboration core defines how multiple AI agents coordinate tasks. Important elements are:

    Message Types and Agent Capabilities – the module defines message types such as task proposals, acceptances, delegations, work progress, results and synthesis requests【276281714377717†L24-L33】. Agents advertise capabilities (text analysis, code generation, research synthesis, creative writing, logical reasoning, data processing, technical documentation, problem solving)【276281714377717†L48-L57】 and maintain profiles with collaboration preferences and independence thresholds【276281714377717†L60-L72】.

    Agent Profiles and Decision Making – each AgentProfile includes specialization scores, collaboration preference, independence threshold and communication style【276281714377717†L60-L72】. When processing user input, an agent analyses whether to handle the task alone or collaborate. The decision uses heuristics (keywords, complexity, length, task type) and the agent’s preference.

    Message Bus and Communication Protocol – the companion protocol module defines priority levels, protocol versions and message statuses (pending, delivered, processing, completed, failed, timeout)【528014049555495†L26-L47】. The AgentCommunicationMessage carries routing data, content and metadata, and provides integrity verification via checksums【528014049555495†L61-L115】. The TaskDelegationProtocol offers methods to create task proposals, acceptances, updates and completion messages with explicit deadlines, estimated effort and success criteria【528014049555495†L137-L169】.

    Task Delegation and Synthesis – the collaboration core can decompose complex queries into subtasks, delegate them to other agents based on capabilities, and aggregate results into a final response. It keeps track of active tasks, pending responses and shared workspaces, and integrates with the memory core so that results are stored permanently.

This core operates inside the AI VM; it uses the memory manager, model loader and session manager to maintain context and to load models for each agent.
System Integration & Subsystems

The system can be viewed as layered modules:

    Hardware & Hypervisor – the host machine runs a hypervisor (e.g. libvirt/QEMU) on which persistent AI VMs are created. VM pool settings control resource allocation, auto‑scaling and backup schedules【659167686334206†L148-L167】.

    Persistent AI VM Layer – each VM contains the following subsystems:

        Memory Core & Cache – long‑term semantic memory stores embeddings of all ingested data. The unified cache provides hot‑storage with namespacing for quick retrieval【837333225491083†L41-L56】.

        Model Loader – dynamically loads local models (GGUF) or proxies API models. It supports quantization, hardware acceleration and hot‑swapping.

        Session/VM Manager – creates and tracks VM instances, enforces resource policies and maintains personality settings【659167686334206†L4-L13】.

        File Upload & Processing – accepts files of any type, uses accelerated file processing to embed and index content, and integrates with git_integration_module for cloning and deep indexing of repositories【918727156888128†L6-L19】【918727156888128†L80-L105】.

        Web Research System – performs privacy‑first web searches and browser automation; results are fed into the memory core.

        Security Modules – red‑team toolkit, blue‑team playbook and purple‑team orchestrator provide offensive and defensive security, network isolation, container escape detection and vulnerability scanning.

        Plugin System – supports dynamic discovery, sandboxing and hot‑reloading of plugins, enabling extensibility.

        Agent Collaboration Core – manages swarms of AI agents, task decomposition, inter‑agent messaging and result synthesis【276281714377717†L24-L33】.

    Artifact System Overlay (Docker) – when users or the AI need to run heavy computations or specialized tools (e.g., training large models, video processing, browser automation), the VM orchestrates the creation of disposable container overlays. These overlays have full resource access and can install any package; after execution the overlay is destroyed, keeping the VM clean【388067229034279†L139-L151】. The VM communicates with overlays via a bridge and stores outputs as persistent artifacts.

    User Interface – a local chat UI provides access to all functionality. Capability triggers allow users to activate overlays, run research tasks, manage projects and collaborate with agents. Progressive enhancement means the base interface stays lightweight; overlays slide in when needed.

Text‑Based Architecture Diagram

[Host Machine / Hypervisor]
    └── VM Pool Manager (creates up to N persistent AI VMs)
           └── [Persistent AI VM]
                 ├── Memory Core & Unified Cache (global, session, user, VM, model, research namespaces)【837333225491083†L41-L49】
                 ├── Model Loader & Quantization
                 ├── Session/VM Settings Manager【659167686334206†L4-L13】
                 ├── File Upload & Accelerated Processing【918727156888128†L6-L19】
                 ├── Git/Code Integration & Repository Indexer【918727156888128†L80-L105】
                 ├── Web Research & Browser Automation
                 ├── Security Layer (Red/Blue/Purple Team modules)
                 ├── Plugin System (discovery, hot‑reloading)
                 ├── Agent Collaboration Core & Communication Protocol【276281714377717†L24-L33】
                 ├── Application Logic / UI Server
                 └── Artifact System Orchestrator
                        └── Disposable Container Overlays (Docker)
                              ├── ML/AI training overlay
                              ├── Video processing overlay
                              ├── Research automation overlay
                              └── … (specialized per task)【388067229034279†L139-L151】

Critical Subsystems & Feature Checklist
Subsystem / Feature	Summary	Status
Persistent VM Manager	Creates and configures VMs, including hardware specs, personalities, backup schedules and resource policies【659167686334206†L4-L13】.	Implemented
Memory Core & Unified Cache	Long‑term semantic memory with encryption; unified cache for hot data with namespaces and LRU eviction【837333225491083†L41-L56】.	Implemented
Model Loader	Loads local models (GGUF) and proxies remote APIs; supports quantization, hot‑swapping and hardware acceleration.	Implemented
Accelerated File Processing	Optimized ingestion and embedding of all file types; integrates with file upload system【918727156888128†L6-L19】.	Completed
Git Integration Module	Clones and deeply indexes repositories; classifies files and feeds them into memory【918727156888128†L80-L105】.	Completed
Artifact System	Dual‑layer architecture with container overlays; unlimited tool installation; orchestrated by VM【388067229034279†L116-L124】.	Implemented
Agent Collaboration Core	Profiles agents, decides when to collaborate and uses structured messages to delegate tasks and synthesize results【276281714377717†L24-L33】.	Implemented
Agent Communication Protocol	Defines message priorities, statuses, checksums and a task delegation protocol【528014049555495†L26-L47】.	Implemented
Web Research & Browser Agent	Sandbox browser for AI learning (planned); integrates with memory core for local web reading.	Planned
Dynamic Memory Management & Tiering	Moves cold data to slower storage; compresses or evicts rarely used items.	Planned
Semantic Search & Knowledge Graph	Builds vector index of local files and supports natural‑language queries.	Planned
Clipboard Monitor & Contextual Actions	Tracks clipboard history and suggests context‑aware actions.	Planned
Local Automation Engine	No‑/low‑code rule engine for automating file and system events.	Research
Digest Generator	Summarizes recent activities and content into daily/weekly briefs.	Planned
Developer Tools	Codebase query/refactoring assistant, linting, Git command assistant with visual graphs.	Planned/Research
Creative Tools	Offline image and audio generation; video analysis.	Planned/Research
UI & Accessibility	Adaptive layouts, themes, keyboard navigation and contextual help.	Planned/Research
Security	Red‑team toolkit, blue‑team playbook, purple‑team orchestrator for continuous testing and defence.	Implemented

Observations & Gaps

    Many advanced features (semantic search, dynamic memory tiering, clipboard monitoring, automation, digest generation) are planned or research‑stage; documentation exists but code may be incomplete. These areas represent technical debt or future work.

    The roadmap outlines numerous creative and UI enhancements that are not yet implemented; further design and development is needed.

    The artifact system specification is detailed, but implementation specifics (container orchestration engine, security policies) may need further elaboration.

    There is minimal documentation on the exact API endpoints or UI flows for multi‑agent collaboration; additional design docs could improve clarity.

Getting Started Guide
For a New AI Agent (LLM)

    Prepare the Host – install the hypervisor (libvirt/QEMU) and ensure your machine has enough CPU, memory and storage. Install Python 3.11 and the required dependencies using pip install -r requirements.txt.

    Boot the System – run main.py or the web server entry point. The VM Pool Manager will initialize persistent VM instances based on VMPoolSettings. Default VMs use 4 vCPUs, 8 GB RAM and 100 GB disk【659167686334206†L68-L72】.

    Create a VM for the Agent – use the VMSettingsManager (via API or CLI) to create a new VM instance. Specify a name, user ID, hardware specs and personality (e.g., research‑oriented with preferred IDE = VS Code and research methodology = systematic)【659167686334206†L68-L103】.

    Load an LLM – within the VM, call the ModelLoader to load a local GGUF model or configure an API proxy. The model will be cached for reuse. For multi‑agent setups, load a model per agent and assign each agent a unique profile specifying its capabilities【276281714377717†L60-L72】.

    Ingest Data – upload files via the file upload API or UI. The accelerated file processor will embed and index them【918727156888128†L6-L19】. Use the git integration to clone repositories for code comprehension【918727156888128†L80-L105】. All embeddings and metadata are stored in the memory core.

    Enable Tools & Plugins – install any tools or plugins you need inside the VM (e.g., analysis libraries, notebooks). The VM persists installed tools, unlike disposable overlays.

    Run Tasks and Collaborate – interact through the chat UI. Use the multi‑agent core to spawn additional agents with different capabilities. The system will decide whether to handle tasks solo or delegate. You can manually instruct agents to collaborate or specify swarms.

    Use the Artifact System – when tasks require heavy computation or additional tools, trigger the Artifact System from the UI. The VM will create a specialized container overlay with the requested tools and run the code. Outputs are stored as artifacts accessible from the persistent VM【388067229034279†L116-L124】.

    Monitor and Manage – view VM metrics (uptime, resource usage, installed tools, learned capabilities), manage backups and snapshots via the VM manager. Use the security modules to run periodic red/blue team tests.

For Developers Extending the System

    Understand the architecture – familiarize yourself with the persistent VM philosophy, dual‑layer artifact system, unified cache and multi‑agent collaboration. Review the core modules under /core and /backend.

    Follow modular design – new features should integrate with the VM via well‑defined interfaces. For example, new research capabilities should use the memory core for storage and the agent collaboration core for synthesis.

    Implement new capabilities – to add a tool that requires external libraries, create a capability definition and configure an overlay image. The artifact system will handle lifecycle and resource allocation.

    Add agents – extend AgentProfile to include new capability types, and implement corresponding processing logic. Use the communication protocol to define new message types if needed.

    Respect security – ensure that plugins or overlays do not compromise the VM. Use the red/blue team toolkit to test for vulnerabilities and follow the separation of concerns model.

For LLM Developers (Adding New Models)

    Model Packaging – convert your model to a local format (e.g., GGUF for quantized models) and place it in the models directory.

    Register with ModelLoader – add your model entry to the models configuration file and specify any quantization or hardware preferences.

    Load and Test – use the ModelLoader API to load your model into a VM. Verify that it responds correctly and that embeddings can be generated. Update agent profiles to use the new model when appropriate.

Conclusion

Somnus Sovereign Chat introduces a radical departure from transient, cloud‑based AI services. Its architecture revolves around persistent, user‑owned VMs that host all intelligence and memory, coupled with disposable container overlays for unlimited computational tasks【388067229034279†L116-L124】. A unified cache and memory system preserves context across sessions, and an advanced collaboration core orchestrates swarms of agents. While many planned features remain to be implemented, the foundation already offers a powerful, sovereign AI environment that can be extended and refined into a truly autonomous OS.
