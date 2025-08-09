# Somnus Sovereign Systems: Gemini Context Document

This document provides a high-level overview and context for the Somnus Sovereign Systems project, synthesizing information from various project documentation and the ongoing development efforts.

## Project Overview and Core Philosophy

*   **The Ultimate SaaS-Killer**: Somnus Sovereign Systems aims to be a paradigm-shifting, locally-first AI operating environment that eliminates subscription dependencies, ensures absolute privacy, and provides unlimited computational capabilities. It's designed to "do everything the Big 3 does, do it better, and then do more things than they ever thought of."
*   **Complete Digital Sovereignty**: Emphasizes zero subscription fees, absolute privacy (100% local processing), no vendor lock-in, and unlimited scalability.
*   **Revolutionary Capabilities**: Focuses on persistent AI intelligence (AIs that remember and evolve), unlimited execution (no timeouts), true multi-agent collaboration, and project-specific VMs.
*   **Dual-Layer Execution Model**: A groundbreaking architectural philosophy separating the **Persistent AI Intelligence Layer (Virtual Machines)** from the **Disposable Computation Layer (Container Overlays)**.

## Key Subsystems and Architecture

### 1. Virtual Machine (VM) System

The VM system is the bedrock of the persistent AI computing environment, where AI agents operate from never-reset virtual machines.

*   **VM Supervisor**: OS-level manager for fleets of persistent AI computers, leveraging Libvirt for direct control over KVM/QEMU. Handles dynamic resource scaling, snapshot management, intelligent monitoring, and VM lifecycle.
*   **In-VM Somnus Agent**: A lightweight Flask agent inside each AI VM for host communication, health monitoring, intelligent error analysis, and soft reboots.
*   **AI Action Orchestrator**: The primary high-level interface for AI agents to interact with their sovereign environment, translating AI intentions into API calls for artifact management, memory integration, and session logging.
*   **Advanced AI Shell**: A unified command execution interface supporting multiple contexts (VM Native, Container Overlay, Multi-Agent, Hybrid), with intelligent command routing and container orchestration.
*   **AI Personal Development Environment**: A self-improving development setup that evolves with AI experience, building capabilities like tool installation, personal libraries, and automation scripts.
*   **AI Browser Research System**: Provides persistent, sophisticated browser-based research capabilities with visual interaction, content extraction, and workflow automation.
*   **Sovereign AI Orchestrator**: The central conductor for provisioning and integrating all elements of a sovereign AI development environment.

### 2. Memory System

A revolutionary dual-memory system enabling persistent AI intelligence, seamless VM hot-swapping, and ultra-low compute requirements.

*   **Memory Core**: Provides enterprise-grade persistent memory with semantic indexing, encryption (user-scoped), and cross-session continuity. Supports multi-modal data and importance-based retention.
*   **Memory Integration**: Seamlessly bridges the persistent memory core with active sessions, enabling intelligent context reconstruction and "app swapping." Manages session memory context and enhances session management.
*   **System Cache**: A high-performance runtime caching layer crucial for "app swapping" and achieving ultra-low compute. Features multi-namespace caching, intelligent priority scoring, background persistence, and dependency tracking.

### 3. Artifact System ("Disposable Super Computers")

Represents a revolutionary approach to AI sovereignty through unlimited execution environments.

*   **Core Philosophy: NO LIMITS**: Zero operational cost, unlimited execution time, complete hardware access, unrestricted internet access, persistent state, and real-time collaboration.
*   **Container-per-Artifact Architecture**: Each artifact receives its own disposable supercomputer, ensuring complete isolation, full hardware access, and hot-swappable environments.
*   **Revolutionary Features**: Includes unlimited execution model, real-time collaborative editing (multi-user and multi-AI), and progressive feature revelation.

### 4. Project Subsystem

Transforms projects from static storage into living, evolving digital organisms, each with its own persistent VM.

*   **Project API**: The integration backbone exposing core functionality via FastAPI endpoints with real-time WebSockets.
*   **Project VM Manager**: Dedicated VM lifecycle manager for project-level compute isolation, resource management, and dynamic environment instantiation (one VM per project).
*   **Project Memory**: Project-level persistent memory engine tracking meaningful project events, structuring them as context-aware memories, and surfacing analytic insights.
*   **Project Knowledge**: Autonomous project knowledge base extracting, organizing, relating, and synthesizing project knowledge automatically, forming a dynamic knowledge graph with semantic search.
*   **Project Core**: Manages project creation, lifecycle, intelligence, persistence, and orchestration, where a "project" becomes a living, evolving digital organism (Project = VM + Model + Autonomous Intelligence).
*   **Project Automation**: An AI-driven, fully modular automation engine for all projects, enabling "self-updating," self-maintaining, and "self-optimizing" operations with rich automation abstractions and AI-driven rule creation.

## Gemini's Role

As the final development engineer, my primary goal is to fully implement every missing or unimplemented definition, function, or code, and improve the project to a production-ready and complex state. I adhere strictly to existing project conventions, verify library/framework availability, mimic existing style and structure, and ensure idiomatic changes. I prioritize user control and safety, explaining critical commands before execution. My work focuses on finishing the ORAMA computer agent, ensuring all code is completely final, production-ready, and field-deployable, avoiding test or demo code.

## Development Process (Gemini's Protocol)

My development process involves a recursive 3-phase analysis/improvement protocol applied at every architectural level:

1.  **Macro Level (Superset)**: Identify every desired feature, what's missing or locked down in existing SaaS/AI apps, and plan next-level features.
2.  **Subsystem Level (Refinement)**: For each backend system (core, memory, tools, artifact, plugin), compare against "Big 3" features, remove constraints, fix broken logic, and add innovative functionalities.
3.  **Micro Level (Transcendence)**: For each module/file and function, analyze best practices, identify shortcomings, and implement solutions that transcend current industry standards, focusing on sovereignty and creativity.

This iterative process ensures that Somnus Sovereign Systems not only matches but surpasses existing solutions, delivering a truly autonomous and sovereign AI operating environment.

## Remaining Development Tasks: Virtual Machine Subfolder

This section outlines remaining development tasks for modules within the `backend/virtual_machine/` subfolder to achieve full functionality and production readiness.

### `ai_browser_research_system.py` - Remaining Work

The `ai_browser_research_system.py` file currently serves as a high-level outline, with many core functionalities represented by placeholder methods or incomplete logic. Significant implementation is required for the following:

#### Core Research Workflow Methods:

*   **`_load_workflow(self, workflow_file)`**: Implement the logic to read and parse research workflow definitions from JSON files. This method is crucial for dynamically configuring research processes.
*   **`_start_research_browser(self)`**: Develop the full implementation for launching and managing the AI's personal browser session (e.g., Firefox/Chrome). This involves establishing a programmatic interface with the browser (e.g., using Playwright or Selenium).
*   **`_generate_research_plan(self, query)`**: Implement the AI reasoning logic to dynamically generate a detailed, step-by-step research plan based on a given query. This should involve breaking down complex queries into actionable research steps.
*   **`_analyze_article_content(self, article_data)`**: Develop the intelligent analysis of extracted article content. This method should process raw text and potentially visual data to generate structured notes, summaries, and identify key insights.
*   **`_save_article_to_db(self, url, article_data, notes)`**: Implement the functionality to persistently save analyzed article data, AI-generated notes, and associated metadata to a personal research database.
*   **`_cross_reference_facts(self, browser, claims_to_verify)`**: Develop the logic for cross-referencing information across multiple web sources to verify claims and assess credibility. This will involve navigating to different URLs and comparing content.
*   **`_download_research_documents(self, browser, document_urls)`**: Implement the automated downloading of research documents (e.g., PDFs, DOCX) from specified URLs and their subsequent processing (e.g., saving, indexing).
*   **`_get_knowledge_base_size(self)`**: Implement this missing method to return the current size or count of entries in the AI's personal knowledge base.

#### Automation and Extension Integration:

*   **`install_research_extension(self, extension_name)`**: Fully implement the actual installation logic for browser extensions within the VM. This involves executing the necessary shell commands or API calls to install the extension.
*   **`create_research_automation(self, workflow_name)`**: The generated automation scripts currently contain placeholder logic. The `execute_workflow`, `quick_research`, and `code_snippet_to_file` functions within these scripts need to be fully fleshed out to perform their intended automated tasks.

#### General Improvements:

*   **Error Handling**: Enhance error handling within all methods to be robust and provide informative logging for debugging and operational insights.
*   **Asynchronous Operations**: Ensure all I/O-bound operations (network requests, file system access) are truly asynchronous to prevent blocking and maximize concurrency.
*   **Modularity**: Review and refactor any tightly coupled logic to improve modularity and testability.
*   **Testing**: Develop comprehensive unit and integration tests for all implemented functionalities to ensure reliability and correctness.