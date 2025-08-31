# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Somnus Sovereign Systems** is a revolutionary AI Operating System that provides complete digital sovereignty through persistent virtual machines and unlimited execution capabilities. This is a Python-based FastAPI application that implements a dual-layer execution model separating AI intelligence (persistent VMs) from computation (disposable containers).

## Key Development Commands

### Setup and Installation

```bash
# Initial setup
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt

# Build base VM image (for production)
docker build -f configs/dockerfile_base.txt -t somnus-base:latest .
```

### Running the Application

```bash
# Start the main application server
python main.py --config configs/base_config.yaml --port 8000

# Development mode with debug logging
python main.py --debug --port 8000

# Alternative backend server (simplified version)
python backend/app_server.py
```

### Testing and Development

```bash
# No formal test suite is present - the system appears to be in active development
# Manual testing is done by running the server and testing endpoints

# Check system configuration
python -c "import yaml; print(yaml.safe_load(open('configs/base_config.yaml')))"

# Validate memory configuration
python -c "from core.memory_core import MemoryConfiguration; print('Config valid')"
```

## High-Level Architecture

### Core Design Philosophy: Dual-Layer Execution

The system implements a groundbreaking separation of intelligence and computation:

#### 🧠 Intelligence Layer (Persistent VMs)
- **Persistent AI computers** that never reset and maintain state permanently
- **Cross-session memory** with semantic knowledge graphs
- **Self-improving environments** through progressive tool installation
- **Personality persistence** with customizable AI behaviors

#### ⚡ Computation Layer (Disposable Containers)
- **Specialized containers** for heavy computational tasks (ML training, video processing)
- **Unlimited tool installation** without VM bloat
- **GPU-accelerated workflows** with automatic cleanup
- **Complete resource access** with security through architectural separation

### Key Components

#### 1. Memory System (`core/memory_core.py`)
- **Semantic vector storage** using embeddings for intelligent retrieval
- **Multi-modal memory support** (text, code, files, images, conversations)
- **User-scoped encryption** with granular privacy controls
- **Importance-based retention** with automatic forgetting mechanisms
- **Cross-session context reconstruction** for seamless conversation continuity

#### 2. Session Management (`backend/memory_integration.py`)
- **Memory-enhanced sessions** that bridge persistent memory with active sessions
- **Cross-session state management** for seamless VM transitions
- **Automatic fact extraction** from conversations with importance scoring
- **Enhanced system prompts** with personalized AI behavior

#### 3. Cache System (`backend/system_cache.py`)
- **Multi-namespace caching** (global, session, user, VM, artifact, model, research)
- **Intelligent "app swapping"** - seamless transitions between AI capabilities
- **LRU eviction with priority scoring** and dependency tracking
- **Background persistence** with compression and integrity checking

#### 4. File Processing (`core/file_upload_system.py` + `core/accelerated_file_processing.py`)
- **Dual-layer architecture**: Core upload system + intelligent acceleration
- **Multi-format support** with security validation and content extraction
- **Priority-based processing queue** with adaptive thread pool management
- **Real-time progress tracking** with user feedback callbacks

#### 5. Virtual Machine Management
- **VM Supervisor** (`backend/virtual_machine/vm_supervisor.py`) - LibVirt integration with QEMU/KVM
- **Somnus Agent** (`backend/virtual_machine/somnus_agent.py`) - In-VM health management
- **AI Development Environment** - Personal coding environments that evolve over time
- **AI Browser Research System** - Personal browser with visual interaction capabilities

#### 6. Multi-Agent Collaboration (`core/agent_collaboration_core.py`)
- **Direct AI-to-AI communication** with structured message passing
- **Task delegation and response synthesis** with confidence scoring
- **Specialized agent profiles** (TextAnalyst, CodeGenerator, ResearchSynthesizer)
- **Consensus building** through structured debate mechanisms

#### 7. Research Engine (`core/web_search_research.py`)
- **Triple-layer research architecture**: Browser VMs, Chat research, Deep research
- **OSINT capabilities** with contradiction detection and resolution
- **Multi-agent research teams** with specialized expertise coordination
- **Real-time streaming** with user intervention capabilities

### Configuration Files

#### Primary Configuration (`configs/base_config.yaml`)
- **VM management settings**: Resource limits, timeouts, storage paths
- **Security enforcement**: Content filtering, audit trails, rate limits
- **Context and memory settings**: Token limits, persistence policies
- **Logging and monitoring**: Comprehensive system observability

#### Memory Configuration (`configs/memory_config.yaml`)
- **Retention policies** by importance level (Critical: never expire, Temporary: 1 day)
- **Embedding models** for semantic search and specialized domains
- **Privacy settings** with encryption and user data isolation
- **Performance tuning** for vector databases and caching strategies

### Directory Structure

```
/
├── main.py                     # Primary FastAPI application entry point
├── backend/
│   ├── app_server.py          # Alternative backend server
│   ├── memory_integration.py   # Memory-session bridge
│   └── system_cache.py        # High-performance caching layer
├── core/
│   ├── memory_core.py         # Persistent memory with semantic indexing
│   ├── session_manager.py     # Session lifecycle management
│   ├── model_loader.py        # Dynamic model loading and quantization
│   ├── file_upload_system.py  # Core file processing foundation
│   ├── accelerated_file_processing.py # Enterprise file acceleration
│   ├── agent_collaboration_core.py # Multi-agent coordination
│   └── web_search_research.py # Research and OSINT capabilities
├── configs/
│   ├── base_config.yaml       # System configuration
│   ├── memory_config.yaml     # Memory system settings
│   ├── models_config.yaml     # AI model configurations
│   └── dockerfile_base.txt    # VM base image definition
├── projects/
│   └── project_api.py         # Project VM management
└── docs/
    ├── somnus_vm_architecture.md # Detailed VM architecture
    └── Various design documents
```

## Development Patterns

### FastAPI Application Structure
- **Modular design** with clear separation between core systems and API layers
- **Async/await throughout** for non-blocking operations
- **Comprehensive error handling** with domain-specific exceptions
- **WebSocket support** for real-time collaboration and streaming

### Memory and Caching Strategy
- **Write-through caching** with persistent memory backing
- **Semantic indexing** using sentence transformers for intelligent retrieval
- **Multi-namespace organization** preventing data conflicts
- **Automatic cleanup** with configurable retention policies

### Security Model
- **User-scoped encryption** with Fernet symmetric encryption
- **Granular privacy controls** with data isolation guarantees
- **Content filtering** and prompt injection detection
- **Capability enforcement** through architectural boundaries

### VM and Container Integration
- **Infrastructure as Code** approach using LibVirt and Docker
- **Snapshot-based development** for safe experimentation
- **Resource profiling** with dynamic scaling capabilities
- **Health monitoring** with semantic error analysis

## Important Implementation Notes

### Memory System Integration
- All user interactions automatically stored with importance assessment
- Semantic search enables intelligent context reconstruction across sessions
- Cross-session continuity maintains conversation context during VM swapping
- Privacy-first design with local-only processing and user-controlled encryption

### "App Swapping" Architecture
- System enables instant transitions between different AI modes (chat, research, coding)
- VM states are cached with high priority for near-zero latency switching
- Model states are preserved to avoid expensive reloading
- Background persistence ensures no data loss during transitions

### Multi-Agent Coordination
- Direct AI-to-AI communication protocols for sophisticated collaboration
- Task delegation based on agent capabilities and content analysis
- Real-time synthesis with contradiction detection and resolution
- Structured debate mechanisms for consensus building

### Unlimited Execution Philosophy
- No artificial timeouts or capability restrictions
- Complete system access within secure architectural boundaries
- GPU acceleration for ML training and media processing
- Persistent environments that accumulate capabilities over time

## Security Considerations

- **Local-first architecture** ensures data never leaves user's machine
- **Encrypted memory storage** with user-scoped key derivation
- **VM isolation** provides security through architectural separation
- **Audit logging** of all system interactions and capability usage
- **Granular permission controls** for network access and resource allocation

This system represents a paradigm shift from traditional AI-as-a-Service toward true digital sovereignty, providing unlimited capabilities while maintaining complete user control and privacy.
