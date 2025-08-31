# VM Subsystem TODO List

This document lists the pending tasks, placeholders, and areas for improvement identified within the `virtual_machine` subsystem.

### General / Architectural

*   **Windows/macOS Support:** Many scripts assume a Linux (`apt-get`) environment, particularly for tool installation (`ai_personal_dev_template.py`). These need to be abstracted or expanded to support other host/guest operating systems. This is a windows 11 native application, not linux. All docker operations are through mainly docker desktop/wsl2 backend on windows 11. The `CAPABILITY_PACKS` in `ai_orchestrator.py` also contain Linux-specific commands (`apt-get`, `snap`).
*   **WSGI Production Server:** The `SomnusAgent`'s Flask app is run with the development server. It should be deployed with a production-grade WSGI server like Gunicorn or uWSGI for stability. (Confirmed in `somnus_agent.py`)

### `vm_supervisor.py`

*   **Soft Reboot Confirmation:** The `soft_reboot` method currently "fires and forgets". It needs a confirmation loop or callback mechanism to verify that the AI processes have successfully restarted inside the VM before clearing the `soft_reboot_pending` flag. (Confirmed in `vm_supervisor.py`)
*   **Asynchronous Operations:** The current implementation uses a mix of `async` and standard `threading`. The background monitor runs in a standard thread and blocks on `time.sleep`. The core `libvirt` calls are also synchronous. A full `asyncio` implementation could offer better performance, especially when managing a large number of VMs. (Confirmed in `vm_supervisor.py`)
*   **Robust XML Generation:** The libvirt XML is generated via f-strings. For more complex or dynamic configurations, using a dedicated library like `lxml` would be more robust and maintainable. (Confirmed in `vm_supervisor.py`)

### `ai_orchestrator.py` (SovereignAIOrchestrator)

*   **Implement Command Execution:** The logic for executing commands during capability pack installation is commented out. This is a critical missing piece. The `await self.vm_supervisor.execute_command_in_vm(...)` call needs to be implemented in the supervisor and called here.
    *   **Note:** `advanced_ai_shell.py`'s `execute_in_vm` method uses `artifact_system.run_command` for VM-native execution. The `vm_supervisor` will call an endpoint on the `somnus_agent` to execute commands inside the VM.
*   **Implement Rollback Logic:** The `try...except` block for capability installation has a commented-out call to `rollback_to_last_snapshot`. This crucial safety feature needs to be implemented to ensure atomic and safe environment setup.
*   **Implement Security Validation:** The security check is explicitly a placeholder. It needs to be integrated with the `SecurityEnforcer` to validate provisioning requests.
*   **Dynamic Chat Session ID:** The `chat_session_id` is hardcoded with a placeholder UUID. This needs to be dynamically linked to a real chat session from the application's frontend/API layer.

### `somnus_agent.py`

*   **Implement Command Execution Endpoint:** Add a new API endpoint (e.g., `/execute_command`) to the `SomnusAgent` that accepts a command string and executes it within the VM, returning stdout, stderr, and exit code. This will be used by the `vm_supervisor` for capability pack installation.

### `ai_browser_research_system.py`

*   **Upgrade NLP Models:** This is the most significant area for improvement. Many analysis functions (`_extract_key_topics`, `_analyze_sentiment`, `_detect_bias_indicators`, `_extract_claims_from_content`, `_calculate_relevance_score`, `_calculate_readability`, `_calculate_fact_density`, `_assess_source_quality`, `_analyze_citations`, `_analyze_content_structure`, `_calculate_information_density`, `_detect_expertise_indicators`, `_calculate_overall_quality`, `_generate_research_insights`, `_generate_primary_findings`, `_calculate_confidence_levels`, `_analyze_source_diversity`, `_perform_temporal_analysis`, `_generate_weighted_conclusions`, `_identify_knowledge_gaps`, `_assess_evidence_strength`, and `_generate_synthesis_summary`) explicitly state they are simplified, keyword-based implementations. They need to be replaced with calls to more sophisticated NLP models (e.g., spaCy, NLTK, or a dedicated LLM) to achieve their full potential.
*   **Remove Duplicated Code:** The file contains duplicated method definitions for `conduct_visual_research`, `_start_research_browser`, and `_generate_research_plan`. This appears to be a copy-paste error and must be refactored.
*   **Implement Incomplete Methods:** The `_compare_with_baseline` and `schedule_regular_research` methods within the auto-generated automation script are empty (`pass`). Their logic needs to be implemented.
*   **Implement Extension Installation:** The `install_research_extension` method contains placeholder comments and does not have the full logic required to properly install a browser extension from the Chrome Web Store.
*   **Refine Generated Scripts:** The `main` block in the auto-generated automation scripts is not fully runnable, as it has commented-out lines for component initialization. It should be made more robust or self-contained.

### `ai_action_orchestrator.py`

*   **Implement Streaming Execution:** The `execute_in_container` client method notes that the API endpoint is synchronous. A more advanced implementation should use WebSockets or another streaming protocol to provide real-time output from container execution to the AI.
    *   **Note:** `advanced_ai_shell.py`'s `execute_in_container` method already supports `stream_output`, which can be leveraged here. The client-side implementation in `ai_action_orchestrator.py` needs to be updated to consume this streaming capability.
*   **Implement Host-Side Endpoints:** The client assumes several API endpoints exist on the host (`/memory/store`, `/container/.../execute-sync`, etc.). These need to be implemented in the host's FastAPI server.
*   **Dynamic AI User ID:** The `ai_user_id` is hardcoded to `"ai_instance_vm"`. This could be dynamically set or passed in during initialization for better traceability.

### `ai_personal_dev_template.py`

*   **Implement Cross-Platform Tool Installation:** The `_install_tools` method is hardcoded for Debian/Ubuntu (`apt-get`). This needs to be expanded with logic for other package managers (e.g., `yum`, `pacman`, `brew`, `winget`) to be truly cross-platform.
*   **Dynamic Utility Generation:** The methods that generate personal library code (`_generate_core_utilities`, `_generate_automation_utilities`, `_generate_research_utilities`, `_generate_development_utilities`) provide static, pre-written strings. A more advanced AI could generate this code dynamically based on its learned needs.
*   **Refine Artifact Creation:** The call to `self.orchestrator.create_and_setup_artifact` is a good integration point, but the capabilities and environment passed (`unlimited_power`, `code_execution`) are hardcoded. This could be made more dynamic based on the library being created.