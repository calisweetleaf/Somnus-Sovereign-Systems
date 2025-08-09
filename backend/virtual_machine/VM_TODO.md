# VM Subfolder TODO List

This document outlines remaining development tasks for modules within the `backend/virtual_machine/` subfolder to achieve full functionality and production readiness.

## `ai_browser_research_system.py` - Remaining Work

The `ai_browser_research_system.py` file currently serves as a high-level outline, with many core functionalities represented by placeholder methods or incomplete logic. Significant implementation is required for the following:

### Core Research Workflow Methods:

*   **`_load_workflow(self, workflow_file)`**: Implement the logic to read and parse research workflow definitions from JSON files. This method is crucial for dynamically configuring research processes.
*   **`_start_research_browser(self)`**: Develop the full implementation for launching and managing the AI's personal browser session (e.g., Firefox/Chrome). This involves establishing a programmatic interface with the browser (e.g., using Playwright or Selenium).
*   **`_generate_research_plan(self, query)`**: Implement the AI reasoning logic to dynamically generate a detailed, step-by-step research plan based on a given query. This should involve breaking down complex queries into actionable research steps.
*   **`_analyze_article_content(self, article_data)`**: Develop the intelligent analysis of extracted article content. This method should process raw text and potentially visual data to generate structured notes, summaries, and identify key insights.
*   **`_save_article_to_db(self, url, article_data, notes)`**: Implement the functionality to persistently save analyzed article data, AI-generated notes, and associated metadata to a personal research database.
*   **`_cross_reference_facts(self, browser, claims_to_verify)`**: Develop the logic for cross-referencing information across multiple web sources to verify claims and assess credibility. This will involve navigating to different URLs and comparing content.
*   **`_download_research_documents(self, browser, document_urls)`**: Implement the automated downloading of research documents (e.g., PDFs, DOCX) from specified URLs and their subsequent processing (e.g., saving, indexing).
*   **`_get_knowledge_base_size(self)`**: Implement this missing method to return the current size or count of entries in the AI's personal knowledge base.

### Automation and Extension Integration:

*   **`install_research_extension(self, extension_name)`**: Fully implement the actual installation logic for browser extensions within the VM. This involves executing the necessary shell commands or API calls to install the extension.
*   **`create_research_automation(self, workflow_name)`**: The generated automation scripts currently contain placeholder logic. The `execute_workflow`, `quick_research`, and `code_snippet_to_file` functions within these scripts need to be fully fleshed out to perform their intended automated tasks.

### General Improvements:

*   **Error Handling**: Enhance error handling within all methods to be robust and provide informative logging for debugging and operational insights.
*   **Asynchronous Operations**: Ensure all I/O-bound operations (network requests, file system access) are truly asynchronous to prevent blocking and maximize concurrency.
*   **Modularity**: Review and refactor any tightly coupled logic to improve modularity and testability.
*   **Testing**: Develop comprehensive unit and integration tests for all implemented functionalities to ensure reliability and correctness.

## `advanced_ai_shell.py` - Remaining Work

This module is largely implemented, but the following areas require further development for production readiness and enhanced sophistication:

1.  **`CollaborationManager._synthesize_responses`**: This method currently uses a "Simple synthesis" approach. For production, it explicitly notes the need to integrate an **LLM (Large Language Model)** for more sophisticated and intelligent synthesis of responses from multiple collaborating agents. This is a critical functional enhancement.

2.  **`AdvancedAIShell._classify_command`**: The current command classification relies on basic keyword matching. To improve intelligent routing and command understanding, this method could be enhanced with more advanced **Natural Language Processing (NLP)** techniques, such as intent recognition or semantic analysis.

3.  **Robustness of VM/Docker Connections**: While `try-except` blocks are present for SSH and Docker client initialization (`_connect_to_vm`, `_initialize_docker_client`), implementing more robust **retry mechanisms with exponential backoff** and defined maximum retry attempts would improve resilience in production environments where network or service transient issues might occur.

## `ai_orchestrator.py` - Remaining Work

This module provides the high-level orchestration logic, but it relies on a critical unimplemented method in `VMSupervisor` and has a placeholder for security validation.

1.  **`SovereignAIOrchestrator.provision_sovereign_environment` - Security Validation**: The security validation logic (commented as a placeholder) needs to be fully implemented. This would involve integrating with the `security_enforcer` to validate provisioning requests based on defined security policies.
2.  **Integration with `VMSupervisor.execute_command_in_vm`**: The core functionality of executing commands within the provisioned VM to install capability packs is currently commented out. This requires the implementation of an `execute_command_in_vm` method within the `VMSupervisor` class, and then uncommenting and utilizing it here. This is a **critical functional gap**.

## `ai_personal_dev_environment.py` - Remaining Work

This file has a significant number of unimplemented helper methods that are crucial for its functionality. The core idea of an evolving personal development environment is present, but the mechanisms for loading preferences, managing tools, and generating code utilities are largely placeholders.

1.  **`_load_ai_preferences(self)`**: Implement the logic to load the AI's evolving development preferences from a persistent source.
2.  **`_get_tools_for_project(self, project_type)`**: Implement the logic to dynamically identify and return suitable tools for a given project type, potentially by querying the VM's installed tools or a knowledge base.
3.  **`_setup_ide_for_project(self, project_type)`**: This method needs to be fleshed out to configure the AI's preferred IDE, including extensions, themes, and project-specific settings.
4.  **`_activate_project_environment(self, project_type)`**: Implement the logic for activating relevant virtual environments (e.g., Python virtual environments, Node.js environments) within the VM for the specified project type.
5.  **`_load_personal_templates(self, project_type)`**: This method requires implementation to load personal code templates or project structures that the AI has created or learned.
6.  **`_create_ai_shortcuts(self, tool_category, cmd)`**: Implement the logic to create system-level shortcuts or aliases within the AI's VM for frequently used tools or commands, enhancing efficiency.
7.  **`_generate_research_helpers()`**: The current placeholder needs to be replaced with actual logic to generate Python utility functions specifically for research tasks.
8.  **`_generate_automation_tools()`**: The current placeholder needs to be replaced with actual logic to generate Python utility functions specifically for automation tasks.
9.  **External Tool Dependencies**: The functionality of this module heavily relies on the presence and correct configuration of external tools (e.g., `firefox`, `jupyter`, `code`, `obsidian`, `npm`, `crontab`) within the VM. While the module assumes their availability, ensuring their proper installation and setup during the VM provisioning process (managed by `ai_orchestrator.py` and `vm_supervisor.py`) is crucial for this module's operational integrity.

## `somnus_agent.py` - Remaining Work

The `somnus_agent.py` module is robust and largely complete for its intended function. However, to achieve full production readiness and enhance its capabilities, the following areas require further development:

1.  **Soft Reboot Completion Signal**: Currently, the `soft_reboot` endpoint assumes that core AI processes will restart after termination and are polled by the VM Supervisor. A more robust implementation would involve the agent explicitly signaling back to the VM Supervisor (or a dedicated monitoring service) when the soft reboot process is truly complete and the monitored processes have successfully restarted. This would provide definitive feedback and improve system reliability.

2.  **Production Deployment Configuration**: The `if __name__ == "__main__":` block uses a simple Flask development server. For production deployment, this should be replaced with a robust WSGI server (e.g., Gunicorn, uWSGI) and integrated with a proper process management system (e.g., systemd) to ensure high availability and efficient resource utilization.

3.  **Enhanced Log File Management**: The agent relies on `log_files` configured in `agent_config.json` for intelligent statistics. For a production environment, the configuration and handling of these log files should be made more robust to account for various log formats, log rotation, and potential large file sizes, ensuring reliable log analysis.

4.  **Robustness in `_cosine_sim`**: While a small epsilon is added to the denominator, explicitly handling cases where the norm of an embedding might be zero (e.g., for all-zero vectors) would prevent potential division-by-zero errors and improve the mathematical robustness of the cosine similarity calculation.

## `vm_supervisor.py` - Remaining Work

This file has several **critical bugs related to asynchronous programming** where synchronous I/O operations are blocking the asyncio event loop. These need to be addressed immediately for the application to perform correctly in an asynchronous environment. Additionally, there's a data model inconsistency and a repeated future enhancement.

1.  **Asynchronous I/O Correction (Critical)**: All synchronous I/O operations within `async` functions must be offloaded to a thread pool executor (e.g., `asyncio.to_thread`) or replaced with their asynchronous counterparts. This applies to:
    *   `create_ai_computer`: `subprocess.run` calls.
    *   `_monitor_loop`: `time.sleep` (should be `await asyncio.sleep`), and `agent_client.get_runtime_stats()` (which internally uses `requests`, so `SomnusVMAgentClient._request` needs to be offloaded).
    *   `create_ai_computer`: `time.sleep` (should be `await asyncio.sleep`)
    *   `_get_vm_ip_address`: (if it performs synchronous Libvirt calls, it should be offloaded)
    *   `soft_reboot`: `agent_client.trigger_soft_reboot()` (which internally uses `requests`, so `SomnusVMAgentClient._request` needs to be offloaded).

2.  **Data Model Consistency**: The `AIVMInstance` BaseModel needs to be updated to include a `personality_config` field, as it is currently passed during `create_ai_computer` but not defined in the model.

3.  **Robust Soft Reboot Completion Signal**: (Repeated from `somnus_agent.py`) Implement a mechanism for the agent to explicitly signal back to the VM Supervisor when a soft reboot is truly complete and the core AI processes have restarted successfully.

4.  **Libvirt Connection Robustness**: Consider adding a retry mechanism or more explicit error handling and user guidance during the `libvirt.open()` call in the `__init__` method, especially if the connection fails.

5.  **Corrupted VM Config Handling**: Implement a more robust strategy for handling corrupted VM configuration files encountered by `_load_vms_from_disk` (e.g., moving them to a quarantine or backup location instead of just logging an error).
