#!/usr/bin/env python3
"""
Somnus Purple Team - Orchestrator
=================================

This module is the central nervous system of the Somnus security architecture.
It acts as the "brain" or "Purple Team," coordinating the offensive actions
of the Red Team Toolkit with the defensive responses of the Blue Team Playbook.

The orchestrator runs a continuous, autonomous loop to:
1.  Decide which systems to test and with what attack vector.
2.  Command the Red Team Toolkit to execute the attack.
3.  Analyze the results of the attack.
4.  If a vulnerability is found, command the Blue Team Playbook to execute
    a defensive response.
5.  Correlate data with the Threat Intelligence Engine to learn and adapt.
6.  Log all activities for audit and review.
"""

import asyncio
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import UUID

# --- Core Imports from the new Security Architecture ---
from red_team_toolkit import RedTeamToolkit, RedTeamTestType, RedTeamTestResult
from blue_team_playbook import BlueTeamPlaybook

# --- Assumed Imports from the existing Somnus Modules ---
try:
    from vm_supervisor import VMSupervisor, AIVMInstance, VMState, SomnusVMAgentClient
    from network_security import SomnusSecurityFramework, ThreatLevel
    from combined_security_system import ThreatIntelligenceEngine
except ImportError as e:
    logging.warning(f"Could not import all Somnus modules: {e}. Using fallback definitions.")
    # Define placeholder classes for type hinting if imports fail
    class VMSupervisor: pass
    class AIVMInstance: pass
    class VMState(str): RUNNING = "running"
    class SomnusVMAgentClient: pass
    class SomnusSecurityFramework: pass
    class ThreatIntelligenceEngine: pass
    class ThreatLevel(int): CRITICAL = 4

logger = logging.getLogger(__name__)

# --- Orchestrator ---

class PurpleTeamOrchestrator:
    """
    The high-level orchestrator that runs the continuous security loop,
    deciding what to do and when.
    """

    def __init__(self,
                 vm_supervisor: VMSupervisor,
                 security_framework: SomnusSecurityFramework,
                 red_team_toolkit: RedTeamToolkit,
                 blue_team_playbook: BlueTeamPlaybook,
                 threat_intelligence: ThreatIntelligenceEngine,
                 config_path: str = "purple_team_config.yaml"):
        """
        Initializes the orchestrator with handles to all its dependencies.
        """
        self.vm_supervisor = vm_supervisor
        self.security_framework = security_framework
        self.red_team_toolkit = red_team_toolkit
        self.blue_team_playbook = blue_team_playbook
        self.threat_intelligence = threat_intelligence
        self.audit_logger = security_framework.audit_logger
        self.config = self._load_config(config_path)

        self.active_tests: Dict[str, asyncio.Task] = {}
        self.test_results_history: List[RedTeamTestResult] = []
        self._stop_event = asyncio.Event()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads orchestrator configuration from a YAML file."""
        default_config = {
            "test_intervals_sec": {
                "prompt_injection": 300,  # 5 minutes
                "api_fuzzing": 900,       # 15 minutes
                "container_escape": 1800, # 30 minutes
            },
            "max_concurrent_tests_per_type": 2,
            "test_aggressiveness": "moderate", # "low", "moderate", "high"
            "auto_remediate_threshold": "HIGH",
        }
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
            logger.info(f"Loaded Purple Team config from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. Using default settings.")
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
        return default_config

    async def start_continuous_operations(self):
        """Starts the main, continuous security testing loops."""
        if self._stop_event.is_set():
            self._stop_event.clear()

        logger.info("Purple Team Orchestrator starting continuous operations...")
        self.main_task = asyncio.gather(
            self._continuous_testing_loop(
                test_type=RedTeamTestType.PROMPT_INJECTION,
                interval=self.config['test_intervals_sec']['prompt_injection'],
                test_function=self._run_prompt_injection_on_vm
            ),
            self._continuous_testing_loop(
                test_type=RedTeamTestType.CONTAINER_ESCAPE,
                interval=self.config['test_intervals_sec']['container_escape'],
                test_function=self._run_container_scan_on_vm
            ),
            # Add other test loops here, e.g., API fuzzing
            return_exceptions=True
        )
        await self.main_task

    async def stop_operations(self):
        """Gracefully stops all testing operations."""
        logger.info("Purple Team Orchestrator stopping operations...")
        self._stop_event.set()
        if self.main_task:
            self.main_task.cancel()
            for task in self.active_tests.values():
                task.cancel()
            try:
                await self.main_task
            except asyncio.CancelledError:
                pass
        logger.info("All operations stopped.")

    async def _continuous_testing_loop(self, test_type: RedTeamTestType, interval: int, test_function: Callable):
        """A generic, continuous loop for running a specific type of test."""
        while not self._stop_event.is_set():
            try:
                logger.info(f"Starting new test cycle for: {test_type.value}")
                active_vms = await self.vm_supervisor.list_vms()
                running_vms = [vm for vm in active_vms if vm.vm_state == VMState.RUNNING]

                # Clean up completed tasks before starting new ones
                await self._cleanup_completed_tasks()

                # Schedule new tests, respecting concurrency limits
                for vm in running_vms:
                    if self._can_schedule_new_test(test_type):
                        task_id = f"{test_type.value}_{vm.vm_id}"
                        if task_id not in self.active_tests:
                            task = asyncio.create_task(test_function(vm))
                            self.active_tests[task_id] = task
                    else:
                        logger.debug(f"Concurrency limit reached for {test_type.value}. Deferring tests.")
                        break
                
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                logger.info(f"Test loop for {test_type.value} cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in continuous testing loop for {test_type.value}: {e}", exc_info=True)
                await asyncio.sleep(60) # Back off on error

    async def _run_prompt_injection_on_vm(self, vm: AIVMInstance):
        """Orchestrates a prompt injection test against a specific VM."""
        logger.debug(f"Running prompt injection test on VM {vm.vm_id}")
        # This assumes the RedTeamToolkit has a method to perform the test.
        # The toolkit would need access to the VM's agent to send the prompt.
        # For this example, we'll assume the toolkit handles that interaction.
        # test_result = await self.red_team_toolkit.perform_prompt_injection(vm)
        
        # Since the toolkit doesn't have the agent client, we pass it here.
        # This is a more realistic workflow.
        agent_client = SomnusVMAgentClient(vm.internal_ip, vm.agent_port)
        test_result = await self.red_team_toolkit.test_prompt_injection_on_agent(agent_client)
        
        await self._process_test_result(test_result)

    async def _run_container_scan_on_vm(self, vm: AIVMInstance):
        """Orchestrates a container escape scan on a specific VM."""
        logger.debug(f"Running container escape scan on VM {vm.vm_id}")
        agent_client = SomnusVMAgentClient(vm.internal_ip, vm.agent_port)
        
        # Call the specific check from the toolkit
        test_result = await self.red_team_toolkit.container_escape_detector.check_for_privileged_container(agent_client)
        
        await self._process_test_result(test_result)
        
    async def _process_test_result(self, result: RedTeamTestResult):
        """Analyzes a test result and triggers a Blue Team response if needed."""
        self.test_results_history.append(result)
        if len(self.test_results_history) > 1000: # Cap history size
            self.test_results_history.pop(0)

        if result.vulnerability_found:
            vuln_class_name = result.vulnerability_class.value if result.vulnerability_class else "unknown"
            severity_name = result.severity.name if hasattr(result.severity, 'name') else str(result.severity)
            
            logger.warning(
                f"Vulnerability Found! Test ID: {result.test_id}, "
                f"Type: {vuln_class_name}, "
                f"Target: {result.target_id}, "
                f"Severity: {severity_name}"
            )
            
            # Correlate with threat intelligence (optional, for future enhancement)
            # self.threat_intelligence.correlate_red_team_finding(result)

            # Check if auto-remediation is enabled for this severity
            auto_remediate_threshold = self.config.get("auto_remediate_threshold", "HIGH")
            if (hasattr(result.severity, 'name') and result.severity.name == auto_remediate_threshold) or \
               (hasattr(result.severity, 'value') and result.severity.value >= 3):  # HIGH = 3, CRITICAL = 4
                logger.info(f"Severity meets auto-remediation threshold. Executing Blue Team playbook.")
                await self.blue_team_playbook.execute_playbook_for_threat(result)
            else:
                logger.info("Vulnerability found, but severity is below auto-remediation threshold. Logging for manual review.")
                # Log for manual review
                if hasattr(self, 'audit_logger') and self.audit_logger:
                    try:
                        self.audit_logger.log_security_event(
                            self.blue_team_playbook._create_playbook_event_for_manual_review(result)
                        )
                    except Exception as e:
                        logger.error(f"Failed to log manual review event: {e}")

    async def _cleanup_completed_tasks(self):
        """Removes finished tasks from the active tests dictionary."""
        completed_ids = [task_id for task_id, task in self.active_tests.items() if task.done()]
        for task_id in completed_ids:
            try:
                await self.active_tests[task_id]  # Await to raise potential exceptions
            except asyncio.CancelledError:
                pass # Task was cancelled, which is fine.
            except Exception as e:
                logger.error(f"Test task {task_id} completed with an error: {e}", exc_info=True)
            del self.active_tests[task_id]
        if completed_ids:
            logger.debug(f"Cleaned up {len(completed_ids)} completed test tasks.")

    def _can_schedule_new_test(self, test_type: RedTeamTestType) -> bool:
        """Checks if a new test of a given type can be scheduled based on concurrency limits."""
        current_count = sum(1 for task_id in self.active_tests if task_id.startswith(test_type.value))
        return current_count < self.config['max_concurrent_tests_per_type']

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Provides a snapshot of the orchestrator's current state and statistics."""
        vulnerabilities_found = sum(1 for r in self.test_results_history if r.vulnerability_found)
        return {
            "status": "running" if not self._stop_event.is_set() else "stopped",
            "config": self.config,
            "active_tests_count": len(self.active_tests),
            "total_tests_run_in_session": len(self.test_results_history),
            "total_vulnerabilities_found_in_session": vulnerabilities_found,
            "active_test_details": list(self.active_tests.keys())
        }

# --- Factory and Main Execution ---

def create_purple_team_orchestrator(
    vm_supervisor: VMSupervisor,
    security_framework: SomnusSecurityFramework,
    threat_intelligence: ThreatIntelligenceEngine
) -> PurpleTeamOrchestrator:
    """
    Factory function to initialize and wire up all components of the
    Purple Team security system.
    """
    red_toolkit = RedTeamToolkit()
    blue_playbook = BlueTeamPlaybook(vm_supervisor, security_framework, threat_intelligence)
    orchestrator = PurpleTeamOrchestrator(
        vm_supervisor,
        security_framework,
        red_toolkit,
        blue_playbook,
        threat_intelligence
    )
    logger.info("Purple Team security system successfully initialized.")
    return orchestrator

if __name__ == '__main__':
    # This block demonstrates how the orchestrator would be initialized and run
    # within the main Somnus application.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def main():
        logger.info("--- Purple Team Orchestrator Standalone Demonstration ---")

        # In a real application, these components would be initialized once
        # and passed to the factory function.
        # Here, we use placeholder objects for demonstration.
        class MockLogger:
            def log_security_event(self, event):
                print(f"[AUDIT LOG] {event.event_type}: {event.details}")
        class MockSecurityFramework:
            def __init__(self): self.audit_logger = MockLogger()
        class MockVMSupervisor:
            async def list_vms(self): return []
        class MockThreatIntel: pass

        mock_supervisor = MockVMSupervisor()
        mock_sec_fw = MockSecurityFramework()
        mock_threat_intel = MockThreatIntel()

        # 1. Create the orchestrator using the factory
        orchestrator = create_purple_team_orchestrator(
            mock_supervisor,
            mock_sec_fw,
            mock_threat_intel
        )

        # 2. Start the continuous operations in the background
        logger.info("Starting orchestrator operations...")
        main_op_task = asyncio.create_task(orchestrator.start_continuous_operations())

        # 3. The main application would continue running here.
        # We'll simulate a runtime and then gracefully shut down.
        try:
            for i in range(10):
                status = orchestrator.get_orchestrator_status()
                print(f"[{datetime.utcnow()}] Orchestrator Status: {status['active_tests_count']} active tests.")
                await asyncio.sleep(5)
        finally:
            logger.info("Shutting down orchestrator...")
            await orchestrator.stop_operations()
            await main_op_task
            logger.info("Demonstration finished.")

    asyncio.run(main())
# --- End of purple_team_orchestrator.py ---
# This module is not intended to be run directly. It is part of the Somnus Purple Team architecture.
# It is designed to be imported and used within the Somnus security framework.