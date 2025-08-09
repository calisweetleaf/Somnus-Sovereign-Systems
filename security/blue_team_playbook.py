#!/usr/bin/env python3
"""
Somnus Purple Team - Blue Team Playbook
=======================================

This module provides the core defensive and mitigation capabilities for the
Somnus Sovereign Systems architecture. It acts as the "shield" of the security
system, containing a library of automated responses to threats identified by
the monitoring systems or the Red Team Toolkit.

This playbook is designed to be invoked by the Purple Team Orchestrator. It
knows *how* to defend and respond, but relies on the orchestrator to decide
*when* to act.

Components:
- Data Models: Defines the structure for actions and their results.
- Defensive Actions: A collection of functions that perform concrete security
  mitigations, such as isolating VMs, patching configurations, and blocking IPs.
- Playbook Executor: A class that maps threat types to predefined sequences
  of defensive actions.
"""

import asyncio
import logging
import subprocess
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from uuid import UUID
import os
import json

# --- Assumed Imports from other Somnus Modules ---
# These imports are based on the existing Somnus architecture. The file paths
# may need adjustment based on the final project structure.

try:
    # From the previously created red_team_toolkit
    from red_team_toolkit import RedTeamTestResult, VulnerabilityClass
    # From the existing security and VM management framework
    from network_security import SomnusSecurityFramework, ThreatLevel
    from vm_supervisor import VMSupervisor, AIVMInstance
    from combined_security_system import ThreatIntelligenceEngine
except ImportError as e:
    # Fallback for standalone analysis, though functionality will be limited.
    logging.warning(f"Could not import all Somnus modules: {e}. Using fallback definitions.")
    # Re-define necessary enums if they can't be imported.
    class ThreatLevel(Enum): LOW = 1; MEDIUM = 2; HIGH = 3; CRITICAL = 4
    class VulnerabilityClass(Enum): 
        INJECTION = "injection"
        SECURITY_MISCONFIG = "security_misconfiguration"
        XSS = "cross_site_scripting"
        BROKEN_ACCESS = "broken_access_control"
        BROKEN_AUTH = "broken_authentication"
        SENSITIVE_DATA = "sensitive_data_exposure"
        XML_EXTERNAL = "xml_external_entities"
        INSECURE_DESERIALIZATION = "insecure_deserialization"
        KNOWN_VULNERABILITIES = "known_vulnerabilities"
        INSUFFICIENT_LOGGING = "insufficient_logging"
    # Define placeholder classes for type hinting - Import RedTeamTestResult from red_team_toolkit
    try:
        from red_team_toolkit import RedTeamTestResult
    except ImportError:
        # Only create fallback if red_team_toolkit isn't available
        from dataclasses import dataclass
        from typing import Optional
        @dataclass
        class RedTeamTestResult:
            test_id: str = ""
            vulnerability_found: bool = False
            vulnerability_class: Optional[VulnerabilityClass] = None
            severity: ThreatLevel = ThreatLevel.LOW
            target_id: str = ""
            exploit_payload: str = ""
            response_summary: str = ""
            timestamp: Any = None
            
    class SomnusSecurityFramework: pass
    class VMSupervisor: pass
    class AIVMInstance: pass
    class ThreatIntelligenceEngine: pass


logger = logging.getLogger(__name__)

# --- Data Models for Blue Team Operations ---

class ActionStatus(Enum):
    """The outcome status of a defensive action."""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING_APPROVAL = "pending_approval"
    NOT_APPLICABLE = "not_applicable"

@dataclass
class DefensiveActionResult:
    """The result of executing a single defensive action."""
    action_name: str
    status: ActionStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: str = ""
    error: Optional[str] = None

@dataclass
class PlaybookExecutionResult:
    """The comprehensive result of executing a full defensive playbook."""
    playbook_name: str
    triggering_event_id: str
    target_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    action_results: List[DefensiveActionResult] = field(default_factory=list)
    overall_status: ActionStatus = ActionStatus.SUCCESS

# --- Blue Team Playbook Engine ---

class BlueTeamPlaybook:
    """
    A library of automated defensive actions and the logic to execute them
    in response to specific threats.
    """

    def __init__(self,
                 vm_supervisor: VMSupervisor,
                 security_framework: SomnusSecurityFramework,
                 threat_intelligence: ThreatIntelligenceEngine):
        """
        Initializes the Blue Team Playbook with handles to the core system
        components it needs to control for defensive actions.
        """
        self.vm_supervisor = vm_supervisor
        self.security_framework = security_framework
        self.threat_intelligence = threat_intelligence
        self.audit_logger = security_framework.audit_logger
        self.playbook_map: Dict[VulnerabilityClass, Callable] = self._initialize_playbook_map()

    def _initialize_playbook_map(self) -> Dict[VulnerabilityClass, Callable]:
        """Maps vulnerability classes to their corresponding defensive playbook methods."""
        return {
            VulnerabilityClass.INJECTION: self.playbook_handle_injection,
            VulnerabilityClass.SECURITY_MISCONFIG: self.playbook_handle_misconfiguration,
            VulnerabilityClass.XSS: self.playbook_handle_injection, # Often similar response
            VulnerabilityClass.BROKEN_ACCESS: self.playbook_handle_access_control,
            # Add mappings for all other vulnerability classes here
        }

    async def execute_playbook_for_threat(self, threat_result: RedTeamTestResult) -> PlaybookExecutionResult:
        """
        The main entry point for the Blue Team. It selects and executes the
        appropriate defensive playbook based on a detected threat.

        Args:
            threat_result: The result object from a Red Team test.

        Returns:
            A PlaybookExecutionResult detailing all actions taken.
        """
        start_time = datetime.utcnow()
        playbook_name = f"playbook_for_{threat_result.vulnerability_class.value}"
        result = PlaybookExecutionResult(
            playbook_name=playbook_name,
            triggering_event_id=threat_result.test_id,
            target_id=threat_result.target_id,
            start_time=start_time
        )

        playbook_func = self.playbook_map.get(threat_result.vulnerability_class)

        if not playbook_func:
            not_applicable_action = DefensiveActionResult(
                action_name="select_playbook",
                status=ActionStatus.NOT_APPLICABLE,
                details=f"No playbook defined for vulnerability class '{threat_result.vulnerability_class.value}'."
            )
            result.action_results.append(not_applicable_action)
            result.overall_status = ActionStatus.FAILURE
        else:
            try:
                await playbook_func(threat_result, result)
            except Exception as e:
                logger.error(f"Critical error executing playbook {playbook_name}: {e}", exc_info=True)
                error_action = DefensiveActionResult(
                    action_name="playbook_execution",
                    status=ActionStatus.FAILURE,
                    error=str(e)
                )
                result.action_results.append(error_action)
                result.overall_status = ActionStatus.FAILURE

        result.end_time = datetime.utcnow()
        # Log the entire playbook result
        self.audit_logger.log_security_event(self._create_playbook_event(result))
        return result

    # --- Individual Defensive Playbooks ---

    async def playbook_handle_injection(self, threat: RedTeamTestResult, playbook_result: PlaybookExecutionResult):
        """Playbook for handling SQL injection, XSS, or command injection."""
        logger.warning(f"Executing INJECTION playbook for target {threat.target_id}.")
        
        # Action 1: Apply dynamic firewall/WAF rule if source IP is known
        if threat.metadata.get('source_ip'):
            await self._run_action(
                self.apply_dynamic_firewall_rule,
                playbook_result,
                ip_address=threat.metadata['source_ip'],
                reason=f"Suspicious injection activity from test {threat.test_id}"
            )
        
        # Action 2: Log a high-severity alert for review
        await self._run_action(
            self.log_high_severity_alert,
            playbook_result,
            message=f"Injection vulnerability detected on {threat.target_id}. Payload: {threat.exploit_payload}"
        )

    async def playbook_handle_misconfiguration(self, threat: RedTeamTestResult, playbook_result: PlaybookExecutionResult):
        """Playbook for handling security misconfigurations like privileged containers."""
        logger.warning(f"Executing MISCONFIGURATION playbook for target {threat.target_id}.")
        
        # Action 1: Snapshot the VM for forensics before making changes.
        vm_id = self._extract_vm_id(threat.target_id)
        if vm_id:
            await self._run_action(
                self.snapshot_and_quarantine_vm,
                playbook_result,
                vm_id=vm_id,
                reason=f"Critical misconfiguration detected: {threat.response_summary}"
            )

        # Action 2: If the threat is critical, immediately isolate the VM network.
        if threat.severity == ThreatLevel.CRITICAL and vm_id:
            await self._run_action(
                self.isolate_vm_network,
                playbook_result,
                vm_id=vm_id
            )

    async def playbook_handle_access_control(self, threat: RedTeamTestResult, playbook_result: PlaybookExecutionResult):
        """Playbook for handling broken access control vulnerabilities."""
        logger.warning(f"Executing ACCESS_CONTROL playbook for target {threat.target_id}.")
        
        # Action 1: Force-expire all sessions related to the compromised resource/VM.
        vm_id = self._extract_vm_id(threat.target_id)
        if vm_id:
            await self._run_action(
                self.force_expire_sessions,
                playbook_result,
                vm_id=vm_id
            )
        
        # Action 2: Log a high-severity alert for immediate review of permissions.
        await self._run_action(
            self.log_high_severity_alert,
            playbook_result,
            message=f"Broken access control detected on {threat.target_id}. Review IAM policies."
        )

    # --- Core Defensive Action Implementations ---

    async def isolate_vm_network(self, vm_id: UUID) -> Tuple[bool, str]:
        """Disconnects a VM's network interface using libvirt via the supervisor."""
        try:
            # Use the VM supervisor to isolate network interfaces
            if hasattr(self.vm_supervisor, 'isolate_vm_network'):
                success = await self.vm_supervisor.isolate_vm_network(vm_id)
                if success:
                    return True, f"Successfully isolated network for VM {vm_id}"
                else:
                    return False, f"Failed to isolate network for VM {vm_id}"
            else:
                # Fallback implementation using direct libvirt commands
                try:
                    # This would require subprocess execution of virsh commands
                    process = await asyncio.create_subprocess_exec(
                        'virsh', 'domif-setlink', str(vm_id), '0', 'down',
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        return True, f"Successfully isolated VM {vm_id} network interface"
                    else:
                        return False, f"Failed to isolate VM network: {stderr.decode()}"
                        
                except Exception as e:
                    return False, f"Error executing virsh command: {e}"
                    
        except Exception as e:
            logger.error(f"Error isolating network for VM {vm_id}: {e}")
            return False, str(e)

    async def snapshot_and_quarantine_vm(self, vm_id: UUID, reason: str) -> Tuple[bool, str]:
        """Takes a forensic snapshot and then shuts down the VM."""
        try:
            # Step 1: Create snapshot
            if hasattr(self.vm_supervisor, 'create_snapshot'):
                snapshot_result = await self.vm_supervisor.create_snapshot(
                    vm_id, 
                    description=f"Security quarantine: {reason}"
                )
            else:
                # Fallback to direct virsh command
                try:
                    snapshot_name = f"security-snapshot-{int(time.time())}"
                    process = await asyncio.create_subprocess_exec(
                        'virsh', 'snapshot-create-as', str(vm_id), snapshot_name,
                        f"Security snapshot: {reason}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    snapshot_result = process.returncode == 0
                    if not snapshot_result:
                        return False, f"Failed to create snapshot: {stderr.decode()}"
                except Exception as e:
                    return False, f"Error creating snapshot: {e}"
            
            # Step 2: Shutdown VM
            if hasattr(self.vm_supervisor, 'shutdown_vm'):
                shutdown_result = await self.vm_supervisor.shutdown_vm(vm_id)
            else:
                # Fallback to direct virsh command
                try:
                    process = await asyncio.create_subprocess_exec(
                        'virsh', 'shutdown', str(vm_id),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    shutdown_result = process.returncode == 0
                    if not shutdown_result:
                        return False, f"Failed to shutdown VM: {stderr.decode()}"
                except Exception as e:
                    return False, f"Error shutting down VM: {e}"
            
            if snapshot_result and shutdown_result:
                return True, f"Successfully quarantined VM {vm_id} with snapshot"
            else:
                return False, "Failed to complete snapshot or shutdown"
                
        except Exception as e:
            logger.error(f"Error quarantining VM {vm_id}: {e}")
            return False, str(e)

    async def apply_dynamic_firewall_rule(self, ip_address: str, reason: str) -> Tuple[bool, str]:
        """Blocks an IP address at the host firewall level."""
        try:
            # Try to use iptables to block the IP
            comment = f"SomnusBlueTeam-{reason}"[:50]  # Limit comment length
            
            # Check if iptables rule already exists
            check_process = await asyncio.create_subprocess_exec(
                'iptables', '-C', 'INPUT', '-s', ip_address, '-j', 'DROP',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await check_process.communicate()
            
            if check_process.returncode == 0:
                return True, f"IP {ip_address} already blocked"
            
            # Add the blocking rule
            process = await asyncio.create_subprocess_exec(
                'iptables', '-A', 'INPUT', '-s', ip_address, '-j', 'DROP',
                '-m', 'comment', '--comment', comment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Also try to notify the security framework if available
                if hasattr(self.security_framework, 'block_ip'):
                    try:
                        self.security_framework.block_ip(ip_address, reason)
                    except Exception as e:
                        logger.debug(f"Could not notify security framework: {e}")
                
                return True, f"Successfully blocked IP {ip_address}"
            else:
                return False, f"Failed to block IP {ip_address}: {stderr.decode()}"
                
        except FileNotFoundError:
            return False, "iptables command not found - cannot apply firewall rule"
        except PermissionError:
            return False, "Insufficient permissions to modify iptables rules"
        except Exception as e:
            logger.error(f"Error applying firewall rule for {ip_address}: {e}")
            return False, str(e)

    async def force_ai_soft_reboot(self, vm_id: UUID) -> Tuple[bool, str]:
        """Triggers a soft reboot of the AI processes inside the VM."""
        try:
            # Try to use VM supervisor's soft reboot capability
            if hasattr(self.vm_supervisor, 'soft_reboot_ai'):
                success = await self.vm_supervisor.soft_reboot_ai(vm_id)
                if success:
                    return True, f"Successfully triggered AI soft reboot for VM {vm_id}"
                else:
                    return False, "VM supervisor failed to trigger AI soft reboot"
                    
            elif hasattr(self.vm_supervisor, 'send_agent_command'):
                # Try to send a reboot command to the VM agent
                try:
                    result = await self.vm_supervisor.send_agent_command(vm_id, "restart_ai_services")
                    if result.get('success'):
                        return True, f"Successfully sent AI restart command to VM {vm_id}"
                    else:
                        return False, f"Agent command failed: {result.get('error', 'Unknown error')}"
                except Exception as e:
                    return False, f"Failed to send agent command: {e}"
            else:
                # Fallback: restart the entire VM (more disruptive)
                try:
                    process = await asyncio.create_subprocess_exec(
                        'virsh', 'reboot', str(vm_id),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        return True, f"Successfully rebooted VM {vm_id} (full reboot fallback)"
                    else:
                        return False, f"Failed to reboot VM: {stderr.decode()}"
                except Exception as e:
                    return False, f"Error executing VM reboot: {e}"
                    
        except Exception as e:
            logger.error(f"Error triggering soft reboot for VM {vm_id}: {e}")
            return False, str(e)

    async def force_expire_sessions(self, vm_id: UUID) -> Tuple[bool, str]:
        """Forces expiration of all sessions associated with a VM with multiple fallback mechanisms."""
        success_methods = []
        failure_reasons = []
        
        try:
            # Method 1: Security framework session management
            if hasattr(self.security_framework, 'expire_vm_sessions'):
                try:
                    success = await asyncio.wait_for(
                        self.security_framework.expire_vm_sessions(vm_id), 
                        timeout=30.0
                    )
                    if success:
                        success_methods.append("security_framework.expire_vm_sessions")
                    else:
                        failure_reasons.append("Security framework reported failure")
                except asyncio.TimeoutError:
                    failure_reasons.append("Security framework timeout")
                except Exception as e:
                    failure_reasons.append(f"Security framework error: {e}")
            
            # Method 2: VM supervisor session management
            if hasattr(self.vm_supervisor, 'expire_vm_sessions'):
                try:
                    success = await asyncio.wait_for(
                        self.vm_supervisor.expire_vm_sessions(vm_id), 
                        timeout=30.0
                    )
                    if success:
                        success_methods.append("vm_supervisor.expire_vm_sessions")
                    else:
                        failure_reasons.append("VM supervisor reported failure")
                except asyncio.TimeoutError:
                    failure_reasons.append("VM supervisor timeout")
                except Exception as e:
                    failure_reasons.append(f"VM supervisor error: {e}")
            
            # Method 3: Direct session termination
            if hasattr(self.security_framework, 'terminate_vm_sessions'):
                try:
                    await asyncio.wait_for(
                        self.security_framework.terminate_vm_sessions(str(vm_id)), 
                        timeout=30.0
                    )
                    success_methods.append("security_framework.terminate_vm_sessions")
                except asyncio.TimeoutError:
                    failure_reasons.append("Session termination timeout")
                except Exception as e:
                    failure_reasons.append(f"Session termination error: {e}")
            
            # Method 4: VM agent command with retry
            if hasattr(self.vm_supervisor, 'send_agent_command'):
                for attempt in range(3):  # 3 retry attempts
                    try:
                        result = await asyncio.wait_for(
                            self.vm_supervisor.send_agent_command(vm_id, "clear_all_sessions"),
                            timeout=15.0
                        )
                        if result and result.get('success'):
                            success_methods.append(f"vm_agent.clear_all_sessions (attempt {attempt + 1})")
                            break
                        else:
                            if attempt == 2:  # Last attempt
                                failure_reasons.append(f"Agent command failed after 3 attempts: {result.get('error', 'Unknown error')}")
                    except asyncio.TimeoutError:
                        if attempt == 2:
                            failure_reasons.append("Agent command timeout after 3 attempts")
                    except Exception as e:
                        if attempt == 2:
                            failure_reasons.append(f"Agent command error after 3 attempts: {e}")
                    
                    if attempt < 2:  # Don't wait after last attempt
                        await asyncio.sleep(2.0)  # Wait before retry
            
            # Method 5: System-level process termination
            if hasattr(self.vm_supervisor, 'execute_vm_command'):
                try:
                    # Kill session-related processes
                    commands = [
                        "pkill -f 'session|auth|token' || true",
                        "systemctl --user stop session-manager || true",
                        "rm -f /tmp/session_* /var/run/session_* || true",
                        "find /proc -name 'session*' -type f -delete 2>/dev/null || true"
                    ]
                    
                    for cmd in commands:
                        try:
                            result = await asyncio.wait_for(
                                self.vm_supervisor.execute_vm_command(vm_id, cmd),
                                timeout=10.0
                            )
                            if result and result.get('success'):
                                success_methods.append(f"system_cleanup: {cmd[:20]}...")
                        except Exception as e:
                            logger.debug(f"System cleanup command failed: {e}")
                            
                except Exception as e:
                    failure_reasons.append(f"System cleanup error: {e}")
            
            # Method 6: Network-based session invalidation
            if hasattr(self.vm_supervisor, 'reset_vm_networking'):
                try:
                    success = await asyncio.wait_for(
                        self.vm_supervisor.reset_vm_networking(vm_id),
                        timeout=20.0
                    )
                    if success:
                        success_methods.append("vm_supervisor.reset_vm_networking")
                    else:
                        failure_reasons.append("Network reset failed")
                except Exception as e:
                    failure_reasons.append(f"Network reset error: {e}")
            
            # Audit logging
            audit_data = {
                'vm_id': str(vm_id),
                'action': 'force_expire_sessions',
                'success_methods': success_methods,
                'failure_reasons': failure_reasons,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if hasattr(self.audit_logger, 'log_security_event'):
                try:
                    self.audit_logger.log_security_event(audit_data)
                except Exception as e:
                    logger.error(f"Failed to log audit event: {e}")
            
            # Determine overall success
            if success_methods:
                return True, f"Session expiration successful via: {', '.join(success_methods)}"
            else:
                return False, f"All session expiration methods failed: {'; '.join(failure_reasons)}"
                
        except Exception as e:
            logger.error(f"Critical error in force_expire_sessions for VM {vm_id}: {e}", exc_info=True)
            return False, f"Critical error: {e}"

    async def log_high_severity_alert(self, message: str) -> Tuple[bool, str]:
        """Logs high-severity security alerts with multiple delivery mechanisms."""
        success_methods = []
        failure_reasons = []
        
        try:
            # Validate input
            if not message or not message.strip():
                return False, "Alert message cannot be empty"
            
            # Sanitize message for logging
            sanitized_message = message.replace('\n', ' ').replace('\r', ' ')[:2000]  # Limit length
            
            # Method 1: Audit logger security event
            if hasattr(self.audit_logger, 'log_security_event'):
                try:
                    event_data = {
                        'event_type': 'HIGH_SEVERITY_ALERT',
                        'message': sanitized_message,
                        'timestamp': datetime.utcnow().isoformat(),
                        'severity': 'HIGH',
                        'source': 'blue_team_playbook',
                        'event_id': f"alert_{int(time.time() * 1000)}"
                    }
                    
                    await asyncio.wait_for(
                        self.audit_logger.log_security_event(event_data),
                        timeout=10.0
                    )
                    success_methods.append("audit_logger.log_security_event")
                except asyncio.TimeoutError:
                    failure_reasons.append("Audit logger timeout")
                except Exception as e:
                    failure_reasons.append(f"Audit logger error: {e}")
            
            # Method 2: Generic audit logging
            if hasattr(self.audit_logger, 'log'):
                try:
                    await asyncio.wait_for(
                        self.audit_logger.log('HIGH', sanitized_message),
                        timeout=5.0
                    )
                    success_methods.append("audit_logger.log")
                except asyncio.TimeoutError:
                    failure_reasons.append("Generic audit log timeout")
                except Exception as e:
                    failure_reasons.append(f"Generic audit log error: {e}")
            
            # Method 3: Security framework alerting
            if hasattr(self.security_framework, 'send_alert'):
                try:
                    await asyncio.wait_for(
                        self.security_framework.send_alert(sanitized_message, 'HIGH'),
                        timeout=10.0
                    )
                    success_methods.append("security_framework.send_alert")
                except asyncio.TimeoutError:
                    failure_reasons.append("Security framework alert timeout")
                except Exception as e:
                    failure_reasons.append(f"Security framework alert error: {e}")
            
            # Method 4: Threat intelligence notification
            if hasattr(self.threat_intelligence, 'log_security_incident'):
                try:
                    incident_data = {
                        'message': sanitized_message,
                        'severity': 'HIGH',
                        'timestamp': datetime.utcnow().isoformat(),
                        'source': 'blue_team_defensive_action'
                    }
                    
                    await asyncio.wait_for(
                        self.threat_intelligence.log_security_incident(incident_data),
                        timeout=10.0
                    )
                    success_methods.append("threat_intelligence.log_security_incident")
                except asyncio.TimeoutError:
                    failure_reasons.append("Threat intelligence timeout")
                except Exception as e:
                    failure_reasons.append(f"Threat intelligence error: {e}")
            
            # Method 5: File-based emergency logging
            try:
                log_dir = "/var/log/somnus_security"
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, mode=0o750, exist_ok=True)
                
                log_file = os.path.join(log_dir, f"high_severity_alerts_{datetime.utcnow().strftime('%Y%m%d')}.log")
                
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'severity': 'HIGH',
                    'message': sanitized_message,
                    'source': 'blue_team_playbook'
                }
                
                async with aiofiles.open(log_file, 'a') as f:
                    await f.write(f"{json.dumps(log_entry)}\n")
                
                success_methods.append("file_based_logging")
            except Exception as e:
                failure_reasons.append(f"File logging error: {e}")
            
            # Method 6: Python logging (always available fallback)
            try:
                logger.critical(f"HIGH SEVERITY ALERT: {sanitized_message}")
                success_methods.append("python_logging")
            except Exception as e:
                failure_reasons.append(f"Python logging error: {e}")
            
            # Method 7: Syslog for system integration
            try:
                import syslog
                syslog.openlog("somnus_security", syslog.LOG_PID, syslog.LOG_SECURITY)
                syslog.syslog(syslog.LOG_CRIT, f"HIGH SEVERITY ALERT: {sanitized_message}")
                syslog.closelog()
                success_methods.append("syslog")
            except Exception as e:
                failure_reasons.append(f"Syslog error: {e}")
            
            # Determine overall success
            if success_methods:
                return True, f"Alert logged successfully via: {', '.join(success_methods)}"
            else:
                return False, f"All alert logging methods failed: {'; '.join(failure_reasons)}"
                
        except Exception as e:
            logger.error(f"Critical error in log_high_severity_alert: {e}", exc_info=True)
            # Emergency fallback
            try:
                logger.critical(f"EMERGENCY ALERT (fallback): {message}")
                return False, f"Critical error, used emergency fallback: {e}"
            except:
                return False, f"Complete logging failure: {e}"

    # --- Helper Methods ---
    
    async def _run_action(self, action_func: Callable, playbook_result: PlaybookExecutionResult, **kwargs):
        """A wrapper to execute a defensive action and record its result."""
        action_name = action_func.__name__
        action_result = DefensiveActionResult(action_name=action_name)
        try:
            success, details = await action_func(**kwargs)
            action_result.status = ActionStatus.SUCCESS if success else ActionStatus.FAILURE
            action_result.details = details
        except Exception as e:
            logger.error(f"Exception during defensive action {action_name}: {e}", exc_info=True)
            action_result.status = ActionStatus.FAILURE
            action_result.error = str(e)
        
        playbook_result.action_results.append(action_result)
        # If any action fails, the whole playbook is marked as failed.
        if action_result.status == ActionStatus.FAILURE:
            playbook_result.overall_status = ActionStatus.FAILURE
            
    def _extract_vm_id(self, target_id: str) -> Optional[UUID]:
        """Helper to safely extract a VM UUID from a target ID string."""
        try:
            # This is a simple implementation. A more robust one might parse URLs
            # or other complex target identifiers.
            return UUID(target_id)
        except (ValueError, TypeError):
            # Could also attempt to look up VM by name via vm_supervisor if needed
            logger.warning(f"Could not extract a valid VM UUID from target_id: {target_id}")
            return None
            
    def _create_playbook_event(self, playbook_result: PlaybookExecutionResult) -> Any:
        """Creates a SecurityEvent object from a playbook result for logging."""
        # This assumes the existence of the SecurityEvent class from network_security
        try:
            from network_security import SecurityEvent
        except ImportError:
            # Fallback: create a simple dict for logging
            class SecurityEvent:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
        
        details = {
            "playbook_name": playbook_result.playbook_name,
            "triggering_event": playbook_result.triggering_event_id,
            "overall_status": playbook_result.overall_status.value,
            "duration_ms": (playbook_result.end_time - playbook_result.start_time).total_seconds() * 1000 if playbook_result.end_time else 0,
            "actions": [
                {
                    "action": ar.action_name,
                    "status": ar.status.value,
                    "details": ar.details,
                    "error": ar.error
                } for ar in playbook_result.action_results
            ]
        }
        
        return SecurityEvent(
            event_type="blue_team_playbook_executed",
            threat_level=ThreatLevel.HIGH, # Defensive actions are always high-importance events
            vm_id=playbook_result.target_id,
            timestamp=playbook_result.start_time,
            details=details
        )
    
    def _create_playbook_event_for_manual_review(self, threat_result) -> Any:
        """Creates a SecurityEvent for threats that require manual review."""
        try:
            from network_security import SecurityEvent
        except ImportError:
            class SecurityEvent:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
        
        details = {
            "test_id": threat_result.test_id,
            "vulnerability_class": threat_result.vulnerability_class.value if threat_result.vulnerability_class else "unknown",
            "severity": threat_result.severity.name if hasattr(threat_result.severity, 'name') else str(threat_result.severity),
            "target_id": threat_result.target_id,
            "exploit_payload": threat_result.exploit_payload,
            "response_summary": threat_result.response_summary,
            "requires_manual_review": True
        }
        
        return SecurityEvent(
            event_type="vulnerability_requires_manual_review",
            threat_level=threat_result.severity,
            vm_id=threat_result.target_id,
            timestamp=threat_result.timestamp,
            details=details
        )


if __name__ == '__main__':
    # This block allows for standalone conceptual testing of the playbook.
    # In a real run, the dependencies would be provided by the main application.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("--- Blue Team Playbook Standalone Test ---")
    print("This module defines a library of defensive actions and playbooks.")
    print("It is designed to be initialized and called by the Purple Team Orchestrator.")
    
    # Conceptual usage:
    # 1. Initialize with real system components:
    #    playbook = BlueTeamPlaybook(vm_supervisor, security_framework, threat_intel)
    #
    # 2. A threat is detected by the Red Team or monitoring:
    #    threat = RedTeamTestResult(...)
    #
    # 3. The orchestrator invokes the playbook:
    #    async def run_playbook():
    #        result = await playbook.execute_playbook_for_threat(threat)
    #        print(f"Playbook executed with status: {result.overall_status}")
    #    asyncio.run(run_playbook())

    print("\n--- Test Complete ---")
    print("This module is not intended to be run directly. It is part of the Somnus Purple Team architecture.")