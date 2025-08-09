#!/usr/bin/env python3
"""
Somnus Purple Team Agent - Blue Team Module
===========================================

Advanced defensive security operations with real-time threat monitoring, ML-driven anomaly detection,
automated self-healing, and dynamic policy enforcement. Integrates with existing Somnus architecture
for comprehensive defense-in-depth protection.

Integration Points:
- SecurityEvent/ThreatLevel from existing security framework
- SecurityAuditLogger for centralized logging
- VMSupervisor for automated response and isolation
- SomnusVMAgentClient for in-VM defensive actions
- ThreatIntelligenceEngine for threat correlation
"""

import asyncio
import json
import logging
import os
import pickle
import psutil
import re
import sqlite3
import subprocess
import syslog
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
from queue import Queue, Empty

# Data science and ML imports
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# File system and network monitoring
import watchdog.observers
from watchdog.events import FileSystemEventHandler
import requests
import yaml
import ipaddress
import socket
import psycopg2

# Metrics and monitoring
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Import existing Somnus security components
from combined_security_system import (
    UnifiedThreatLevel,
    UnifiedSecurityResult,
    ThreatIntelligenceEngine,
    IntegratedSecurityOrchestrator
)
from network_security import (
    SecurityEvent,
    ThreatLevel,
    SecurityAuditLogger,
    SomnusSecurityFramework
)
from vm_supervisor import (
    VMSupervisor,
    SomnusVMAgentClient,
    AIVMInstance,
    VMState
)

logger = logging.getLogger(__name__)

# Prometheus metrics for monitoring
THREATS_DETECTED = Counter('somnus_threats_detected_total', 'Total threats detected by type', ['threat_type'])
ANOMALIES_DETECTED = Counter('somnus_anomalies_detected_total', 'Total anomalies detected by source', ['source'])
RESPONSE_TIME = Histogram('somnus_response_time_seconds', 'Response time for defensive actions', ['action_type'])
ACTIVE_MONITORS = Gauge('somnus_active_monitors', 'Number of active monitoring processes')
SYSTEM_HEALTH = Gauge('somnus_system_health_score', 'Overall system health score (0-100)')


class MonitoringSource(Enum):
    """Sources of security monitoring data"""
    SYSTEM_LOGS = "system_logs"
    NETWORK_TRAFFIC = "network_traffic"
    PROCESS_ACTIVITY = "process_activity"
    FILE_SYSTEM = "file_system"
    API_TRAFFIC = "api_traffic"
    AGENT_BEHAVIOR = "agent_behavior"
    RESOURCE_USAGE = "resource_usage"
    DATABASE_ACTIVITY = "database_activity"


class ResponseAction(Enum):
    """Types of automated defensive responses"""
    ALERT_ONLY = "alert_only"
    RATE_LIMIT = "rate_limit"
    QUARANTINE_PROCESS = "quarantine_process"
    ISOLATE_VM = "isolate_vm"
    RESTART_SERVICE = "restart_service"
    BLOCK_IP = "block_ip"
    REVOKE_PERMISSIONS = "revoke_permissions"
    CREATE_SNAPSHOT = "create_snapshot"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class AnomalySignature:
    """Machine learning signature for anomaly detection"""
    signature_id: str
    source: MonitoringSource
    feature_vector: np.ndarray
    confidence_score: float
    threat_level: ThreatLevel
    pattern_description: str
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    occurrence_count: int = 1
    false_positive_rate: float = 0.0


@dataclass
class DefensiveAction:
    """Record of defensive action taken"""
    action_id: str = field(default_factory=lambda: f"def_{int(time.time())}_{os.urandom(4).hex()}")
    action_type: ResponseAction = ResponseAction.ALERT_ONLY
    trigger_event: str = ""
    target_vm_id: Optional[str] = None
    target_process_id: Optional[int] = None
    target_ip: Optional[str] = None
    execution_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    rollback_possible: bool = True
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class LogAnalysisEngine:
    """Real-time log analysis and threat detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_patterns = self._compile_threat_patterns()
        self.log_queues: Dict[str, Queue] = {}
        self.active_monitors: Dict[str, Union[threading.Thread, asyncio.Task]] = {}
        self.threat_counters = defaultdict(int)
        self.recent_events = deque(maxlen=10000)
        
        # ML models for log anomaly detection
        self.log_vectorizer = None
        self.log_anomaly_model = None
        self._initialize_ml_models()
    
    def _compile_threat_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for threat detection in logs"""
        patterns = {
            "injection_attempts": [
                re.compile(r"(union\s+select|drop\s+table|exec\s*\(|eval\s*\()", re.IGNORECASE),
                re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE),
                re.compile(r"javascript:\s*alert\s*\(", re.IGNORECASE),
                re.compile(r"\$\{.*?\}", re.IGNORECASE),  # SSTI patterns
                re.compile(r"{{.*?}}", re.IGNORECASE),
            ],
            "authentication_anomalies": [
                re.compile(r"authentication\s+failed", re.IGNORECASE),
                re.compile(r"invalid\s+credentials", re.IGNORECASE),
                re.compile(r"login\s+attempt.*?failed", re.IGNORECASE),
                re.compile(r"brute\s*force", re.IGNORECASE),
            ],
            "privilege_escalation": [
                re.compile(r"sudo\s+.*?root", re.IGNORECASE),
                re.compile(r"setuid.*?root", re.IGNORECASE),
                re.compile(r"privilege.*?escalat", re.IGNORECASE),
                re.compile(r"unauthorized\s+access", re.IGNORECASE),
            ],
            "data_exfiltration": [
                re.compile(r"large\s+data\s+transfer", re.IGNORECASE),
                re.compile(r"downloading.*?\d+\s*mb", re.IGNORECASE),
                re.compile(r"file\s+copy.*?external", re.IGNORECASE),
                re.compile(r"suspicious\s+upload", re.IGNORECASE),
            ],
            "system_compromise": [
                re.compile(r"malware\s+detected", re.IGNORECASE),
                re.compile(r"rootkit", re.IGNORECASE),
                re.compile(r"backdoor", re.IGNORECASE),
                re.compile(r"command\s+and\s+control", re.IGNORECASE),
            ]
        }
        return patterns
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for log anomaly detection"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import IsolationForest
            
            # Initialize TF-IDF vectorizer for log text analysis
            self.log_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3),
                max_df=0.95,
                min_df=2
            )
            
            # Initialize isolation forest for anomaly detection
            self.log_anomaly_model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Try to load existing models
            model_path = Path("models/log_analysis")
            if model_path.exists():
                try:
                    self.log_vectorizer = joblib.load(model_path / "vectorizer.pkl")
                    self.log_anomaly_model = joblib.load(model_path / "anomaly_model.pkl")
                    logger.info("Loaded existing log analysis models")
                except Exception as e:
                    logger.warning(f"Failed to load existing models: {e}")
        
        except ImportError as e:
            logger.error(f"Failed to initialize ML models: {e}")
            self.log_vectorizer = None
            self.log_anomaly_model = None
    
    async def start_monitoring(self, log_sources: List[str]):
        """Start monitoring multiple log sources"""
        for log_source in log_sources:
            if log_source not in self.active_monitors:
                queue = Queue()
                self.log_queues[log_source] = queue
                
                monitor_thread = threading.Thread(
                    target=self._monitor_log_file,
                    args=(log_source, queue),
                    daemon=True
                )
                monitor_thread.start()
                self.active_monitors[log_source] = monitor_thread
                
                # Start async analysis task for this log source
                analysis_task = asyncio.create_task(
                    self._analyze_log_stream(log_source, queue)
                )
                self.active_monitors[f"{log_source}_analysis"] = analysis_task
        
        logger.info(f"Started monitoring {len(log_sources)} log sources")
    
    def _monitor_log_file(self, log_file: str, queue: Queue):
        """Monitor a log file for new entries"""
        try:
            # Use tail -f equivalent for real-time monitoring
            if not os.path.exists(log_file):
                logger.warning(f"Log file {log_file} does not exist")
                return
            
            with open(log_file, 'r') as f:
                # Seek to end of file
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        queue.put(line.strip())
                    else:
                        time.sleep(0.1)  # Brief pause if no new data
        
        except Exception as e:
            logger.error(f"Error monitoring log file {log_file}: {e}")
    
    async def _analyze_log_stream(self, log_source: str, queue: Queue):
        """Analyze log entries from queue for threats"""
        batch_size = 100
        batch = []
        
        while True:
            try:
                # Collect batch of log entries
                try:
                    while len(batch) < batch_size:
                        line = queue.get(timeout=1.0)
                        batch.append(line)
                except Empty:
                    if batch:
                        # Process partial batch if we have data
                        threats = self._analyze_log_batch(batch, log_source)
                        for threat in threats:
                            await self._handle_log_threat(threat)
                        batch = []
                    else:
                        # Brief pause to prevent tight loop
                        await asyncio.sleep(0.1)
                        continue
                
                # Analyze batch for threats
                threats = self._analyze_log_batch(batch, log_source)
                
                # Process detected threats
                for threat in threats:
                    self.recent_events.append(threat)
                    THREATS_DETECTED.labels(threat_type=threat['type']).inc()
                    
                    # Trigger defensive actions if needed
                    await self._handle_log_threat(threat)
                
                # Clear batch
                batch = []
                
            except Exception as e:
                logger.error(f"Error analyzing log stream {log_source}: {e}")
                await asyncio.sleep(1)  # Brief pause on error
                logger.error(f"Error analyzing log stream {log_source}: {e}")
                time.sleep(1)
    
    def _analyze_log_batch(self, log_lines: List[str], source: str) -> List[Dict[str, Any]]:
        """Analyze a batch of log lines for threats"""
        threats = []
        
        for line in log_lines:
            # Pattern-based detection
            for threat_type, patterns in self.log_patterns.items():
                for pattern in patterns:
                    if pattern.search(line):
                        threat = {
                            'type': threat_type,
                            'source': source,
                            'line': line,
                            'pattern': pattern.pattern,
                            'timestamp': datetime.utcnow(),
                            'severity': self._assess_threat_severity(threat_type, line)
                        }
                        threats.append(threat)
                        self.threat_counters[threat_type] += 1
                        break
            
            # ML-based anomaly detection
            if self.log_vectorizer and self.log_anomaly_model:
                anomaly_score = self._detect_log_anomaly(line)
                if anomaly_score < -0.5:  # Threshold for anomaly
                    threat = {
                        'type': 'ml_anomaly',
                        'source': source,
                        'line': line,
                        'anomaly_score': anomaly_score,
                        'timestamp': datetime.utcnow(),
                        'severity': ThreatLevel.MEDIUM
                    }
                    threats.append(threat)
        
        return threats
    
    def _detect_log_anomaly(self, log_line: str) -> float:
        """Use ML model to detect anomalous log entries"""
        try:
            # Vectorize log line
            vector = self.log_vectorizer.transform([log_line])
            
            # Get anomaly score
            score = self.log_anomaly_model.decision_function(vector)[0]
            return score
        
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
            return 0.0
    
    def _assess_threat_severity(self, threat_type: str, log_line: str) -> ThreatLevel:
        """Assess severity of detected threat"""
        severity_map = {
            "injection_attempts": ThreatLevel.HIGH,
            "authentication_anomalies": ThreatLevel.MEDIUM,
            "privilege_escalation": ThreatLevel.CRITICAL,
            "data_exfiltration": ThreatLevel.HIGH,
            "system_compromise": ThreatLevel.CRITICAL
        }
        
        base_severity = severity_map.get(threat_type, ThreatLevel.LOW)
        
        # Elevate severity based on specific indicators
        if any(keyword in log_line.lower() for keyword in ['root', 'admin', 'sudo']):
            if base_severity.value < ThreatLevel.HIGH.value:
                base_severity = ThreatLevel.HIGH
        
        if any(keyword in log_line.lower() for keyword in ['critical', 'emergency', 'exploit']):
            base_severity = ThreatLevel.CRITICAL
        
        return base_severity
    
    async def _handle_log_threat(self, threat: Dict[str, Any]):
        """Handle detected log threat with comprehensive defensive actions"""
        try:
            threat_type = threat.get('type', 'unknown')
            severity = threat.get('severity', ThreatLevel.MEDIUM)
            source_ip = threat.get('source_ip')
            log_line = threat.get('log_line', '')
            
            logger.warning(f"Log threat detected: {threat_type} from {source_ip or 'unknown'}")
            
            # Create security event for correlation
            security_event = SecurityEvent(
                event_type=f"log_threat_{threat_type}",
                threat_level=severity,
                details={
                    'threat_type': threat_type,
                    'source_ip': source_ip,
                    'log_line': log_line[:500],  # Truncate for storage
                    'detection_timestamp': datetime.utcnow().isoformat(),
                    'analyzer': 'log_analysis_engine'
                },
                timestamp=datetime.utcnow()
            )
            
            # Increment threat counter
            THREATS_DETECTED.labels(threat_type=threat_type).inc()
            
            # Determine response actions based on threat type and severity
            if threat_type == 'injection_attempts':
                await self._handle_injection_threat(threat, security_event)
            elif threat_type == 'authentication_anomalies':
                await self._handle_auth_threat(threat, security_event)
            elif threat_type == 'privilege_escalation':
                await self._handle_privilege_escalation(threat, security_event)
            elif threat_type == 'data_exfiltration':
                await self._handle_data_exfiltration(threat, security_event)
            elif threat_type == 'system_compromise':
                await self._handle_system_compromise(threat, security_event)
            else:
                # Generic threat handling
                await self._handle_generic_threat(threat, security_event)
            
            # Store recent threat for correlation
            self.recent_events.append({
                'type': 'log_threat',
                'threat_type': threat_type,
                'severity': severity.value if hasattr(severity, 'value') else str(severity),
                'source_ip': source_ip,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error handling log threat: {e}", exc_info=True)
    
    async def _handle_injection_threat(self, threat: Dict[str, Any], event: SecurityEvent):
        """Handle injection attempt threats"""
        source_ip = threat.get('source_ip')
        if source_ip:
            # Rate limit the IP
            await self._apply_rate_limiting(source_ip, "injection_attempt")
            
            # If repeated attempts, block temporarily
            if self.threat_counters[f"injection_{source_ip}"] > 3:
                await self._block_ip_temporarily(source_ip, "repeated_injection_attempts")
        
        # Alert security team
        await self._send_security_alert(f"Injection attempt detected: {threat.get('log_line', '')[:200]}")
    
    async def _handle_auth_threat(self, threat: Dict[str, Any], event: SecurityEvent):
        """Handle authentication anomaly threats"""
        source_ip = threat.get('source_ip')
        if source_ip:
            # Count failed attempts
            self.threat_counters[f"auth_fail_{source_ip}"] += 1
            
            # Rate limit after multiple failures
            if self.threat_counters[f"auth_fail_{source_ip}"] > 5:
                await self._apply_rate_limiting(source_ip, "brute_force_attempt")
            
            # Block after excessive failures
            if self.threat_counters[f"auth_fail_{source_ip}"] > 10:
                await self._block_ip_temporarily(source_ip, "brute_force_attack", duration=7200)
    
    async def _handle_privilege_escalation(self, threat: Dict[str, Any], event: SecurityEvent):
        """Handle privilege escalation threats"""
        # This is critical - immediate response required
        await self._send_security_alert(f"CRITICAL: Privilege escalation detected - {threat.get('log_line', '')[:200]}")
        
        # If we can identify the source VM/process, take immediate action
        if 'vm_id' in threat:
            await self._isolate_vm_immediately(threat['vm_id'])
        
        # Create forensic snapshot if possible
        await self._create_forensic_snapshot(threat)
    
    async def _handle_data_exfiltration(self, threat: Dict[str, Any], event: SecurityEvent):
        """Handle data exfiltration threats"""
        source_ip = threat.get('source_ip')
        if source_ip:
            # Immediate blocking for data exfiltration
            await self._block_ip_temporarily(source_ip, "data_exfiltration", duration=86400)
        
        # Monitor network traffic more closely
        await self._increase_monitoring_sensitivity()
        
        # Alert security team immediately
        await self._send_security_alert(f"URGENT: Data exfiltration detected - {threat.get('log_line', '')[:200]}")
    
    async def _handle_system_compromise(self, threat: Dict[str, Any], event: SecurityEvent):
        """Handle system compromise threats"""
        # Maximum severity response
        await self._send_security_alert(f"CRITICAL: System compromise detected - {threat.get('log_line', '')[:200]}")
        
        # Immediate isolation if VM identified
        if 'vm_id' in threat:
            await self._isolate_vm_immediately(threat['vm_id'])
        
        # Create emergency snapshot
        await self._create_forensic_snapshot(threat)
        
        # Initiate incident response
        await self._initiate_incident_response(threat)
    
    async def _handle_generic_threat(self, threat: Dict[str, Any], event: SecurityEvent):
        """Handle generic threats"""
        # Basic logging and monitoring
        await self._send_security_alert(f"Security threat detected: {threat.get('type', 'unknown')} - {threat.get('log_line', '')[:200]}")
        
        # Apply basic rate limiting if source IP available
        source_ip = threat.get('source_ip')
        if source_ip:
            await self._apply_rate_limiting(source_ip, "generic_threat")
    
    # Helper methods for threat handling
    async def _apply_rate_limiting(self, ip: str, reason: str):
        """Apply rate limiting to an IP address using iptables"""
        try:
            # Validate IP address format
            ipaddress.ip_address(ip)
            
            # Create iptables rule for rate limiting
            rate_limit_rule = [
                'iptables', '-A', 'INPUT', '-s', ip,
                '-m', 'limit', '--limit', '10/minute', '--limit-burst', '5',
                '-j', 'ACCEPT'
            ]
            
            # Create logging rule
            log_rule = [
                'iptables', '-A', 'INPUT', '-s', ip,
                '-j', 'LOG', '--log-prefix', f'RATE_LIMITED_{reason}:',
                '--log-level', '4'
            ]
            
            # Execute iptables commands
            for rule in [rate_limit_rule, log_rule]:
                process = await asyncio.create_subprocess_exec(
                    *rule,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Failed to apply iptables rule: {stderr.decode()}")
                    return False
            
            logger.info(f"Applied rate limiting to {ip} for {reason}")
            return True
            
        except ValueError:
            logger.error(f"Invalid IP address format: {ip}")
            return False
        except Exception as e:
            logger.error(f"Failed to apply rate limiting to {ip}: {e}")
            return False
    
    async def _block_ip_temporarily(self, ip: str, reason: str, duration: int = 3600):
        """Temporarily block an IP address using iptables with automatic unblock"""
        try:
            # Validate IP address format
            ipaddress.ip_address(ip)
            
            # Create blocking rule with comment
            block_rule = [
                'iptables', '-A', 'INPUT', '-s', ip, '-j', 'DROP',
                '-m', 'comment', '--comment', f'TEMP_BLOCK_{reason}_{int(time.time())}'
            ]
            
            # Execute blocking rule
            process = await asyncio.create_subprocess_exec(
                *block_rule,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Failed to block IP {ip}: {stderr.decode()}")
                return False
            
            logger.warning(f"Blocked IP {ip} for {duration}s due to {reason}")
            
            # Schedule automatic unblock
            asyncio.create_task(self._unblock_ip_after_delay(ip, duration))
            
            return True
            
        except ValueError:
            logger.error(f"Invalid IP address format: {ip}")
            return False
        except Exception as e:
            logger.error(f"Failed to block IP {ip}: {e}")
            return False
    
    async def _unblock_ip_after_delay(self, ip: str, delay: int):
        """Unblock IP after specified delay"""
        try:
            await asyncio.sleep(delay)
            
            # Remove all blocking rules for this IP
            list_process = await asyncio.create_subprocess_exec(
                'iptables', '-L', 'INPUT', '--line-numbers', '-n',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await list_process.communicate()
            
            if list_process.returncode == 0:
                lines = stdout.decode().split('\n')
                for line in lines:
                    if ip in line and 'DROP' in line:
                        # Extract line number
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            line_num = parts[0]
                            
                            # Delete the rule
                            delete_process = await asyncio.create_subprocess_exec(
                                'iptables', '-D', 'INPUT', line_num,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE
                            )
                            await delete_process.communicate()
            
            logger.info(f"Unblocked IP {ip} after {delay} seconds")
            
        except Exception as e:
            logger.error(f"Failed to unblock IP {ip}: {e}")
    
    async def _send_security_alert(self, message: str):
        """Send security alert to multiple monitoring systems"""
        try:
            # Send to system logger
            logger.critical(f"SECURITY ALERT: {message}")
            
            # Send to syslog with security facility
            try:
                syslog.openlog("somnus_security", syslog.LOG_PID, syslog.LOG_SECURITY)
                syslog.syslog(syslog.LOG_CRIT, f"ALERT: {message}")
                syslog.closelog()
            except Exception as e:
                logger.debug(f"Syslog error: {e}")
            
            # Write to security alert file
            try:
                alert_file = Path("/var/log/somnus_security_alerts.log")
                alert_file.parent.mkdir(exist_ok=True)
                
                alert_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "CRITICAL",
                    "message": message,
                    "source": "blue_team_log_analysis"
                }
                
                with open(alert_file, "a") as f:
                    f.write(f"{json.dumps(alert_entry)}\n")
                    
            except Exception as e:
                logger.debug(f"File logging error: {e}")
            
            # Send to metrics (if available)
            try:
                THREATS_DETECTED.labels(threat_type="security_alert").inc()
            except Exception as e:
                logger.debug(f"Metrics error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send security alert: {e}")
            return False
    
    async def _isolate_vm_immediately(self, vm_id: str):
        """Immediately isolate a VM by blocking all network access and suspending operations"""
        try:
            logger.critical(f"Isolating VM {vm_id} due to security threat")
            
            # Step 1: Block all network access using iptables
            network_isolation_rules = [
                ['iptables', '-I', 'FORWARD', '-i', f'vnet-{vm_id}', '-j', 'DROP'],
                ['iptables', '-I', 'FORWARD', '-o', f'vnet-{vm_id}', '-j', 'DROP'],
                ['iptables', '-I', 'INPUT', '-i', f'vnet-{vm_id}', '-j', 'DROP'],
                ['iptables', '-I', 'OUTPUT', '-o', f'vnet-{vm_id}', '-j', 'DROP']
            ]
            
            for rule in network_isolation_rules:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *rule,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await process.communicate()
                except Exception as rule_error:
                    logger.error(f"Failed to apply network isolation rule: {rule_error}")
            
            # Step 2: Suspend the VM if using libvirt/KVM
            try:
                suspend_cmd = ['virsh', 'suspend', vm_id]
                process = await asyncio.create_subprocess_exec(
                    *suspend_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"VM {vm_id} suspended successfully")
                else:
                    logger.warning(f"Failed to suspend VM {vm_id}: {stderr.decode()}")
            except Exception as suspend_error:
                logger.error(f"Error suspending VM {vm_id}: {suspend_error}")
            
            # Step 3: Mark VM as isolated in tracking system
            isolation_record = {
                'vm_id': vm_id,
                'isolation_timestamp': datetime.utcnow().isoformat(),
                'isolation_reason': 'security_threat',
                'isolation_type': 'full_network_and_compute',
                'recovery_possible': True
            }
            
            # Store isolation record
            isolation_file = Path(f"/var/log/somnus_vm_isolations.json")
            isolation_file.parent.mkdir(exist_ok=True)
            
            try:
                existing_isolations = []
                if isolation_file.exists():
                    with open(isolation_file, 'r') as f:
                        existing_isolations = json.load(f)
                
                existing_isolations.append(isolation_record)
                
                with open(isolation_file, 'w') as f:
                    json.dump(existing_isolations, f, indent=2)
                    
                logger.info(f"VM {vm_id} isolation record stored")
            except Exception as record_error:
                logger.error(f"Failed to store isolation record: {record_error}")
            
            # Step 4: Send alert to security team
            await self._send_security_alert(f"VM {vm_id} has been isolated due to security threat")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to isolate VM {vm_id}: {e}")
            return False
    
    async def _create_forensic_snapshot(self, threat: Dict[str, Any]):
        """Create forensic snapshot for investigation with detailed metadata"""
        try:
            vm_id = threat.get('vm_id')
            snapshot_id = f"forensic_{int(time.time())}_{os.urandom(4).hex()}"
            
            logger.info(f"Creating forensic snapshot {snapshot_id} for security investigation")
            
            # Step 1: Create memory dump if VM is running
            memory_dump_path = f"/var/forensics/memory_dumps/{snapshot_id}.dump"
            memory_dump_dir = Path(memory_dump_path).parent
            memory_dump_dir.mkdir(parents=True, exist_ok=True)
            
            if vm_id:
                try:
                    # Create memory dump using virsh
                    dump_cmd = ['virsh', 'dump', vm_id, memory_dump_path, '--memory-only']
                    process = await asyncio.create_subprocess_exec(
                        *dump_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        logger.info(f"Memory dump created: {memory_dump_path}")
                    else:
                        logger.warning(f"Memory dump failed: {stderr.decode()}")
                except Exception as dump_error:
                    logger.error(f"Error creating memory dump: {dump_error}")
            
            # Step 2: Create disk snapshot
            disk_snapshot_path = f"/var/forensics/disk_snapshots/{snapshot_id}.qcow2"
            disk_snapshot_dir = Path(disk_snapshot_path).parent
            disk_snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            if vm_id:
                try:
                    # Create disk snapshot using qemu-img
                    snapshot_cmd = ['qemu-img', 'create', '-f', 'qcow2', '-o', 
                                   f'backing_file=/var/lib/libvirt/images/{vm_id}.qcow2', 
                                   disk_snapshot_path]
                    process = await asyncio.create_subprocess_exec(
                        *snapshot_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        logger.info(f"Disk snapshot created: {disk_snapshot_path}")
                    else:
                        logger.warning(f"Disk snapshot failed: {stderr.decode()}")
                except Exception as snapshot_error:
                    logger.error(f"Error creating disk snapshot: {snapshot_error}")
            
            # Step 3: Collect system state information
            system_state = {
                'processes': [],
                'network_connections': [],
                'open_files': [],
                'system_info': {}
            }
            
            try:
                # Get process list
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                    try:
                        system_state['processes'].append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Get network connections
                for conn in psutil.net_connections():
                    system_state['network_connections'].append({
                        'local_addr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                        'remote_addr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                        'status': conn.status,
                        'pid': conn.pid
                    })
                
                # Get system info
                system_state['system_info'] = {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total,
                    'disk_usage': {path: psutil.disk_usage(path)._asdict() for path in ['/']},
                    'boot_time': psutil.boot_time(),
                    'users': [user._asdict() for user in psutil.users()]
                }
                
            except Exception as state_error:
                logger.error(f"Error collecting system state: {state_error}")
            
            # Step 4: Create comprehensive forensic metadata
            forensic_metadata = {
                'snapshot_id': snapshot_id,
                'creation_timestamp': datetime.utcnow().isoformat(),
                'threat_details': threat,
                'vm_id': vm_id,
                'memory_dump_path': memory_dump_path if vm_id else None,
                'disk_snapshot_path': disk_snapshot_path if vm_id else None,
                'system_state': system_state,
                'investigator': 'somnus_blue_team_automated',
                'chain_of_custody': [{
                    'timestamp': datetime.utcnow().isoformat(),
                    'action': 'forensic_snapshot_created',
                    'agent': 'blue_team_log_analysis_engine'
                }],
                'hash_verification': {}
            }
            
            # Calculate hashes for integrity verification
            for file_path in [memory_dump_path, disk_snapshot_path]:
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.sha256(f.read()).hexdigest()
                            forensic_metadata['hash_verification'][file_path] = {
                                'sha256': file_hash,
                                'timestamp': datetime.utcnow().isoformat()
                            }
                    except Exception as hash_error:
                        logger.error(f"Error calculating hash for {file_path}: {hash_error}")
            
            # Step 5: Store forensic metadata
            metadata_path = f"/var/forensics/metadata/{snapshot_id}.json"
            metadata_dir = Path(metadata_path).parent
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(forensic_metadata, f, indent=2, default=str)
                logger.info(f"Forensic metadata stored: {metadata_path}")
            except Exception as metadata_error:
                logger.error(f"Error storing forensic metadata: {metadata_error}")
            
            # Step 6: Send notification to security team
            await self._send_security_alert(f"Forensic snapshot {snapshot_id} created for security investigation")
            
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Failed to create forensic snapshot: {e}")
            return None
    
    async def _increase_monitoring_sensitivity(self):
        """Increase monitoring sensitivity during threats by adjusting detection thresholds"""
        try:
            logger.info("Increasing monitoring sensitivity due to active threats")
            
            # Step 1: Reduce detection thresholds
            sensitivity_config = {
                'log_analysis_threshold': 0.3,  # More sensitive anomaly detection
                'network_anomaly_threshold': 0.4,
                'resource_anomaly_threshold': 0.3,
                'failed_auth_threshold': 3,  # Lower threshold for failed auths
                'connection_rate_threshold': 25,  # Lower threshold for connections
                'cpu_threshold': 80,  # Lower CPU threshold
                'memory_threshold': 80,  # Lower memory threshold
                'monitoring_interval': 2  # More frequent monitoring (seconds)
            }
            
            # Step 2: Store current sensitivity settings
            sensitivity_file = Path("/var/log/somnus_sensitivity_config.json")
            sensitivity_file.parent.mkdir(exist_ok=True)
            
            try:
                with open(sensitivity_file, 'w') as f:
                    json.dump({
                        'increased_sensitivity': True,
                        'config': sensitivity_config,
                        'timestamp': datetime.utcnow().isoformat(),
                        'duration': 3600  # 1 hour of increased sensitivity
                    }, f, indent=2)
                logger.info("Sensitivity configuration stored")
            except Exception as config_error:
                logger.error(f"Error storing sensitivity config: {config_error}")
            
            # Step 3: Increase log monitoring frequency
            try:
                # Reduce log analysis batch size for faster processing
                if hasattr(self, 'log_analysis_batch_size'):
                    self.log_analysis_batch_size = 50  # Smaller batches
                
                # Increase monitoring threads if possible
                log_files = [
                    '/var/log/auth.log',
                    '/var/log/syslog',
                    '/var/log/apache2/access.log',
                    '/var/log/apache2/error.log',
                    '/var/log/somnus_security.log'
                ]
                
                for log_file in log_files:
                    if os.path.exists(log_file) and log_file not in getattr(self, 'active_monitors', {}):
                        await self.start_monitoring([log_file])
                        
            except Exception as log_error:
                logger.error(f"Error increasing log monitoring: {log_error}")
            
            # Step 4: Enable additional monitoring features
            try:
                # Monitor additional system calls
                auditd_rules = [
                    ['auditctl', '-w', '/etc/passwd', '-p', 'wa', '-k', 'passwd_changes'],
                    ['auditctl', '-w', '/etc/shadow', '-p', 'wa', '-k', 'shadow_changes'],
                    ['auditctl', '-w', '/etc/sudoers', '-p', 'wa', '-k', 'sudoers_changes'],
                    ['auditctl', '-a', 'exit,always', '-S', 'execve', '-k', 'exec_monitoring']
                ]
                
                for rule in auditd_rules:
                    try:
                        process = await asyncio.create_subprocess_exec(
                            *rule,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        await process.communicate()
                    except Exception as rule_error:
                        logger.debug(f"Audit rule error (may be normal): {rule_error}")
                        
            except Exception as audit_error:
                logger.error(f"Error configuring audit rules: {audit_error}")
            
            # Step 5: Schedule return to normal sensitivity
            asyncio.create_task(self._restore_normal_sensitivity_after_delay(3600))
            
            # Step 6: Update metrics
            try:
                SYSTEM_HEALTH.set(SYSTEM_HEALTH._value._value - 10)  # Reduce health during high sensitivity
            except Exception as metrics_error:
                logger.debug(f"Metrics update error: {metrics_error}")
            
            logger.info("Monitoring sensitivity increased successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to increase monitoring sensitivity: {e}")
            return False
    
    async def _initiate_incident_response(self, threat: Dict[str, Any]):
        """Initiate formal incident response procedures with comprehensive workflow"""
        try:
            incident_id = f"INC_{int(time.time())}_{os.urandom(4).hex()}"
            logger.critical(f"Initiating incident response {incident_id} for security threat")
            
            # Step 1: Create incident record
            incident_record = {
                'incident_id': incident_id,
                'creation_timestamp': datetime.utcnow().isoformat(),
                'threat_details': threat,
                'severity': threat.get('severity', 'HIGH'),
                'status': 'ACTIVE',
                'response_team': 'somnus_blue_team',
                'affected_systems': [],
                'containment_actions': [],
                'investigation_notes': [],
                'timeline': [{
                    'timestamp': datetime.utcnow().isoformat(),
                    'event': 'incident_created',
                    'details': 'Automated incident response initiated'
                }]
            }
            
            # Step 2: Identify affected systems
            vm_id = threat.get('vm_id')
            source_ip = threat.get('source_ip')
            
            if vm_id:
                incident_record['affected_systems'].append({
                    'type': 'virtual_machine',
                    'identifier': vm_id,
                    'impact': 'direct'
                })
            
            if source_ip:
                incident_record['affected_systems'].append({
                    'type': 'network_source',
                    'identifier': source_ip,
                    'impact': 'source'
                })
            
            # Step 3: Execute immediate containment actions
            containment_actions = []
            
            # Isolate affected VM
            if vm_id:
                isolation_success = await self._isolate_vm_immediately(vm_id)
                containment_actions.append({
                    'action': 'vm_isolation',
                    'target': vm_id,
                    'success': isolation_success,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Block source IP
            if source_ip:
                block_success = await self._block_ip_temporarily(source_ip, "incident_response", 86400)
                containment_actions.append({
                    'action': 'ip_block',
                    'target': source_ip,
                    'success': block_success,
                    'duration': 86400,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Create forensic snapshot
            snapshot_id = await self._create_forensic_snapshot(threat)
            if snapshot_id:
                containment_actions.append({
                    'action': 'forensic_snapshot',
                    'snapshot_id': snapshot_id,
                    'success': True,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            incident_record['containment_actions'] = containment_actions
            
            # Step 4: Collect additional evidence
            evidence_collection = {
                'network_logs': [],
                'system_logs': [],
                'process_dumps': [],
                'file_checksums': []
            }
            
            try:
                # Collect recent log entries
                log_files = ['/var/log/auth.log', '/var/log/syslog', '/var/log/apache2/access.log']
                for log_file in log_files:
                    if os.path.exists(log_file):
                        try:
                            # Get last 100 lines
                            with open(log_file, 'r') as f:
                                lines = f.readlines()[-100:]
                                evidence_collection['system_logs'].append({
                                    'file': log_file,
                                    'lines': len(lines),
                                    'sample': lines[:10]  # Store sample for analysis
                                })
                        except Exception as log_error:
                            logger.error(f"Error collecting logs from {log_file}: {log_error}")
                
                # Collect network connection information
                for conn in psutil.net_connections():
                    if conn.status == psutil.CONN_ESTABLISHED:
                        evidence_collection['network_logs'].append({
                            'local_addr': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                            'remote_addr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                            'pid': conn.pid,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                
            except Exception as evidence_error:
                logger.error(f"Error collecting evidence: {evidence_error}")
            
            incident_record['evidence'] = evidence_collection
            
            # Step 5: Generate incident report
            incident_report = {
                'executive_summary': self._generate_incident_summary(threat),
                'technical_details': incident_record,
                'recommendations': self._generate_incident_recommendations(threat),
                'next_steps': [
                    'Continue monitoring affected systems',
                    'Conduct detailed forensic analysis',
                    'Review security policies',
                    'Implement additional safeguards'
                ]
            }
            
            # Step 6: Store incident record
            incident_file = Path(f"/var/incidents/{incident_id}.json")
            incident_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(incident_file, 'w') as f:
                    json.dump(incident_report, f, indent=2, default=str)
                logger.info(f"Incident record stored: {incident_file}")
            except Exception as storage_error:
                logger.error(f"Error storing incident record: {storage_error}")
            
            # Step 7: Send notifications
            await self._send_security_alert(f"INCIDENT RESPONSE: {incident_id} - {threat.get('type', 'Unknown threat')} detected")
            
            # Step 8: Schedule follow-up tasks
            asyncio.create_task(self._schedule_incident_follow_up(incident_id, 3600))
            
            logger.critical(f"Incident response {incident_id} initiated successfully")
            return incident_id
            
        except Exception as e:
            logger.error(f"Failed to initiate incident response: {e}")
            return None
    
    def train_models(self, training_data: List[str]):
        """Train ML models on historical log data"""
        if not self.log_vectorizer or not self.log_anomaly_model:
            return
        
        try:
            # Fit vectorizer on training data
            vectors = self.log_vectorizer.fit_transform(training_data)
            
            # Train anomaly detection model
            self.log_anomaly_model.fit(vectors)
            
            # Save models
            model_path = Path("models/log_analysis")
            model_path.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.log_vectorizer, model_path / "vectorizer.pkl")
            joblib.dump(self.log_anomaly_model, model_path / "anomaly_model.pkl")
            
            logger.info(f"Trained log analysis models on {len(training_data)} samples")
        
        except Exception as e:
            logger.error(f"Error training models: {e}")


class NetworkTrafficMonitor:
    """Real-time network traffic analysis and threat detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.traffic_data = deque(maxlen=50000)
        self.connection_stats = defaultdict(lambda: {'count': 0, 'bytes': 0, 'last_seen': datetime.utcnow()})
        self.blocked_ips: Set[str] = set()
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # ML model for network anomaly detection
        self.network_scaler = StandardScaler()
        self.network_anomaly_model = IsolationForest(contamination=0.1, random_state=42)
        self.feature_columns = [
            'bytes_per_second', 'packets_per_second', 'connection_duration',
            'unique_ports', 'failed_connections', 'geographic_distance'
        ]
        
        # Initialize models
        self._initialize_network_models()
    
    def _initialize_network_models(self):
        """Initialize network traffic analysis models"""
        try:
            model_path = Path("models/network_analysis")
            if model_path.exists():
                self.network_scaler = joblib.load(model_path / "scaler.pkl")
                self.network_anomaly_model = joblib.load(model_path / "network_anomaly_model.pkl")
                logger.info("Loaded existing network analysis models")
        except Exception as e:
            logger.warning(f"Failed to load network models: {e}")
    
    async def start_monitoring(self):
        """Start network traffic monitoring"""
        # Start packet capture thread
        capture_thread = threading.Thread(target=self._capture_packets, daemon=True)
        capture_thread.start()
        
        # Start connection monitoring thread
        connection_thread = threading.Thread(target=self._monitor_connections, daemon=True)
        connection_thread.start()
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self._analyze_traffic, daemon=True)
        analysis_thread.start()
        
        logger.info("Started network traffic monitoring")
    
    def _capture_packets(self):
        """Capture network packets for analysis"""
        try:
            # Use netstat/ss for connection monitoring (more reliable than raw packet capture)
            while True:
                try:
                    # Get network connections
                    connections = psutil.net_connections(kind='inet')
                    
                    for conn in connections:
                        if conn.status == psutil.CONN_ESTABLISHED:
                            conn_data = {
                                'local_addr': conn.laddr.ip if conn.laddr else None,
                                'local_port': conn.laddr.port if conn.laddr else None,
                                'remote_addr': conn.raddr.ip if conn.raddr else None,
                                'remote_port': conn.raddr.port if conn.raddr else None,
                                'pid': conn.pid,
                                'timestamp': datetime.utcnow(),
                                'status': conn.status
                            }
                            self.traffic_data.append(conn_data)
                    
                    time.sleep(1)  # Collect data every second
                
                except Exception as e:
                    logger.error(f"Error capturing packets: {e}")
                    time.sleep(5)
        
        except Exception as e:
            logger.error(f"Fatal error in packet capture: {e}")
    
    async def _monitor_connections(self):
        """Monitor network connections for anomalies"""
        while True:
            try:
                # Analyze recent traffic data
                if len(self.traffic_data) > 100:
                    recent_data = list(self.traffic_data)[-100:]
                    
                    # Update connection statistics
                    for conn in recent_data:
                        if conn['remote_addr']:
                            key = f"{conn['remote_addr']}:{conn['remote_port']}"
                            self.connection_stats[key]['count'] += 1
                            self.connection_stats[key]['last_seen'] = conn['timestamp']
                    
                    # Detect anomalies
                    anomalies = self._detect_network_anomalies(recent_data)
                    
                    for anomaly in anomalies:
                        ANOMALIES_DETECTED.labels(source='network').inc()
                        await self._handle_network_anomaly(anomaly)
                
                await asyncio.sleep(5)  # Analysis every 5 seconds
            
            except Exception as e:
                logger.error(f"Error monitoring connections: {e}")
                await asyncio.sleep(10)
    
    def _analyze_traffic(self):
        """Analyze traffic patterns for threats"""
        while True:
            try:
                if len(self.traffic_data) > 1000:
                    # Prepare data for ML analysis
                    features = self._extract_network_features()
                    
                    if len(features) > 0:
                        # Detect anomalies using ML
                        anomaly_scores = self._detect_ml_network_anomalies(features)
                        
                        # Process anomalies
                        for i, score in enumerate(anomaly_scores):
                            if score == -1:  # Anomaly detected
                                anomaly_data = features.iloc[i]
                                await self._handle_ml_network_anomaly(anomaly_data)
                
                time.sleep(30)  # ML analysis every 30 seconds
            
            except Exception as e:
                logger.error(f"Error in traffic analysis: {e}")
                time.sleep(30)
    
    def _extract_network_features(self) -> pd.DataFrame:
        """Extract features from network traffic for ML analysis"""
        try:
            # Convert recent traffic data to features
            recent_data = list(self.traffic_data)[-1000:]
            
            # Group by source IP for feature extraction
            ip_stats = defaultdict(lambda: {
                'bytes_per_second': 0,
                'packets_per_second': 0,
                'connection_duration': 0,
                'unique_ports': set(),
                'failed_connections': 0,
                'geographic_distance': 0  # Placeholder for GeoIP data
            })
            
            # Calculate features per IP
            for conn in recent_data:
                if conn['remote_addr']:
                    ip = conn['remote_addr']
                    ip_stats[ip]['packets_per_second'] += 1
                    ip_stats[ip]['unique_ports'].add(conn['remote_port'])
                    
                    # Simulate geographic distance calculation
                    if self._is_suspicious_ip(ip):
                        ip_stats[ip]['geographic_distance'] = 10000  # Far distance
                    else:
                        ip_stats[ip]['geographic_distance'] = 1000   # Normal distance
            
            # Convert to DataFrame
            feature_data = []
            for ip, stats in ip_stats.items():
                feature_data.append({
                    'ip': ip,
                    'bytes_per_second': stats['bytes_per_second'],
                    'packets_per_second': stats['packets_per_second'],
                    'connection_duration': stats['connection_duration'],
                    'unique_ports': len(stats['unique_ports']),
                    'failed_connections': stats['failed_connections'],
                    'geographic_distance': stats['geographic_distance']
                })
            
            return pd.DataFrame(feature_data)
        
        except Exception as e:
            logger.error(f"Error extracting network features: {e}")
            return pd.DataFrame()
    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is suspicious based on known indicators"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            # Check for private/local IPs (generally safe)
            if ip_obj.is_private or ip_obj.is_loopback:
                return False
            
            # Check against threat intelligence feeds (placeholder)
            # In production, this would check against real threat feeds
            suspicious_ranges = [
                "185.220.100.0/24",  # Known Tor exit nodes
                "198.98.51.0/24",    # Known malicious range
            ]
            
            for range_str in suspicious_ranges:
                if ip_obj in ipaddress.ip_network(range_str):
                    return True
            
            return False
        
        except Exception:
            return False
    
    def _detect_network_anomalies(self, traffic_data: List[Dict]) -> List[Dict]:
        """Detect network anomalies using rule-based detection"""
        anomalies = []
        
        # Connection frequency analysis
        ip_connections = defaultdict(int)
        port_connections = defaultdict(int)
        
        for conn in traffic_data:
            if conn['remote_addr']:
                ip_connections[conn['remote_addr']] += 1
                port_connections[conn['remote_port']] += 1
        
        # Detect high connection frequency (potential DDoS/scanning)
        for ip, count in ip_connections.items():
            if count > 50:  # Threshold for suspicious activity
                anomalies.append({
                    'type': 'high_connection_frequency',
                    'source_ip': ip,
                    'connection_count': count,
                    'severity': ThreatLevel.HIGH if count > 100 else ThreatLevel.MEDIUM,
                    'timestamp': datetime.utcnow()
                })
        
        # Detect port scanning
        for ip, ports in [(ip, [c['remote_port'] for c in traffic_data if c['remote_addr'] == ip]) 
                         for ip in ip_connections.keys()]:
            unique_ports = len(set(ports))
            if unique_ports > 20:  # Scanning multiple ports
                anomalies.append({
                    'type': 'port_scanning',
                    'source_ip': ip,
                    'scanned_ports': unique_ports,
                    'severity': ThreatLevel.HIGH,
                    'timestamp': datetime.utcnow()
                })
        
        return anomalies
    
    def _detect_ml_network_anomalies(self, features: pd.DataFrame) -> List[int]:
        """Detect network anomalies using machine learning"""
        try:
            if len(features) == 0:
                return []
            
            # Scale features
            feature_matrix = features[self.feature_columns].fillna(0)
            scaled_features = self.network_scaler.fit_transform(feature_matrix)
            
            # Predict anomalies
            predictions = self.network_anomaly_model.fit_predict(scaled_features)
            
            return predictions.tolist()
        
        except Exception as e:
            logger.error(f"Error in ML network anomaly detection: {e}")
            return []
    
    async def _handle_network_anomaly(self, anomaly: Dict[str, Any]):
        """Handle detected network anomaly"""
        logger.warning(f"Network anomaly detected: {anomaly['type']} from {anomaly.get('source_ip', 'unknown')}")
        
        # Implement defensive actions based on anomaly type
        if anomaly['type'] == 'high_connection_frequency':
            await self._rate_limit_ip(anomaly['source_ip'])
        elif anomaly['type'] == 'port_scanning':
            await self._block_ip_temporarily(anomaly['source_ip'])
    
    async def _handle_ml_network_anomaly(self, anomaly_data: pd.Series):
        """Handle ML-detected network anomaly"""
        logger.warning(f"ML network anomaly detected for IP: {anomaly_data.get('ip', 'unknown')}")
        
        # Implement ML-based defensive responses
        if anomaly_data.get('packets_per_second', 0) > 100:
            await self._rate_limit_ip(anomaly_data['ip'])
    
    async def _rate_limit_ip(self, ip: str):
        """Apply rate limiting to suspicious IP"""
        # Implementation would use iptables or similar
        logger.info(f"Applying rate limiting to IP: {ip}")
        
        # Add to rate limiter
        self.rate_limiters[ip].append(datetime.utcnow())
    
    async def _block_ip_temporarily(self, ip: str, duration: int = 3600):
        """Temporarily block suspicious IP"""
        logger.info(f"Temporarily blocking IP: {ip} for {duration} seconds")
        
        self.blocked_ips.add(ip)
        
        # Schedule unblock
        asyncio.create_task(self._unblock_ip_after_delay(ip, duration))
    
    async def _unblock_ip_after_delay(self, ip: str, delay: int):
        """Unblock IP after specified delay"""
        await asyncio.sleep(delay)
        self.blocked_ips.discard(ip)
        logger.info(f"Unblocked IP: {ip}")


class ResourceMonitor:
    """Monitor system resources for anomalies and security threats"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_history = deque(maxlen=1000)
        self.process_baselines = {}
        self.memory_threshold = config.get('memory_threshold', 90)
        self.cpu_threshold = config.get('cpu_threshold', 95)
        self.disk_threshold = config.get('disk_threshold', 95)
        
        # ML model for resource anomaly detection
        self.resource_scaler = StandardScaler()
        self.resource_anomaly_model = IsolationForest(contamination=0.1, random_state=42)
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        
        analysis_thread = threading.Thread(target=self._analyze_resource_usage, daemon=True)
        analysis_thread.start()
        
        logger.info("Started resource monitoring")
    
    def _monitor_resources(self):
        """Monitor system resources continuously"""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Collect process information
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                resource_data = {
                    'timestamp': datetime.utcnow(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'disk_percent': disk.percent,
                    'processes': processes,
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                }
                
                self.resource_history.append(resource_data)
                
                # Update system health gauge
                health_score = self._calculate_health_score(resource_data)
                SYSTEM_HEALTH.set(health_score)
                
                time.sleep(5)  # Collect every 5 seconds
            
            except Exception as e:
                logger.error(f"Error monitoring resources: {e}")
                time.sleep(10)
    
    def _analyze_resource_usage(self):
        """Analyze resource usage for anomalies"""
        while True:
            try:
                if len(self.resource_history) > 10:
                    recent_data = list(self.resource_history)[-10:]
                    
                    # Check for resource exhaustion
                    for data in recent_data:
                        await self._check_resource_thresholds(data)
                    
                    # Check for process anomalies
                    await self._check_process_anomalies(recent_data)
                    
                    # ML-based anomaly detection
                    if len(self.resource_history) > 100:
                        await self._detect_ml_resource_anomalies()
                
                time.sleep(30)  # Analysis every 30 seconds
            
            except Exception as e:
                logger.error(f"Error analyzing resource usage: {e}")
                time.sleep(30)
    
    async def _check_resource_thresholds(self, data: Dict[str, Any]):
        """Check if resources exceed critical thresholds"""
        alerts = []
        
        if data['cpu_percent'] > self.cpu_threshold:
            alerts.append({
                'type': 'cpu_exhaustion',
                'value': data['cpu_percent'],
                'threshold': self.cpu_threshold,
                'severity': ThreatLevel.HIGH
            })
        
        if data['memory_percent'] > self.memory_threshold:
            alerts.append({
                'type': 'memory_exhaustion',
                'value': data['memory_percent'],
                'threshold': self.memory_threshold,
                'severity': ThreatLevel.HIGH
            })
        
        if data['disk_percent'] > self.disk_threshold:
            alerts.append({
                'type': 'disk_exhaustion',
                'value': data['disk_percent'],
                'threshold': self.disk_threshold,
                'severity': ThreatLevel.CRITICAL
            })
        
        for alert in alerts:
            logger.warning(f"Resource threshold exceeded: {alert}")
            ANOMALIES_DETECTED.labels(source='resource').inc()
    
    async def _check_process_anomalies(self, recent_data: List[Dict[str, Any]]):
        """Check for process-level anomalies"""
        # Aggregate process data
        process_stats = defaultdict(lambda: {'cpu_total': 0, 'memory_total': 0, 'count': 0})
        
        for data in recent_data:
            for proc in data['processes']:
                name = proc['name']
                process_stats[name]['cpu_total'] += proc.get('cpu_percent', 0) or 0
                process_stats[name]['memory_total'] += proc.get('memory_percent', 0) or 0
                process_stats[name]['count'] += 1
        
        # Check for anomalous processes
        for proc_name, stats in process_stats.items():
            avg_cpu = stats['cpu_total'] / stats['count'] if stats['count'] > 0 else 0
            avg_memory = stats['memory_total'] / stats['count'] if stats['count'] > 0 else 0
            
            # Check against baselines
            if proc_name in self.process_baselines:
                baseline = self.process_baselines[proc_name]
                cpu_deviation = abs(avg_cpu - baseline['cpu']) / (baseline['cpu'] + 1)
                memory_deviation = abs(avg_memory - baseline['memory']) / (baseline['memory'] + 1)
                
                if cpu_deviation > 2.0 or memory_deviation > 2.0:  # 200% deviation
                    logger.warning(f"Process anomaly detected: {proc_name} - CPU: {avg_cpu}%, Memory: {avg_memory}%")
                    ANOMALIES_DETECTED.labels(source='process').inc()
            else:
                # Establish baseline
                self.process_baselines[proc_name] = {'cpu': avg_cpu, 'memory': avg_memory}
    
    async def _detect_ml_resource_anomalies(self):
        """Use ML to detect resource usage anomalies"""
        try:
            # Prepare feature matrix
            features = []
            for data in list(self.resource_history)[-100:]:
                features.append([
                    data['cpu_percent'],
                    data['memory_percent'],
                    data['disk_percent'],
                    data['load_average'][0],
                    len(data['processes'])
                ])
            
            if len(features) > 10:
                # Scale features
                scaled_features = self.resource_scaler.fit_transform(features)
                
                # Detect anomalies
                anomaly_scores = self.resource_anomaly_model.fit_predict(scaled_features)
                
                # Process anomalies
                for i, score in enumerate(anomaly_scores):
                    if score == -1:  # Anomaly detected
                        logger.warning(f"ML resource anomaly detected at index {i}")
                        ANOMALIES_DETECTED.labels(source='ml_resource').inc()
        
        except Exception as e:
            logger.error(f"Error in ML resource anomaly detection: {e}")
    
    def _calculate_health_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)"""
        # Weight different metrics
        cpu_score = max(0, 100 - data['cpu_percent'])
        memory_score = max(0, 100 - data['memory_percent'])
        disk_score = max(0, 100 - data['disk_percent'])
        
        # Overall health score
        health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
        
        return health_score


class AutomatedResponseOrchestrator:
    """Orchestrates automated defensive responses to threats"""
    
    def __init__(self, vm_supervisor: VMSupervisor, security_framework: SomnusSecurityFramework):
        self.vm_supervisor = vm_supervisor
        self.security_framework = security_framework
        self.action_history: List[DefensiveAction] = []
        self.active_quarantines: Dict[str, datetime] = {}
        self.response_config = self._load_response_config()
        
        # Action cooldowns to prevent response loops
        self.action_cooldowns: Dict[str, datetime] = defaultdict(lambda: datetime.min)
    
    def _load_response_config(self) -> Dict[str, Any]:
        """Load automated response configuration"""
        default_config = {
            'response_thresholds': {
                ThreatLevel.LOW: [ResponseAction.ALERT_ONLY],
                ThreatLevel.MEDIUM: [ResponseAction.ALERT_ONLY, ResponseAction.RATE_LIMIT],
                ThreatLevel.HIGH: [ResponseAction.RATE_LIMIT, ResponseAction.QUARANTINE_PROCESS],
                ThreatLevel.CRITICAL: [ResponseAction.ISOLATE_VM, ResponseAction.CREATE_SNAPSHOT]
            },
            'cooldown_periods': {
                ResponseAction.RATE_LIMIT: 300,      # 5 minutes
                ResponseAction.QUARANTINE_PROCESS: 600,  # 10 minutes
                ResponseAction.ISOLATE_VM: 1800,     # 30 minutes
                ResponseAction.RESTART_SERVICE: 900   # 15 minutes
            },
            'auto_response_enabled': True,
            'require_confirmation': {
                ResponseAction.ISOLATE_VM: True,
                ResponseAction.EMERGENCY_SHUTDOWN: True
            }
        }
        
        # Try to load from configuration file
        config_path = Path("response_config.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load response config: {e}")
        
        return default_config
    
    async def handle_threat(self, threat_event: SecurityEvent, context: Optional[Dict[str, Any]] = None) -> List[DefensiveAction]:
        """Handle a detected threat with appropriate response"""
        if not self.response_config['auto_response_enabled']:
            return []
        
        actions_taken = []
        
        # Determine appropriate responses based on threat level
        threat_level = threat_event.threat_level
        possible_actions = self.response_config['response_thresholds'].get(threat_level, [])
        
        for action_type in possible_actions:
            # Check cooldown period
            cooldown_key = f"{action_type.value}_{threat_event.vm_id}"
            if self._is_action_on_cooldown(action_type, cooldown_key):
                continue
            
            # Check if action requires confirmation
            if action_type in self.response_config.get('require_confirmation', {}):
                # In production, this would integrate with user notification system
                logger.warning(f"Action {action_type.value} requires manual confirmation")
                continue
            
            # Execute defensive action
            defensive_action = await self._execute_defensive_action(
                action_type, threat_event, context
            )
            
            if defensive_action.success:
                actions_taken.append(defensive_action)
                self.action_history.append(defensive_action)
                
                # Set cooldown
                self.action_cooldowns[cooldown_key] = datetime.utcnow()
                
                # Update metrics
                with RESPONSE_TIME.labels(action_type=action_type.value).time():
                    pass  # Time already recorded in _execute_defensive_action
        
        return actions_taken
    
    def _is_action_on_cooldown(self, action_type: ResponseAction, cooldown_key: str) -> bool:
        """Check if action is on cooldown"""
        cooldown_period = self.response_config['cooldown_periods'].get(action_type, 0)
        last_action = self.action_cooldowns.get(cooldown_key, datetime.min)
        
        return (datetime.utcnow() - last_action).total_seconds() < cooldown_period
    
    async def _execute_defensive_action(self, action_type: ResponseAction, 
                                      threat_event: SecurityEvent, 
                                      context: Optional[Dict[str, Any]] = None) -> DefensiveAction:
        """Execute a specific defensive action"""
        start_time = time.time()
        
        # Handle None context
        if context is None:
            context = {}
        
        defensive_action = DefensiveAction(
            action_type=action_type,
            trigger_event=threat_event.event_id,
            target_vm_id=threat_event.vm_id
        )
        
        try:
            if action_type == ResponseAction.RATE_LIMIT:
                success = await self._apply_rate_limiting(threat_event, context)
            elif action_type == ResponseAction.QUARANTINE_PROCESS:
                success = await self._quarantine_process(threat_event, context)
            elif action_type == ResponseAction.ISOLATE_VM:
                success = await self._isolate_vm(threat_event, context)
            elif action_type == ResponseAction.RESTART_SERVICE:
                success = await self._restart_service(threat_event, context)
            elif action_type == ResponseAction.BLOCK_IP:
                success = await self._block_ip(threat_event, context)
            elif action_type == ResponseAction.CREATE_SNAPSHOT:
                success = await self._create_emergency_snapshot(threat_event, context)
            elif action_type == ResponseAction.REVOKE_PERMISSIONS:
                success = await self._revoke_permissions(threat_event, context)
            elif action_type == ResponseAction.EMERGENCY_SHUTDOWN:
                success = await self._emergency_shutdown(threat_event, context)
            else:
                # Default to alert only
                success = await self._send_alert(threat_event, context)
            
            defensive_action.success = success
        
        except Exception as e:
            logger.error(f"Error executing defensive action {action_type.value}: {e}")
            defensive_action.success = False
            defensive_action.error_message = str(e)
        
        defensive_action.execution_time = time.time() - start_time
        
        return defensive_action
    
    async def _apply_rate_limiting(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Apply rate limiting to threat source"""
        try:
            # Extract source information
            source_ip = threat_event.details.get('source_ip')
            if not source_ip:
                return False
            
            # Apply rate limiting (would integrate with firewall/iptables)
            logger.info(f"Applying rate limiting to {source_ip}")
            
            # In production, this would execute actual iptables commands
            # subprocess.run(['iptables', '-A', 'INPUT', '-s', source_ip, '-m', 'limit', '--limit', '10/min', '-j', 'ACCEPT'])
            
            return True
        
        except Exception as e:
            logger.error(f"Error applying rate limiting: {e}")
            return False
    
    async def _quarantine_process(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Quarantine suspicious process"""
        try:
            process_id = threat_event.details.get('process_id')
            if not process_id:
                return False
            
            # Suspend process
            try:
                process = psutil.Process(process_id)
                process.suspend()
                
                # Track quarantined process
                self.active_quarantines[str(process_id)] = datetime.utcnow()
                
                logger.info(f"Quarantined process {process_id}")
                return True
            
            except psutil.NoSuchProcess:
                logger.warning(f"Process {process_id} not found for quarantine")
                return False
        
        except Exception as e:
            logger.error(f"Error quarantining process: {e}")
            return False
    
    async def _isolate_vm(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Isolate VM from network"""
        try:
            vm_id = threat_event.vm_id
            if not vm_id:
                return False
            
            # Get VM instance
            vm_instance = await self.vm_supervisor.get_vm(vm_id)
            if not vm_instance:
                return False
            
            # Implement network isolation (would integrate with VM management)
            logger.critical(f"Isolating VM {vm_id} from network")
            
            # In production, this would modify VM network configuration
            # or use hypervisor commands to isolate the VM
            
            return True
        
        except Exception as e:
            logger.error(f"Error isolating VM: {e}")
            return False
    
    async def _restart_service(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Restart compromised service"""
        try:
            service_name = threat_event.details.get('service_name', 'somnus-agent')
            
            # Restart service using systemctl
            result = subprocess.run(['systemctl', 'restart', service_name], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully restarted service {service_name}")
                return True
            else:
                logger.error(f"Failed to restart service {service_name}: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error restarting service: {e}")
            return False
    
    async def _block_ip(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Block malicious IP address"""
        try:
            source_ip = threat_event.details.get('source_ip')
            if not source_ip:
                return False
            
            # Block IP using iptables
            result = subprocess.run(['iptables', '-A', 'INPUT', '-s', source_ip, '-j', 'DROP'],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully blocked IP {source_ip}")
                return True
            else:
                logger.error(f"Failed to block IP {source_ip}: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
            return False
    
    async def _create_emergency_snapshot(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Create emergency VM snapshot for forensics"""
        try:
            vm_id = threat_event.vm_id
            if not vm_id:
                return False
            
            # Create snapshot via VM supervisor
            snapshot_name = f"emergency_{threat_event.event_id}_{int(time.time())}"
            
            # In production, this would integrate with VM management
            logger.info(f"Creating emergency snapshot {snapshot_name} for VM {vm_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating emergency snapshot: {e}")
            return False
    
    async def _revoke_permissions(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Revoke permissions for compromised user/service"""
        try:
            user_id = threat_event.details.get('user_id')
            if not user_id:
                return False
            
            logger.info(f"Revoking permissions for user {user_id}")
            
            # In production, this would integrate with authentication system
            # to revoke API keys, sessions, etc.
            
            return True
        
        except Exception as e:
            logger.error(f"Error revoking permissions: {e}")
            return False
    
    async def _emergency_shutdown(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Emergency shutdown of system"""
        try:
            logger.critical("EMERGENCY SHUTDOWN INITIATED")
            
            # This is a last resort action
            # In production, would gracefully shut down services
            # subprocess.run(['shutdown', '-h', '+1'])  # Shutdown in 1 minute
            
            return True
        
        except Exception as e:
            logger.error(f"Error in emergency shutdown: {e}")
            return False
    
    async def _send_alert(self, threat_event: SecurityEvent, context: Dict[str, Any]) -> bool:
        """Send alert notification"""
        try:
            logger.warning(f"SECURITY ALERT: {threat_event.event_type} - {threat_event.details}")
            
            # In production, this would integrate with notification systems
            # (email, Slack, PagerDuty, etc.)
            
            return True
        
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False


class PolicyEnforcementEngine:
    """Dynamic security policy enforcement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_policies: Dict[str, Dict[str, Any]] = {}
        self.policy_violations: List[Dict[str, Any]] = []
        self.enforcement_rules = self._load_enforcement_rules()
    
    def _load_enforcement_rules(self) -> Dict[str, Any]:
        """Load policy enforcement rules"""
        default_rules = {
            'network_policies': {
                'block_tor_exits': True,
                'rate_limit_per_ip': 100,  # requests per minute
                'allowed_ports': [22, 80, 443, 8000, 9901],
                'geo_blocking_enabled': False
            },
            'process_policies': {
                'max_cpu_per_process': 80,
                'max_memory_per_process': 70,
                'forbidden_processes': ['bitcoin-miner', 'cryptominer'],
                'required_signatures': True
            },
            'file_policies': {
                'protected_directories': ['/etc', '/root', '/var/lib/somnus'],
                'scan_uploads': True,
                'quarantine_malware': True,
                'backup_before_modify': True
            },
            'api_policies': {
                'require_authentication': True,
                'max_request_size': 10485760,  # 10MB
                'rate_limit_per_user': 1000,   # per hour
                'log_all_requests': True
            }
        }
        
        # Load from configuration file if available
        config_path = Path("policy_config.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = yaml.safe_load(f)
                    default_rules.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load policy config: {e}")
        
        return default_rules
    
    async def enforce_network_policies(self, connection_data: Dict[str, Any]) -> bool:
        """Enforce network security policies"""
        try:
            policies = self.enforcement_rules['network_policies']
            
            # Check allowed ports
            port = connection_data.get('port')
            if port and port not in policies['allowed_ports']:
                await self._record_policy_violation(
                    'network_policy_violation',
                    f"Connection to unauthorized port {port}",
                    connection_data
                )
                return False
            
            # Check IP reputation
            ip = connection_data.get('ip')
            if ip and policies.get('block_tor_exits') and await self._is_tor_exit(ip):
                await self._record_policy_violation(
                    'tor_exit_blocked',
                    f"Connection from Tor exit node {ip}",
                    connection_data
                )
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error enforcing network policies: {e}")
            return True  # Fail open for availability
    
    async def enforce_process_policies(self, process_data: Dict[str, Any]) -> bool:
        """Enforce process security policies"""
        try:
            policies = self.enforcement_rules['process_policies']
            
            # Check forbidden processes
            process_name = process_data.get('name', '').lower()
            for forbidden in policies['forbidden_processes']:
                if forbidden in process_name:
                    await self._record_policy_violation(
                        'forbidden_process',
                        f"Forbidden process detected: {process_name}",
                        process_data
                    )
                    
                    # Terminate forbidden process
                    await self._terminate_process(process_data.get('pid'))
                    return False
            
            # Check resource limits
            cpu_percent = process_data.get('cpu_percent', 0)
            if cpu_percent > policies['max_cpu_per_process']:
                await self._record_policy_violation(
                    'cpu_limit_exceeded',
                    f"Process {process_name} exceeds CPU limit: {cpu_percent}%",
                    process_data
                )
                return False
            
            memory_percent = process_data.get('memory_percent', 0)
            if memory_percent > policies['max_memory_per_process']:
                await self._record_policy_violation(
                    'memory_limit_exceeded',
                    f"Process {process_name} exceeds memory limit: {memory_percent}%",
                    process_data
                )
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error enforcing process policies: {e}")
            return True
    
    async def enforce_file_policies(self, file_operation: Dict[str, Any]) -> bool:
        """Enforce file system security policies"""
        try:
            policies = self.enforcement_rules['file_policies']
            
            # Check protected directories
            file_path = file_operation.get('path', '')
            operation = file_operation.get('operation', '')
            
            for protected_dir in policies['protected_directories']:
                if file_path.startswith(protected_dir) and operation in ['write', 'delete']:
                    await self._record_policy_violation(
                        'protected_directory_access',
                        f"Attempted {operation} in protected directory: {file_path}",
                        file_operation
                    )
                    return False
            
            # Backup before modification if required
            if policies.get('backup_before_modify') and operation == 'write':
                await self._create_file_backup(file_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Error enforcing file policies: {e}")
            return True
    
    async def enforce_api_policies(self, api_request: Dict[str, Any]) -> bool:
        """Enforce API security policies"""
        try:
            policies = self.enforcement_rules['api_policies']
            
            # Check authentication
            if policies['require_authentication']:
                if not api_request.get('authenticated'):
                    await self._record_policy_violation(
                        'unauthenticated_request',
                        "API request without authentication",
                        api_request
                    )
                    return False
            
            # Check request size
            content_length = api_request.get('content_length', 0)
            if content_length > policies['max_request_size']:
                await self._record_policy_violation(
                    'oversized_request',
                    f"Request exceeds size limit: {content_length} bytes",
                    api_request
                )
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error enforcing API policies: {e}")
            return True
    
    async def _record_policy_violation(self, violation_type: str, description: str, data: Dict[str, Any]):
        """Record policy violation"""
        violation = {
            'type': violation_type,
            'description': description,
            'timestamp': datetime.utcnow(),
            'data': data
        }
        
        self.policy_violations.append(violation)
        logger.warning(f"Policy violation: {description}")
        
        # Trigger security event
        event = SecurityEvent(
            event_type=f"policy_violation_{violation_type}",
            threat_level=ThreatLevel.MEDIUM,
            details=violation
        )
        
        # Would integrate with main security framework here
    
    async def _is_tor_exit(self, ip: str) -> bool:
        """Check if IP is a Tor exit node"""
        # In production, this would check against Tor exit node lists
        # For now, simulate the check
        return False
    
    async def _terminate_process(self, pid: int) -> bool:
        """Terminate a process"""
        try:
            if pid:
                process = psutil.Process(pid)
                process.terminate()
                logger.info(f"Terminated process {pid}")
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Could not terminate process {pid}: {e}")
        
        return False
    
    async def _create_file_backup(self, file_path: str):
        """Create backup of file before modification"""
        try:
            if os.path.exists(file_path):
                backup_path = f"{file_path}.backup.{int(time.time())}"
                subprocess.run(['cp', file_path, backup_path])
                logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.error(f"Error creating file backup: {e}")


class SomnusBlueTeamAgent:
    """Main Blue Team Agent for automated defensive security operations"""
    
    def __init__(self, security_framework: SomnusSecurityFramework,
                 vm_supervisor: VMSupervisor,
                 threat_intelligence: ThreatIntelligenceEngine):
        self.security_framework = security_framework
        self.vm_supervisor = vm_supervisor
        self.threat_intelligence = threat_intelligence
        self.audit_logger = security_framework.audit_logger
        
        # Load configuration
        self.config = self._load_blue_team_config()
        
        # Initialize monitoring engines
        self.log_analyzer = LogAnalysisEngine(self.config.get('log_analysis', {}))
        self.network_monitor = NetworkTrafficMonitor(self.config.get('network_monitoring', {}))
        self.resource_monitor = ResourceMonitor(self.config.get('resource_monitoring', {}))
        
        # Initialize response and policy engines
        self.response_orchestrator = AutomatedResponseOrchestrator(vm_supervisor, security_framework)
        self.policy_engine = PolicyEnforcementEngine(self.config.get('policy_enforcement', {}))
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.start_time = datetime.utcnow()
        self.events_processed = 0
        self.threats_mitigated = 0
        
        logger.info("Somnus Blue Team Agent initialized")
    
    def _load_blue_team_config(self) -> Dict[str, Any]:
        """Load blue team configuration"""
        default_config = {
            'monitoring_enabled': True,
            'auto_response_enabled': True,
            'log_sources': [
                '/var/log/auth.log',
                '/var/log/syslog',
                '/var/log/somnus/security.log',
                '/var/log/somnus/api.log'
            ],
            'alert_thresholds': {
                'events_per_minute': 100,
                'failed_auth_per_hour': 50,
                'resource_exhaustion_threshold': 95
            },
            'prometheus_port': 8001
        }
        
        # Load from configuration file
        config_path = Path("blue_team_config.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load blue team config: {e}")
        
        return default_config
    
    async def start_monitoring(self):
        """Start all blue team monitoring operations"""
        if self.monitoring_active:
            logger.warning("Blue team monitoring already active")
            return
        
        logger.info("Starting Somnus Blue Team monitoring operations")
        
        # Start Prometheus metrics server
        if self.config.get('prometheus_port'):
            start_http_server(self.config['prometheus_port'])
            logger.info(f"Started Prometheus metrics server on port {self.config['prometheus_port']}")
        
        # Start monitoring engines
        monitoring_tasks = [
            asyncio.create_task(self.log_analyzer.start_monitoring(self.config['log_sources'])),
            asyncio.create_task(self.network_monitor.start_monitoring()),
            asyncio.create_task(self.resource_monitor.start_monitoring()),
            asyncio.create_task(self._threat_correlation_engine()),
            asyncio.create_task(self._health_monitoring_loop())
        ]
        
        self.monitoring_tasks.extend(monitoring_tasks)
        self.monitoring_active = True
        
        # Update metrics
        ACTIVE_MONITORS.set(len(monitoring_tasks))
        
        logger.info("Blue Team monitoring operations started successfully")
        
        # Wait for monitoring tasks
        try:
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in blue team monitoring: {e}")
        finally:
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop all monitoring operations"""
        logger.info("Stopping Blue Team monitoring operations")
        
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        ACTIVE_MONITORS.set(0)
        
        logger.info("Blue Team monitoring operations stopped")
    
    async def _threat_correlation_engine(self):
        """Correlate threats across different monitoring sources"""
        correlation_window = timedelta(minutes=5)
        
        while self.monitoring_active:
            try:
                # Collect recent events from all sources
                current_time = datetime.utcnow()
                
                # Get recent log events
                recent_log_events = [
                    event for event in self.log_analyzer.recent_events
                    if current_time - event['timestamp'] <= correlation_window
                ]
                
                # Get recent network events
                recent_network_data = [
                    data for data in self.network_monitor.traffic_data
                    if current_time - data['timestamp'] <= correlation_window
                ]
                
                # Correlate events
                correlated_threats = await self._correlate_threat_events(
                    recent_log_events, recent_network_data
                )
                
                # Process correlated threats
                for threat in correlated_threats:
                    await self._handle_correlated_threat(threat)
                
                await asyncio.sleep(30)  # Correlate every 30 seconds
            
            except Exception as e:
                logger.error(f"Error in threat correlation: {e}")
                await asyncio.sleep(60)
    
    async def _correlate_threat_events(self, log_events: List[Dict], 
                                     network_events: List[Dict]) -> List[Dict[str, Any]]:
        """Correlate threat events from different sources"""
        correlated_threats = []
        
        # Group events by source IP
        ip_events = defaultdict(lambda: {'log_events': [], 'network_events': []})
        
        for log_event in log_events:
            # Extract IP from log event (would need more sophisticated parsing)
            ip = self._extract_ip_from_log(log_event.get('line', ''))
            if ip:
                ip_events[ip]['log_events'].append(log_event)
        
        for net_event in network_events:
            ip = net_event.get('remote_addr')
            if ip:
                ip_events[ip]['network_events'].append(net_event)
        
        # Analyze correlations
        for ip, events in ip_events.items():
            if len(events['log_events']) > 0 and len(events['network_events']) > 0:
                # Potential coordinated attack
                threat_score = self._calculate_threat_score(events)
                
                if threat_score > 0.7:  # High correlation threshold
                    correlated_threats.append({
                        'type': 'coordinated_attack',
                        'source_ip': ip,
                        'threat_score': threat_score,
                        'log_events': events['log_events'],
                        'network_events': events['network_events'],
                        'timestamp': datetime.utcnow()
                    })
        
        return correlated_threats
    
    def _extract_ip_from_log(self, log_line: str) -> Optional[str]:
        """Extract IP address from log line"""
        # Simple IP extraction regex
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        match = re.search(ip_pattern, log_line)
        return match.group() if match else None
    
    def _calculate_threat_score(self, events: Dict[str, List]) -> float:
        """Calculate threat score based on correlated events"""
        score = 0.0
        
        # Weight based on number of different event types
        log_event_types = len(set(event.get('type') for event in events['log_events']))
        network_event_variety = len(set(event.get('remote_port') for event in events['network_events']))
        
        # Higher score for multiple event types
        score += min(log_event_types * 0.3, 0.6)
        score += min(network_event_variety * 0.1, 0.4)
        
        return min(score, 1.0)
    
    async def _handle_correlated_threat(self, threat: Dict[str, Any]):
        """Handle correlated threat with elevated response"""
        logger.critical(f"Correlated threat detected: {threat['type']} from {threat['source_ip']}")
        
        # Create security event
        event = SecurityEvent(
            event_type="correlated_threat",
            threat_level=ThreatLevel.CRITICAL,
            source_ip=threat['source_ip'],
            details=threat
        )
        
        # Trigger automated response
        actions = await self.response_orchestrator.handle_threat(event)
        
        self.threats_mitigated += 1
        
        logger.info(f"Executed {len(actions)} defensive actions for correlated threat")
    
    async def _health_monitoring_loop(self):
        """Monitor blue team agent health and performance"""
        while self.monitoring_active:
            try:
                # Calculate uptime
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                
                # Check monitoring engine health
                engine_health = {
                    'log_analyzer': len(self.log_analyzer.active_monitors) > 0,
                    'network_monitor': len(self.network_monitor.traffic_data) > 0,
                    'resource_monitor': len(self.resource_monitor.resource_history) > 0
                }
                
                # Calculate overall health score
                healthy_engines = sum(engine_health.values())
                health_percentage = (healthy_engines / len(engine_health)) * 100
                
                # Update health metrics
                SYSTEM_HEALTH.set(health_percentage)
                
                # Log health status
                if health_percentage < 80:
                    logger.warning(f"Blue Team health degraded: {health_percentage}%")
                
                # Performance metrics
                events_per_second = self.events_processed / max(uptime, 1)
                logger.debug(f"Blue Team performance: {events_per_second:.2f} events/sec, {self.threats_mitigated} threats mitigated")
                
                await asyncio.sleep(60)  # Health check every minute
            
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def get_blue_team_status(self) -> Dict[str, Any]:
        """Get comprehensive blue team status"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'status': 'active' if self.monitoring_active else 'inactive',
            'uptime_seconds': uptime,
            'events_processed': self.events_processed,
            'threats_mitigated': self.threats_mitigated,
            'monitoring_engines': {
                'log_analyzer': {
                    'active_monitors': len(self.log_analyzer.active_monitors),
                    'recent_events': len(self.log_analyzer.recent_events),
                    'threat_counters': dict(self.log_analyzer.threat_counters)
                },
                'network_monitor': {
                    'traffic_data_points': len(self.network_monitor.traffic_data),
                    'blocked_ips': len(self.network_monitor.blocked_ips),
                    'rate_limited_ips': len(self.network_monitor.rate_limiters)
                },
                'resource_monitor': {
                    'resource_history': len(self.resource_monitor.resource_history),
                    'process_baselines': len(self.resource_monitor.process_baselines)
                }
            },
            'response_orchestrator': {
                'actions_taken': len(self.response_orchestrator.action_history),
                'active_quarantines': len(self.response_orchestrator.active_quarantines)
            },
            'policy_engine': {
                'active_policies': len(self.policy_engine.active_policies),
                'policy_violations': len(self.policy_engine.policy_violations)
            }
        }
    
    async def force_security_scan(self, vm_id: str) -> Dict[str, Any]:
        """Force immediate security scan of specific VM"""
        logger.info(f"Forcing security scan of VM {vm_id}")
        
        vm_instance = await self.vm_supervisor.get_vm(vm_id)
        if not vm_instance:
            raise ValueError(f"VM {vm_id} not found")
        
        scan_results = {
            'vm_id': vm_id,
            'scan_timestamp': datetime.utcnow().isoformat(),
            'findings': []
        }
        
        # Perform comprehensive scan
        try:
            # Resource analysis
            if vm_instance.internal_ip:
                agent_client = SomnusVMAgentClient(vm_instance.internal_ip, vm_instance.agent_port)
                stats = await agent_client.get_runtime_stats()
                
                if stats.overall_cpu_percent > 90:
                    scan_results['findings'].append({
                        'type': 'high_cpu_usage',
                        'severity': 'medium',
                        'value': stats.overall_cpu_percent
                    })
                
                if stats.overall_memory_percent > 90:
                    scan_results['findings'].append({
                        'type': 'high_memory_usage',
                        'severity': 'medium',
                        'value': stats.overall_memory_percent
                    })
        
        except Exception as e:
            scan_results['error'] = str(e)
        
        return scan_results


def create_blue_team_agent(security_framework: SomnusSecurityFramework,
                          vm_supervisor: VMSupervisor,
                          threat_intelligence: ThreatIntelligenceEngine) -> SomnusBlueTeamAgent:
    """Factory function to create and configure blue team agent"""
    agent = SomnusBlueTeamAgent(security_framework, vm_supervisor, threat_intelligence)
    return agent


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize dependencies
        security_framework = SomnusSecurityFramework()
        vm_supervisor = VMSupervisor()
        threat_intelligence = ThreatIntelligenceEngine()
        
        # Create blue team agent
        blue_team = create_blue_team_agent(security_framework, vm_supervisor, threat_intelligence)
        
        # Start defensive operations
        logger.info("Starting Somnus Blue Team Agent")
        await blue_team.start_monitoring()
    
    asyncio.run(main())