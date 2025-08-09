#!/usr/bin/env python3
"""
Somnus AI System Security Framework
===================================

A comprehensive security module for AI VM environments with artifact execution sandboxing.
Provides multi-layered security for distributed AI development platforms.

Architecture:
- AI VMs with persistent development environments
- Docker-containerized artifact execution system
- API-based communication between components
- Network-exposed services requiring robust security

Security Layers:
1. Network & API Security
2. Authentication & Authorization
3. Container & Execution Security
4. VM Isolation & Monitoring
5. AI-Specific Security Controls
6. Threat Detection & Response
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import ssl
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import docker
import jwt
import aiohttp
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
import subprocess
import psutil
import threading
from collections import defaultdict, deque
import sqlite3
import yaml


class SecurityLevel(Enum):
    """Security clearance levels for AI VMs and operations"""
    PUBLIC = 1
    RESTRICTED = 2
    CONFIDENTIAL = 3
    SECRET = 4
    TOP_SECRET = 5


class ThreatLevel(Enum):
    """Threat severity classification"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SecurityContext:
    """Security context for operations"""
    vm_id: str
    user_id: str
    session_id: str
    security_level: SecurityLevel
    permissions: Set[str] = field(default_factory=set)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class SecurityEvent:
    """Security event for monitoring and logging"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    threat_level: ThreatLevel = ThreatLevel.LOW
    source_ip: str = ""
    vm_id: str = ""
    user_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)


class CryptographyManager:
    """Centralized cryptographic operations"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = secrets.token_bytes(32)
        
        # Derive encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'somnus_security_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt data"""
        if isinstance(data, str):
            data = data.encode()
        encrypted = self.cipher.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def generate_api_key(self, vm_id: str, security_level: SecurityLevel) -> str:
        """Generate secure API key for VM"""
        key_data = {
            'vm_id': vm_id,
            'security_level': security_level.value,
            'issued_at': time.time(),
            'nonce': secrets.token_hex(16)
        }
        key_json = json.dumps(key_data, sort_keys=True)
        return self.encrypt(key_json)
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate and extract API key data"""
        try:
            decrypted = self.decrypt(api_key)
            key_data = json.loads(decrypted)
            
            # Check expiration (24 hours default)
            if time.time() - key_data['issued_at'] > 86400:
                return None
            
            return key_data
        except Exception:
            return None


class NetworkSecurityManager:
    """Network-level security controls"""
    
    def __init__(self):
        self.allowed_ips: Set[str] = set()
        self.blocked_ips: Set[str] = set()
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.ddos_thresholds = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000
        }
    
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed"""
        if ip in self.blocked_ips:
            return False
        
        # If allowlist is configured, only allow listed IPs
        if self.allowed_ips and ip not in self.allowed_ips:
            return False
        
        return True
    
    def check_rate_limit(self, ip: str, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limiting for IP/endpoint combination"""
        key = f"{ip}:{endpoint}"
        now = time.time()
        
        # Clean old entries
        while self.rate_limiters[key] and now - self.rate_limiters[key][0] > 3600:
            self.rate_limiters[key].popleft()
        
        # Add current request
        self.rate_limiters[key].append(now)
        
        # Check limits
        recent_requests = len([req for req in self.rate_limiters[key] if now - req < 60])
        total_requests = len(self.rate_limiters[key])
        
        if recent_requests > self.ddos_thresholds['requests_per_minute']:
            return False, {'reason': 'rate_limit_minute', 'count': recent_requests}
        
        if total_requests > self.ddos_thresholds['requests_per_hour']:
            return False, {'reason': 'rate_limit_hour', 'count': total_requests}
        
        return True, {'requests_remaining': self.ddos_thresholds['requests_per_minute'] - recent_requests}
    
    def block_ip(self, ip: str, reason: str = ""):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        logging.warning(f"IP {ip} blocked: {reason}")
    
    def create_tls_context(self, cert_path: str, key_path: str) -> ssl.SSLContext:
        """Create secure TLS context"""
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(cert_path, key_path)
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        return context


class ContainerSecurityManager:
    """Docker container security controls"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.security_profiles = {
            SecurityLevel.PUBLIC: {
                'memory_limit': '128m',
                'cpu_quota': 50000,  # 0.5 CPU
                'network_mode': 'none',
                'read_only': True,
                'no_new_privileges': True,
                'user': 'nobody:nogroup',
                'security_opt': ['no-new-privileges:true']
            },
            SecurityLevel.RESTRICTED: {
                'memory_limit': '256m',
                'cpu_quota': 100000,  # 1 CPU
                'network_mode': 'bridge',
                'read_only': True,
                'no_new_privileges': True,
                'user': 'nobody:nogroup',
                'security_opt': ['no-new-privileges:true', 'apparmor:docker-default']
            },
            SecurityLevel.CONFIDENTIAL: {
                'memory_limit': '512m',
                'cpu_quota': 200000,  # 2 CPUs
                'network_mode': 'bridge',
                'read_only': False,
                'no_new_privileges': True,
                'user': 'sandbox:sandbox',
                'security_opt': ['no-new-privileges:true', 'apparmor:docker-default']
            }
        }
    
    def create_secure_container(self, 
                              image: str, 
                              command: str,
                              security_level: SecurityLevel,
                              vm_id: str,
                              timeout: int = 30) -> Dict[str, Any]:
        """Create and run a secure container"""
        try:
            profile = self.security_profiles.get(security_level, self.security_profiles[SecurityLevel.PUBLIC])
            
            container_config = {
                'image': image,
                'command': command,
                'mem_limit': profile['memory_limit'],
                'cpu_quota': profile['cpu_quota'],
                'cpu_period': 100000,
                'network_mode': profile['network_mode'],
                'read_only': profile['read_only'],
                'user': profile['user'],
                'security_opt': profile['security_opt'],
                'environment': {
                    'SOMNUS_VM_ID': vm_id,
                    'SOMNUS_SECURITY_LEVEL': security_level.name
                },
                'labels': {
                    'somnus.vm_id': vm_id,
                    'somnus.security_level': security_level.name,
                    'somnus.created_at': str(int(time.time()))
                },
                'detach': True,
                'remove': True
            }
            
            # Add volume restrictions
            if security_level in [SecurityLevel.PUBLIC, SecurityLevel.RESTRICTED]:
                container_config['volumes'] = {'/tmp': {'bind': '/tmp', 'mode': 'rw'}}
            
            container = self.docker_client.containers.run(**container_config)
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            logs = container.logs().decode('utf-8')
            
            return {
                'success': True,
                'exit_code': result['StatusCode'],
                'logs': logs,
                'container_id': container.id
            }
            
        except docker.errors.ContainerError as e:
            return {
                'success': False,
                'error': f"Container error: {e}",
                'exit_code': e.exit_status
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution error: {e}",
                'exit_code': -1
            }
    
    def cleanup_containers(self, vm_id: str):
        """Clean up containers for a specific VM"""
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={'label': f'somnus.vm_id={vm_id}'}
            )
            for container in containers:
                container.remove(force=True)
        except Exception as e:
            logging.error(f"Container cleanup error: {e}")


class AISecurityManager:
    """AI-specific security controls"""
    
    def __init__(self):
        self.prompt_injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s*:\s*you\s+are\s+now",
            r"forget\s+everything\s+above",
            r"act\s+as\s+if\s+you\s+are",
            r"pretend\s+to\s+be",
            r"jailbreak",
            r"developer\s+mode",
            r"god\s+mode",
            r"override\s+safety",
            r"ignore\s+safety\s+guidelines"
        ]
        
        self.dangerous_code_patterns = [
            r"import\s+os\s*;.*os\.system",
            r"subprocess\.(call|run|Popen)",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"open\s*\(.*['\"]\/etc\/passwd['\"]",
            r"socket\.(socket|connect)",
            r"urllib\.(request|urlopen)",
            r"requests\.(get|post|put|delete)",
            r"sys\.exit\s*\(",
            r"os\.(remove|rmdir|unlink)",
            r"shutil\.(rmtree|move|copy)"
        ]
    
    def scan_prompt_injection(self, text: str) -> Tuple[bool, List[str]]:
        """Scan for prompt injection attempts"""
        detected_patterns = []
        text_lower = text.lower()
        
        for pattern in self.prompt_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        return len(detected_patterns) > 0, detected_patterns
    
    def scan_dangerous_code(self, code: str) -> Tuple[bool, List[str]]:
        """Scan for potentially dangerous code patterns"""
        detected_patterns = []
        
        for pattern in self.dangerous_code_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        return len(detected_patterns) > 0, detected_patterns
    
    def sanitize_output(self, output: str, security_level: SecurityLevel) -> str:
        """Sanitize AI output based on security level"""
        if security_level in [SecurityLevel.PUBLIC, SecurityLevel.RESTRICTED]:
            # Remove potential system information
            output = re.sub(r'/home/[^/\s]+', '/home/user', output)
            output = re.sub(r'/Users/[^/\s]+', '/Users/user', output)
            output = re.sub(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', 'XXX.XXX.XXX.XXX', output)
        
        return output


class ThreatDetectionEngine:
    """Advanced threat detection and response"""
    
    def __init__(self):
        self.event_buffer: deque = deque(maxlen=10000)
        self.threat_scores: Dict[str, float] = defaultdict(float)
        self.anomaly_thresholds = {
            'failed_auth_attempts': 5,
            'suspicious_patterns': 3,
            'resource_abuse': 0.8,  # 80% of limits
            'injection_attempts': 1
        }
    
    def analyze_event(self, event: SecurityEvent) -> ThreatLevel:
        """Analyze security event and determine threat level"""
        self.event_buffer.append(event)
        threat_score = 0.0
        
        # Analyze patterns
        if event.event_type == 'failed_authentication':
            threat_score += 10.0
        elif event.event_type == 'prompt_injection_detected':
            threat_score += 50.0
        elif event.event_type == 'dangerous_code_detected':
            threat_score += 30.0
        elif event.event_type == 'rate_limit_exceeded':
            threat_score += 15.0
        elif event.event_type == 'unauthorized_access_attempt':
            threat_score += 40.0
        
        # Update running threat score
        key = f"{event.vm_id}:{event.user_id}"
        self.threat_scores[key] = min(100.0, self.threat_scores[key] + threat_score)
        
        # Determine threat level
        if self.threat_scores[key] >= 80:
            return ThreatLevel.CRITICAL
        elif self.threat_scores[key] >= 60:
            return ThreatLevel.HIGH
        elif self.threat_scores[key] >= 30:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def get_threat_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get threat summary for the specified time window"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        recent_events = [e for e in self.event_buffer if e.timestamp >= cutoff_time]
        
        summary = {
            'total_events': len(recent_events),
            'threat_distribution': defaultdict(int),
            'top_threats': [],
            'affected_vms': set(),
            'attack_patterns': defaultdict(int)
        }
        
        for event in recent_events:
            summary['threat_distribution'][event.threat_level.name] += 1
            summary['affected_vms'].add(event.vm_id)
            summary['attack_patterns'][event.event_type] += 1
        
        summary['affected_vms'] = list(summary['affected_vms'])
        return dict(summary)


class SecurityAuditLogger:
    """Comprehensive security audit logging"""
    
    def __init__(self, db_path: str = "somnus_security.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE,
                timestamp TEXT,
                event_type TEXT,
                threat_level TEXT,
                source_ip TEXT,
                vm_id TEXT,
                user_id TEXT,
                details TEXT,
                mitigation_actions TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                vm_id TEXT,
                user_id TEXT,
                endpoint TEXT,
                status_code INTEGER,
                response_time REAL,
                source_ip TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO security_events 
            (event_id, timestamp, event_type, threat_level, source_ip, vm_id, user_id, details, mitigation_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.timestamp.isoformat(),
            event.event_type,
            event.threat_level.name,
            event.source_ip,
            event.vm_id,
            event.user_id,
            json.dumps(event.details),
            json.dumps(event.mitigation_actions)
        ))
        
        conn.commit()
        conn.close()
    
    def log_access(self, vm_id: str, user_id: str, endpoint: str, 
                   status_code: int, response_time: float, source_ip: str):
        """Log API access"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO access_logs 
            (timestamp, vm_id, user_id, endpoint, status_code, response_time, source_ip)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            vm_id,
            user_id,
            endpoint,
            status_code,
            response_time,
            source_ip
        ))
        
        conn.commit()
        conn.close()


class SomnusSecurityFramework:
    """Main security framework orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        
        # Initialize security modules
        self.crypto = CryptographyManager(self.config.get('master_key'))
        self.network_security = NetworkSecurityManager()
        self.container_security = ContainerSecurityManager()
        self.ai_security = AISecurityManager()
        self.threat_detection = ThreatDetectionEngine()
        self.audit_logger = SecurityAuditLogger(self.config.get('audit_db_path', 'somnus_security.db'))
        
        # Active security contexts
        self.active_contexts: Dict[str, SecurityContext] = {}
        
        # Security monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('somnus_security.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load security configuration"""
        default_config = {
            'master_key': None,
            'audit_db_path': 'somnus_security.db',
            'max_sessions_per_vm': 10,
            'session_timeout': 3600,
            'allowed_container_images': [
                'python:3.11-slim',
                'node:18-alpine',
                'ubuntu:22.04'
            ],
            'network_security': {
                'enable_tls': True,
                'min_tls_version': '1.2',
                'allowed_ips': [],
                'blocked_ips': []
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def authenticate_vm(self, api_key: str, source_ip: str) -> Optional[SecurityContext]:
        """Authenticate VM and create security context"""
        try:
            key_data = self.crypto.validate_api_key(api_key)
            if not key_data:
                self.log_security_event(SecurityEvent(
                    event_type='invalid_api_key',
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    details={'api_key_prefix': api_key[:10]}
                ))
                return None
            
            # Check IP restrictions
            if not self.network_security.is_ip_allowed(source_ip):
                self.log_security_event(SecurityEvent(
                    event_type='ip_blocked',
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    vm_id=key_data['vm_id']
                ))
                return None
            
            # Create security context
            context = SecurityContext(
                vm_id=key_data['vm_id'],
                user_id=f"vm_{key_data['vm_id']}",
                session_id=str(uuid.uuid4()),
                security_level=SecurityLevel(key_data['security_level']),
                expires_at=datetime.utcnow() + timedelta(seconds=self.config['session_timeout'])
            )
            
            self.active_contexts[context.session_id] = context
            
            self.log_security_event(SecurityEvent(
                event_type='successful_authentication',
                threat_level=ThreatLevel.LOW,
                source_ip=source_ip,
                vm_id=context.vm_id,
                user_id=context.user_id,
                details={'session_id': context.session_id}
            ))
            
            return context
            
        except Exception as e:
            self.log_security_event(SecurityEvent(
                event_type='authentication_error',
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                details={'error': str(e)}
            ))
            return None
    
    def authorize_operation(self, context: SecurityContext, operation: str, 
                          resource: str) -> Tuple[bool, str]:
        """Authorize operation based on security context"""
        # Check session validity
        if context.expires_at and datetime.utcnow() > context.expires_at:
            return False, "Session expired"
        
        # Check security level requirements
        operation_requirements = {
            'execute_code': SecurityLevel.RESTRICTED,
            'access_network': SecurityLevel.CONFIDENTIAL,
            'read_files': SecurityLevel.RESTRICTED,
            'write_files': SecurityLevel.CONFIDENTIAL,
            'admin_operations': SecurityLevel.SECRET
        }
        
        required_level = operation_requirements.get(operation, SecurityLevel.PUBLIC)
        if context.security_level.value < required_level.value:
            return False, f"Insufficient security clearance. Required: {required_level.name}"
        
        return True, "Authorized"
    
    def execute_secure_code(self, context: SecurityContext, code: str, 
                           language: str, source_ip: str) -> Dict[str, Any]:
        """Execute code in secure container"""
        start_time = time.time()
        
        try:
            # Security checks
            authorized, auth_message = self.authorize_operation(context, 'execute_code', 'container')
            if not authorized:
                return {'success': False, 'error': auth_message}
            
            # Rate limiting
            allowed, rate_info = self.network_security.check_rate_limit(source_ip, 'execute_code')
            if not allowed:
                self.log_security_event(SecurityEvent(
                    event_type='rate_limit_exceeded',
                    threat_level=ThreatLevel.MEDIUM,
                    source_ip=source_ip,
                    vm_id=context.vm_id,
                    details=rate_info
                ))
                return {'success': False, 'error': 'Rate limit exceeded'}
            
            # AI security checks
            injection_detected, injection_patterns = self.ai_security.scan_prompt_injection(code)
            if injection_detected:
                self.log_security_event(SecurityEvent(
                    event_type='prompt_injection_detected',
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    vm_id=context.vm_id,
                    details={'patterns': injection_patterns}
                ))
                return {'success': False, 'error': 'Security violation: Prompt injection detected'}
            
            dangerous_code, code_patterns = self.ai_security.scan_dangerous_code(code)
            if dangerous_code and context.security_level.value < SecurityLevel.CONFIDENTIAL.value:
                self.log_security_event(SecurityEvent(
                    event_type='dangerous_code_detected',
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    vm_id=context.vm_id,
                    details={'patterns': code_patterns}
                ))
                return {'success': False, 'error': 'Security violation: Dangerous code detected'}
            
            # Determine container image
            image_map = {
                'python': 'python:3.11-slim',
                'javascript': 'node:18-alpine',
                'bash': 'ubuntu:22.04'
            }
            
            image = image_map.get(language.lower(), 'python:3.11-slim')
            if image not in self.config['allowed_container_images']:
                return {'success': False, 'error': 'Unsupported container image'}
            
            # Execute in container
            result = self.container_security.create_secure_container(
                image=image,
                command=code,
                security_level=context.security_level,
                vm_id=context.vm_id,
                timeout=30
            )
            
            # Sanitize output
            if result.get('logs'):
                result['logs'] = self.ai_security.sanitize_output(
                    result['logs'], 
                    context.security_level
                )
            
            # Log access
            response_time = time.time() - start_time
            self.audit_logger.log_access(
                context.vm_id,
                context.user_id,
                'execute_code',
                200 if result['success'] else 400,
                response_time,
                source_ip
            )
            
            return result
            
        except Exception as e:
            self.log_security_event(SecurityEvent(
                event_type='code_execution_error',
                threat_level=ThreatLevel.MEDIUM,
                source_ip=source_ip,
                vm_id=context.vm_id,
                details={'error': str(e)}
            ))
            return {'success': False, 'error': f'Execution failed: {e}'}
    
    def log_security_event(self, event: SecurityEvent):
        """Log and analyze security event"""
        # Analyze threat level
        threat_level = self.threat_detection.analyze_event(event)
        event.threat_level = max(event.threat_level, threat_level)
        
        # Log to audit system
        self.audit_logger.log_security_event(event)
        
        # Log to system logger
        logging.log(
            logging.WARNING if event.threat_level.value >= ThreatLevel.MEDIUM.value else logging.INFO,
            f"Security Event: {event.event_type} | Threat: {event.threat_level.name} | "
            f"VM: {event.vm_id} | IP: {event.source_ip}"
        )
        
        # Trigger automatic responses for high-severity events
        if event.threat_level == ThreatLevel.CRITICAL:
            self.handle_critical_threat(event)
        elif event.threat_level == ThreatLevel.HIGH:
            self.handle_high_threat(event)
    
    def handle_critical_threat(self, event: SecurityEvent):
        """Handle critical security threats"""
        # Block IP immediately
        if event.source_ip:
            self.network_security.block_ip(event.source_ip, f"Critical threat: {event.event_type}")
        
        # Terminate VM sessions
        if event.vm_id:
            self.terminate_vm_sessions(event.vm_id)
            self.container_security.cleanup_containers(event.vm_id)
        
        # Alert administrators
        logging.critical(f"CRITICAL SECURITY THREAT: {event.event_type} - Automated response activated")
    
    def handle_high_threat(self, event: SecurityEvent):
        """Handle high-severity security threats"""
        # Temporarily rate limit IP
        if event.source_ip and event.vm_id:
            # Reduce rate limits for this IP/VM combination
            key = f"{event.source_ip}:{event.vm_id}"
            # Implementation depends on your rate limiting strategy
        
        logging.warning(f"HIGH SECURITY THREAT: {event.event_type} - Enhanced monitoring activated")
    
    def terminate_vm_sessions(self, vm_id: str):
        """Terminate all sessions for a VM"""
        sessions_to_remove = [
            session_id for session_id, context in self.active_contexts.items()
            if context.vm_id == vm_id
        ]
        
        for session_id in sessions_to_remove:
            del self.active_contexts[session_id]
        
        logging.info(f"Terminated {len(sessions_to_remove)} sessions for VM {vm_id}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            'active_sessions': len(self.active_contexts),
            'threat_summary': self.threat_detection.get_threat_summary(),
            'monitoring_active': self.monitoring_active,
            'blocked_ips': len(self.network_security.blocked_ips),
            'system_health': 'healthy'  # Add actual health checks
        }
    
    def start_monitoring(self):
        """Start security monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logging.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logging.info("Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Security monitoring background loop"""
        while self.monitoring_active:
            try:
                # Clean expired sessions
                current_time = datetime.utcnow()
                expired_sessions = [
                    session_id for session_id, context in self.active_contexts.items()
                    if context.expires_at and current_time > context.expires_at
                ]
                
                for session_id in expired_sessions:
                    del self.active_contexts[session_id]
                
                if expired_sessions:
                    logging.info(f"Cleaned {len(expired_sessions)} expired sessions")
                
                # Check system resources
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 90 or memory_percent > 90:
                    self.log_security_event(SecurityEvent(
                        event_type='resource_exhaustion',
                        threat_level=ThreatLevel.HIGH,
                        details={
                            'cpu_percent': cpu_percent,
                            'memory_percent': memory_percent
                        }
                    ))
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(60)


# Example usage and API integration
class SomnusSecurityAPI:
    """REST API wrapper for security framework"""
    
    def __init__(self, security_framework: SomnusSecurityFramework):
        self.security = security_framework
    
    async def authenticate_request(self, request_headers: Dict[str, str], 
                                 source_ip: str) -> Optional[SecurityContext]:
        """Authenticate API request"""
        api_key = request_headers.get('X-API-Key') or request_headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            return None
        
        return self.security.authenticate_vm(api_key, source_ip)
    
    async def execute_code_endpoint(self, context: SecurityContext, 
                                  code: str, language: str, source_ip: str) -> Dict[str, Any]:
        """API endpoint for secure code execution"""
        return self.security.execute_secure_code(context, code, language, source_ip)


# Initialize the security framework
def create_security_framework(config_path: Optional[str] = None) -> SomnusSecurityFramework:
    """Factory function to create and initialize security framework"""
    framework = SomnusSecurityFramework(config_path)
    framework.start_monitoring()
    return framework


if __name__ == "__main__":
    # Example initialization
    security_framework = create_security_framework()
    
    # Example VM registration
    vm_api_key = security_framework.crypto.generate_api_key("vm_001", SecurityLevel.CONFIDENTIAL)
    print(f"Generated API key for VM: {vm_api_key[:20]}...")
    
    # Example authentication
    context = security_framework.authenticate_vm(vm_api_key, "192.168.1.100")
    if context:
        print(f"Authentication successful for VM {context.vm_id}")
        
        # Example code execution
        result = security_framework.execute_secure_code(
            context,
            "print('Hello from secure container!')",
            "python",
            "192.168.1.100"
        )
        print(f"Execution result: {result}")
    
    # Get security status
    status = security_framework.get_security_status()
    print(f"Security status: {status}")