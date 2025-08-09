#!/usr/bin/env python3
"""
Integrated Somnus Security Architecture
=====================================

Combines existing AI-focused security layer with infrastructure security framework.
Creates comprehensive defense-in-depth protection for distributed AI systems.

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED THREAT INTELLIGENCE                  │
│                     (Shared Threat Scoring)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
┌───────────────▼───────────────┐             ┌▼─────────────────────────┐
│     APPLICATION LAYER         │             │   INFRASTRUCTURE LAYER   │
│   (Existing Security Layer)   │             │    (New Framework)       │
│                               │             │                          │
│ • Content Classification      │◄────────────┤ • Network Security       │
│ • Prompt Injection Detection  │             │ • Container Isolation    │
│ • Capability Enforcement      │             │ • API Authentication     │
│ • AI-Specific Threats         │             │ • System Monitoring      │
│ • Rate Limiting               │             │ • Automated Response     │
└───────────────────────────────┘             └──────────────────────────┘
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import existing security components
from security_layer import (
    SecurityEnforcer as AISecurityEnforcer,
    SecurityAnalysisResult,
    ThreatLevel as AIThreatLevel,
    SecurityViolationType
)

# Import new infrastructure components (from my previous framework)
from infrastructure_security import (
    SomnusSecurityFramework,
    SecurityLevel,
    SecurityContext,
    SecurityEvent,
    ThreatLevel as InfraThreatLevel
)

logger = logging.getLogger(__name__)


class UnifiedThreatLevel(Enum):
    """Unified threat classification combining AI and infrastructure assessments"""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


@dataclass
class UnifiedSecurityResult:
    """Comprehensive security result combining both layers"""
    allowed: bool
    threat_level: UnifiedThreatLevel
    combined_score: float
    ai_analysis: SecurityAnalysisResult
    infra_analysis: Optional[Dict[str, Any]] = None
    coordinated_actions: List[str] = field(default_factory=list)
    security_context: Optional[SecurityContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreatIntelligenceEngine:
    """
    Unified threat intelligence that correlates AI and infrastructure threats.
    Provides coordinated response across both security layers.
    """
    
    def __init__(self):
        self.threat_correlations = {}
        self.incident_history = []
        
        # Threat escalation rules
        self.escalation_matrix = {
            # (AI_threat, Infra_threat) -> Unified_threat
            (AIThreatLevel.CRITICAL, InfraThreatLevel.CRITICAL): UnifiedThreatLevel.CRITICAL,
            (AIThreatLevel.CRITICAL, InfraThreatLevel.HIGH): UnifiedThreatLevel.CRITICAL,
            (AIThreatLevel.HIGH, InfraThreatLevel.CRITICAL): UnifiedThreatLevel.CRITICAL,
            (AIThreatLevel.HIGH, InfraThreatLevel.HIGH): UnifiedThreatLevel.HIGH,
            (AIThreatLevel.MEDIUM, InfraThreatLevel.CRITICAL): UnifiedThreatLevel.HIGH,
            (AIThreatLevel.CRITICAL, InfraThreatLevel.MEDIUM): UnifiedThreatLevel.HIGH,
            # Add more combinations as needed
        }
    
    def correlate_threats(self, 
                         ai_threat: AIThreatLevel, 
                         infra_threat: InfraThreatLevel,
                         ai_score: float,
                         infra_score: float) -> Tuple[UnifiedThreatLevel, float]:
        """
        Correlate threats from both layers to determine unified threat level.
        Uses weighted scoring and escalation matrix.
        """
        # Direct mapping if exists
        threat_tuple = (ai_threat, infra_threat)
        if threat_tuple in self.escalation_matrix:
            unified_level = self.escalation_matrix[threat_tuple]
        else:
            # Fallback to score-based calculation
            combined_score = (ai_score * 0.6) + (infra_score * 0.4)  # AI threats weighted higher
            
            if combined_score >= 0.9:
                unified_level = UnifiedThreatLevel.CRITICAL
            elif combined_score >= 0.7:
                unified_level = UnifiedThreatLevel.HIGH
            elif combined_score >= 0.5:
                unified_level = UnifiedThreatLevel.MEDIUM
            elif combined_score >= 0.3:
                unified_level = UnifiedThreatLevel.LOW
            else:
                unified_level = UnifiedThreatLevel.MINIMAL
        
        # Calculate final combined score
        final_score = min((ai_score * 0.6) + (infra_score * 0.4), 1.0)
        
        # Apply amplification for certain threat combinations
        if self._has_threat_amplification(ai_threat, infra_threat):
            final_score = min(final_score * 1.3, 1.0)
            unified_level = UnifiedThreatLevel(min(unified_level.value + 1, 5))
        
        return unified_level, final_score
    
    def _has_threat_amplification(self, ai_threat: AIThreatLevel, infra_threat: InfraThreatLevel) -> bool:
        """Check if threat combination requires amplification"""
        # Prompt injection + network intrusion = escalated threat
        # Resource abuse + content violations = coordinated attack
        amplification_patterns = [
            (AIThreatLevel.HIGH, InfraThreatLevel.MEDIUM),
            (AIThreatLevel.MEDIUM, InfraThreatLevel.HIGH)
        ]
        
        return (ai_threat, infra_threat) in amplification_patterns
    
    def generate_coordinated_actions(self, 
                                   unified_threat: UnifiedThreatLevel,
                                   ai_analysis: SecurityAnalysisResult,
                                   infra_context: Optional[SecurityContext]) -> List[str]:
        """Generate coordinated response actions across both security layers"""
        actions = []
        
        if unified_threat == UnifiedThreatLevel.CRITICAL:
            actions.extend([
                "immediate_session_termination",
                "ip_blocking_activation",
                "container_isolation_enforcement",
                "admin_alert_critical",
                "forensic_data_collection"
            ])
        
        elif unified_threat == UnifiedThreatLevel.HIGH:
            actions.extend([
                "enhanced_monitoring_activation",
                "rate_limiting_strictness_increase",
                "container_resource_reduction",
                "admin_alert_high"
            ])
        
        elif unified_threat == UnifiedThreatLevel.MEDIUM:
            actions.extend([
                "increased_logging_verbosity",
                "capability_restriction_enforcement",
                "monitoring_window_extension"
            ])
        
        # Add specific actions based on violation types
        if SecurityViolationType.PROMPT_INJECTION in ai_analysis.violations:
            actions.append("prompt_injection_pattern_update")
        
        if SecurityViolationType.CAPABILITY_VIOLATION in ai_analysis.violations:
            actions.append("capability_permissions_review")
        
        return actions


class IntegratedSecurityOrchestrator:
    """
    Master orchestrator that coordinates AI and infrastructure security layers.
    Provides unified interface for all security operations.
    """
    
    def __init__(self, 
                 ai_security_config: Dict[str, Any],
                 infra_security_config: Dict[str, Any]):
        
        # Initialize security layers
        self.ai_security = AISecurityEnforcer(ai_security_config)
        self.infra_security = SomnusSecurityFramework(infra_security_config.get('config_path'))
        
        # Initialize unified components
        self.threat_intelligence = ThreatIntelligenceEngine()
        
        # Active security contexts
        self.active_contexts: Dict[str, SecurityContext] = {}
        
        logger.info("Integrated security orchestrator initialized")
    
    async def authenticate_and_validate_session(self, 
                                               api_key: str,
                                               session_request: Any,
                                               source_ip: str) -> UnifiedSecurityResult:
        """
        Comprehensive session authentication and validation using both security layers.
        """
        try:
            # Infrastructure layer: Authenticate VM and create security context
            security_context = self.infra_security.authenticate_vm(api_key, source_ip)
            
            if not security_context:
                return UnifiedSecurityResult(
                    allowed=False,
                    threat_level=UnifiedThreatLevel.HIGH,
                    combined_score=0.8,
                    ai_analysis=SecurityAnalysisResult(
                        allowed=False,
                        threat_level=AIThreatLevel.HIGH,
                        threat_score=0.8,
                        reason="Infrastructure authentication failed"
                    ),
                    infra_analysis={"auth_failed": True, "source_ip": source_ip}
                )
            
            # Store security context
            self.active_contexts[security_context.session_id] = security_context
            
            # Application layer: Validate session request content
            ai_analysis = await self.ai_security.validate_session_request(session_request)
            
            # Map threat levels for correlation
            infra_threat_level = self._map_security_level_to_threat(security_context.security_level)
            
            # Correlate threats through intelligence engine
            unified_threat, combined_score = self.threat_intelligence.correlate_threats(
                ai_analysis.threat_level,
                infra_threat_level,
                ai_analysis.threat_score,
                0.2  # Low infrastructure threat for successful auth
            )
            
            # Generate coordinated actions
            coordinated_actions = self.threat_intelligence.generate_coordinated_actions(
                unified_threat,
                ai_analysis,
                security_context
            )
            
            # Execute coordinated actions if needed
            if unified_threat.value >= UnifiedThreatLevel.HIGH.value:
                await self._execute_coordinated_actions(coordinated_actions, security_context, source_ip)
            
            # Determine final allowed status
            allowed = (ai_analysis.allowed and 
                      security_context is not None and 
                      unified_threat.value <= UnifiedThreatLevel.MEDIUM.value)
            
            return UnifiedSecurityResult(
                allowed=allowed,
                threat_level=unified_threat,
                combined_score=combined_score,
                ai_analysis=ai_analysis,
                infra_analysis={
                    "authenticated": True,
                    "security_level": security_context.security_level.name,
                    "vm_id": security_context.vm_id
                },
                coordinated_actions=coordinated_actions,
                security_context=security_context
            )
            
        except Exception as e:
            logger.error(f"Integrated session validation failed: {e}")
            return UnifiedSecurityResult(
                allowed=False,
                threat_level=UnifiedThreatLevel.CRITICAL,
                combined_score=1.0,
                ai_analysis=SecurityAnalysisResult(
                    allowed=False,
                    threat_level=AIThreatLevel.CRITICAL,
                    threat_score=1.0,
                    reason=f"Security validation error: {str(e)}"
                )
            )
    
    async def validate_message_and_execute(self,
                                         message: str,
                                         session_id: str,
                                         source_ip: str,
                                         execution_context: Optional[Dict[str, Any]] = None) -> UnifiedSecurityResult:
        """
        Validate message through both layers and coordinate secure execution.
        """
        try:
            # Get security context
            security_context = self.active_contexts.get(session_id)
            if not security_context:
                return UnifiedSecurityResult(
                    allowed=False,
                    threat_level=UnifiedThreatLevel.HIGH,
                    combined_score=0.8,
                    ai_analysis=SecurityAnalysisResult(
                        allowed=False,
                        threat_level=AIThreatLevel.HIGH,
                        threat_score=0.8,
                        reason="Invalid or expired session"
                    )
                )
            
            # Application layer: Validate message content
            ai_analysis = await self.ai_security.validate_message(message, session_id, execution_context)
            
            # Infrastructure layer: Check rate limits and network security
            rate_allowed, rate_info = self.infra_security.network_security.check_rate_limit(source_ip, 'message')
            
            infra_threat_score = 0.0 if rate_allowed else 0.6
            infra_threat_level = InfraThreatLevel.LOW if rate_allowed else InfraThreatLevel.MEDIUM
            
            # Correlate threats
            unified_threat, combined_score = self.threat_intelligence.correlate_threats(
                ai_analysis.threat_level,
                infra_threat_level,
                ai_analysis.threat_score,
                infra_threat_score
            )
            
            # Generate and execute coordinated actions
            coordinated_actions = self.threat_intelligence.generate_coordinated_actions(
                unified_threat,
                ai_analysis,
                security_context
            )
            
            if unified_threat.value >= UnifiedThreatLevel.HIGH.value:
                await self._execute_coordinated_actions(coordinated_actions, security_context, source_ip)
            
            # If execution is requested and allowed, coordinate secure execution
            execution_result = None
            if (execution_context and 
                ai_analysis.allowed and 
                rate_allowed and 
                unified_threat.value <= UnifiedThreatLevel.MEDIUM.value):
                
                execution_result = await self._coordinate_secure_execution(
                    execution_context,
                    security_context,
                    source_ip
                )
            
            # Determine final allowed status
            allowed = (ai_analysis.allowed and 
                      rate_allowed and 
                      unified_threat.value <= UnifiedThreatLevel.MEDIUM.value)
            
            return UnifiedSecurityResult(
                allowed=allowed,
                threat_level=unified_threat,
                combined_score=combined_score,
                ai_analysis=ai_analysis,
                infra_analysis={
                    "rate_allowed": rate_allowed,
                    "rate_info": rate_info,
                    "execution_result": execution_result
                },
                coordinated_actions=coordinated_actions,
                security_context=security_context
            )
            
        except Exception as e:
            logger.error(f"Integrated message validation failed: {e}")
            return UnifiedSecurityResult(
                allowed=False,
                threat_level=UnifiedThreatLevel.CRITICAL,
                combined_score=1.0,
                ai_analysis=SecurityAnalysisResult(
                    allowed=False,
                    threat_level=AIThreatLevel.CRITICAL,
                    threat_score=1.0,
                    reason=f"Message validation error: {str(e)}"
                )
            )
    
    async def _coordinate_secure_execution(self,
                                         execution_context: Dict[str, Any],
                                         security_context: SecurityContext,
                                         source_ip: str) -> Dict[str, Any]:
        """Coordinate secure code execution through infrastructure layer"""
        code = execution_context.get('code', '')
        language = execution_context.get('language', 'python')
        
        # Execute through infrastructure security with container isolation
        result = self.infra_security.execute_secure_code(
            security_context,
            code,
            language,
            source_ip
        )
        
        return result
    
    async def _execute_coordinated_actions(self,
                                         actions: List[str],
                                         security_context: SecurityContext,
                                         source_ip: str):
        """Execute coordinated security actions across both layers"""
        for action in actions:
            try:
                if action == "immediate_session_termination":
                    # Terminate session in both layers
                    if security_context.session_id in self.active_contexts:
                        del self.active_contexts[security_context.session_id]
                    self.infra_security.terminate_vm_sessions(security_context.vm_id)
                
                elif action == "ip_blocking_activation":
                    # Block IP at infrastructure layer
                    self.infra_security.network_security.block_ip(source_ip, "Coordinated threat response")
                
                elif action == "container_isolation_enforcement":
                    # Clean up containers for the VM
                    self.infra_security.container_security.cleanup_containers(security_context.vm_id)
                
                elif action == "enhanced_monitoring_activation":
                    # Increase monitoring sensitivity
                    # Implementation depends on monitoring system
                    pass
                
                elif action == "admin_alert_critical":
                    # Send critical alert to administrators
                    logger.critical(f"CRITICAL SECURITY THREAT: Coordinated response activated for VM {security_context.vm_id}")
                
                # Add more action implementations as needed
                
            except Exception as e:
                logger.error(f"Failed to execute coordinated action {action}: {e}")
    
    def _map_security_level_to_threat(self, security_level: SecurityLevel) -> InfraThreatLevel:
        """Map security clearance level to threat level"""
        mapping = {
            SecurityLevel.PUBLIC: InfraThreatLevel.LOW,
            SecurityLevel.RESTRICTED: InfraThreatLevel.LOW,
            SecurityLevel.CONFIDENTIAL: InfraThreatLevel.MEDIUM,
            SecurityLevel.SECRET: InfraThreatLevel.HIGH,
            SecurityLevel.TOP_SECRET: InfraThreatLevel.CRITICAL
        }
        return mapping.get(security_level, InfraThreatLevel.MEDIUM)
    
    def get_comprehensive_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status from both layers"""
        ai_metrics = self.ai_security.get_security_metrics()
        infra_status = self.infra_security.get_security_status()
        
        return {
            "ai_security": ai_metrics,
            "infrastructure_security": infra_status,
            "active_sessions": len(self.active_contexts),
            "unified_threat_intelligence": {
                "correlations_tracked": len(self.threat_intelligence.threat_correlations),
                "incidents_logged": len(self.threat_intelligence.incident_history)
            },
            "integration_health": "healthy"
        }


# Example integration configuration
INTEGRATION_CONFIG = {
    "ai_security": {
        "content_security": {
            "enabled": True,
            "prompt_injection": {
                "enabled": True,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "similarity_threshold": 0.85
            }
        },
        "plugin_security": {
            "enabled": True,
            "supported_capabilities": [
                'file_read', 'file_write', 'network_request', 'code_execution'
            ]
        },
        "rate_limiting": {
            "request_limits": {
                "per_user": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000
                }
            }
        }
    },
    "infrastructure_security": {
        "config_path": "security_config.yaml",
        "master_key": None,  # Will be generated
        "max_sessions_per_vm": 10,
        "session_timeout": 3600,
        "allowed_container_images": [
            "python:3.11-slim",
            "node:18-alpine"
        ]
    }
}


def create_integrated_security_system(config: Dict[str, Any] = None) -> IntegratedSecurityOrchestrator:
    """Factory function to create integrated security system"""
    if config is None:
        config = INTEGRATION_CONFIG
    
    orchestrator = IntegratedSecurityOrchestrator(
        config["ai_security"],
        config["infrastructure_security"]
    )
    
    # Start infrastructure monitoring
    orchestrator.infra_security.start_monitoring()
    
    return orchestrator


if __name__ == "__main__":
    # Example usage
    security_system = create_integrated_security_system()
    
    print("Integrated Somnus Security System initialized")
    print("- AI Security Layer: Content filtering, prompt injection detection")
    print("- Infrastructure Layer: Network security, container isolation, API authentication")
    print("- Unified Threat Intelligence: Coordinated threat analysis and response")
    
    status = security_system.get_comprehensive_security_status()
    print(f"System Status: {status}")