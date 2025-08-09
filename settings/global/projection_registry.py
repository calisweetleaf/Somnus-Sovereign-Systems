"""
SOMNUS SYSTEMS - Canonical Projection Registry
Single Source of Truth for User Profile → AI Capability Projections

ARCHITECTURE PHILOSOPHY:
- One authoritative mapping table prevents drift across subsystems
- Testable, deterministic projections with hash verification
- Explicit scope defaults and conditional DSL rules
- Complete coverage of all profile fields with safe defaults

This registry ensures that user data transforms consistently into AI capabilities
across all subsystems while maintaining privacy and preventing information leakage.
"""

import re
import hashlib
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import only FieldMeta; align types with user_registry
from user_registry import FieldMeta


# Add lightweight shim enums (string-backed) to interoperate with user_registry.FieldMeta
class ExposureScope(str, Enum):
    SESSION = "session"
    ALWAYS = "always"
    # task:/agent: scopes are dynamic strings; rules may specify them directly as strings

class RedactionLevel(str, Enum):
    NONE = "none"
    COARSE = "coarse"
    MASKED = "masked"
    HIDDEN = "hidden"

class ProvenanceType(str, Enum):
    USER_INPUT = "user_input"
    IMPORT = "import"
    AI_OBSERVATION = "ai_observation"
    SYSTEM_DETECTED = "system_detected"


# ============================================================================
# PROJECTION RULE TYPES
# ============================================================================

@dataclass
class ProjectionRule:
    """Complete projection rule for a user profile field"""
    # Target mapping
    section: str  # "who", "work_env", "prefs", "capabilities", "context"
    key: str      # Target key in projection
    
    # DSL rules for conditional/complex mappings
    dsl_rules: List[str] = field(default_factory=list)
    
    # Default metadata
    meta: FieldMeta = field(default_factory=FieldMeta)
    
    # Transform function for complex mappings
    transform: Optional[Callable[[Any], Any]] = None
    
    # Validation rules
    required_scopes: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)

class ProjectionSection(str, Enum):
    """Valid projection sections"""
    WHO = "who"                    # Identity information
    WORK_ENV = "work_env"         # Technical environment
    PREFS = "prefs"               # User preferences
    CAPABILITIES = "capabilities"  # Abstract capabilities
    CONTEXT = "context"           # Situational context

class CapabilityFlag(str, Enum):
    """Canonical capability flags"""
    # Hardware capabilities
    GPU_ACCELERATION = "gpu_acceleration"
    HIGH_MEMORY_AVAILABLE = "high_memory_available"
    SSD_STORAGE = "ssd_storage"
    MULTIPLE_MONITORS = "multiple_monitors"
    HIGH_BANDWIDTH = "high_bandwidth"
    
    # Software capabilities
    CONTAINERIZATION = "containerization"
    VERSION_CONTROL = "version_control"
    CODE_EDITING = "code_editing"
    ML_FRAMEWORKS = "ml_frameworks"
    CLOUD_PLATFORMS = "cloud_platforms"
    
    # Development capabilities
    WEB_DEVELOPMENT = "web_development"
    MOBILE_DEVELOPMENT = "mobile_development"
    DATA_SCIENCE = "data_science"
    DEVOPS = "devops"
    SECURITY_TOOLS = "security_tools"
    
    # Domain knowledge
    AI_EXPERTISE = "ai_expertise"
    WEB3_KNOWLEDGE = "web3_knowledge"
    CYBERSECURITY = "cybersecurity"
    DATA_ANALYSIS = "data_analysis"
    SYSTEM_ADMINISTRATION = "system_administration"


# ============================================================================
# CANONICAL PROJECTION REGISTRY
# ============================================================================

PROJECTION_REGISTRY: Dict[str, ProjectionRule] = {
    
    # ========================================================================
    # IDENTITY PROJECTIONS
    # ========================================================================
    
    "identity.display_name": ProjectionRule(
        section=ProjectionSection.WHO,
        key="display_name",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,  # shim enum; value is a str for FieldMeta
            scope=ExposureScope.ALWAYS,
            provenance=ProvenanceType.USER_INPUT
        )
    ),
    
    "identity.pronouns": ProjectionRule(
        section=ProjectionSection.WHO,
        key="pronouns",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "identity.role": ProjectionRule(
        section=ProjectionSection.WHO,
        key="role",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "identity.expertise": ProjectionRule(
        section=ProjectionSection.CAPABILITIES,
        key="domain_knowledge",
        dsl_rules=[
            "identity.expertise ~ /python/i -> capabilities.flags.programming_python",
            "identity.expertise ~ /javascript/i -> capabilities.flags.programming_javascript",
            "identity.expertise ~ /ai|machine.learning|ml/i -> capabilities.flags.ai_expertise",
            "identity.expertise ~ /devops|infrastructure/i -> capabilities.flags.devops",
            "identity.expertise ~ /security|cybersecurity/i -> capabilities.flags.cybersecurity",
            "identity.expertise ~ /data.science|analytics/i -> capabilities.flags.data_analysis"
        ],
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "identity.timezone": ProjectionRule(
        section=ProjectionSection.CONTEXT,
        key="timezone",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "identity.location_context": ProjectionRule(
        section=ProjectionSection.CONTEXT,
        key="location_context",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.SESSION
        )
    ),
    
    # ========================================================================
    # TECHNICAL PROJECTIONS
    # ========================================================================
    
    "technical.primary_machine": ProjectionRule(
        section=ProjectionSection.CAPABILITIES,
        key="hardware_capabilities",
        dsl_rules=[
            "technical.primary_machine.ram_gb >= 16 -> capabilities.flags.high_memory_available",
            "technical.primary_machine.ram_gb >= 32 -> capabilities.flags.high_memory_workstation",
            "technical.primary_machine.gpu_model -> capabilities.flags.gpu_acceleration",
            "technical.primary_machine.gpu_model ~ /rtx|quadro|tesla/i -> capabilities.flags.ml_capable_gpu",
            "technical.primary_machine.storage_type ~ /ssd|nvme/i -> capabilities.flags.ssd_storage",
            "technical.primary_machine.cpu_cores >= 8 -> capabilities.flags.high_performance_cpu"
        ],
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.HIDDEN,  # Raw specs hidden, only capabilities exposed
            scope=ExposureScope.ALWAYS,
            project_as="capabilities.flags.gpu_acceleration,capabilities.flags.high_memory_available"
        )
    ),
    
    "technical.tools": ProjectionRule(
        section=ProjectionSection.CAPABILITIES,
        key="software_capabilities",
        dsl_rules=[
            "technical.tools ~ /docker|podman|container/i -> capabilities.flags.containerization",
            "technical.tools ~ /git|github|gitlab/i -> capabilities.flags.version_control",
            "technical.tools ~ /vscode|vim|emacs|intellij/i -> capabilities.flags.code_editing",
            "technical.tools ~ /tensorflow|pytorch|keras/i -> capabilities.flags.ml_frameworks",
            "technical.tools ~ /aws|gcp|azure|cloud/i -> capabilities.flags.cloud_platforms",
            "technical.tools ~ /react|vue|angular|node/i -> capabilities.flags.web_development",
            "technical.tools ~ /jupyter|pandas|numpy/i -> capabilities.flags.data_science",
            "technical.tools ~ /kubernetes|terraform|ansible/i -> capabilities.flags.devops",
            "technical.tools ~ /wireshark|nmap|burp/i -> capabilities.flags.security_tools"
        ],
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "technical.programming_languages": ProjectionRule(
        section=ProjectionSection.CAPABILITIES,
        key="programming_languages",
        dsl_rules=[
            "technical.programming_languages ~ /python/i -> capabilities.flags.programming_python",
            "technical.programming_languages ~ /javascript|typescript/i -> capabilities.flags.programming_javascript",
            "technical.programming_languages ~ /rust/i -> capabilities.flags.programming_rust",
            "technical.programming_languages ~ /go|golang/i -> capabilities.flags.programming_go",
            "technical.programming_languages ~ /c\+\+|cpp/i -> capabilities.flags.programming_cpp",
            "technical.programming_languages ~ /java/i -> capabilities.flags.programming_java"
        ],
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "technical.network": ProjectionRule(
        section=ProjectionSection.CAPABILITIES,
        key="network_capabilities",
        dsl_rules=[
            "technical.network.bandwidth_tier == 'high' -> capabilities.flags.high_bandwidth",
            "technical.network.bandwidth_tier == 'unlimited' -> capabilities.flags.unlimited_bandwidth",
            "technical.network.cloud_storage_access == true -> capabilities.flags.cloud_storage",
            "technical.network.offline_capable == true -> capabilities.flags.offline_capable"
        ],
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "technical.accessibility": ProjectionRule(
        section=ProjectionSection.WORK_ENV,
        key="accessibility_tools",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    # ========================================================================
    # WORKFLOW PROJECTIONS
    # ========================================================================
    
    "workflow.communication": ProjectionRule(
        section=ProjectionSection.PREFS,
        key="communication_style",
        dsl_rules=[
            "workflow.communication.style == 'technical' -> prefs.explanation_depth = 'detailed'",
            "workflow.communication.formality == 'formal' -> prefs.tone = 'professional'",
            "workflow.communication.depth == 'detailed' -> prefs.verbosity = 'comprehensive'"
        ],
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "workflow.doc_prefs": ProjectionRule(
        section=ProjectionSection.PREFS,
        key="documentation_style",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "workflow.peak_hours": ProjectionRule(
        section=ProjectionSection.CONTEXT,
        key="availability_patterns",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.SESSION
        )
    ),
    
    "workflow.current_projects": ProjectionRule(
        section=ProjectionSection.CONTEXT,
        key="active_projects",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.SESSION,
            expires_at=None  # Session-specific, auto-expires
        )
    ),
    
    # ========================================================================
    # LEARNING PROJECTIONS
    # ========================================================================
    
    "learning.styles": ProjectionRule(
        section=ProjectionSection.PREFS,
        key="learning_preferences",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "learning.depth": ProjectionRule(
        section=ProjectionSection.PREFS,
        key="explanation_depth",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "learning.feedback": ProjectionRule(
        section=ProjectionSection.PREFS,
        key="feedback_style",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "learning.code_commenting": ProjectionRule(
        section=ProjectionSection.PREFS,
        key="code_verbosity",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.NONE,
            scope="task:coding"
        )
    ),
    
    "learning.prior_knowledge": ProjectionRule(
        section=ProjectionSection.CAPABILITIES,
        key="knowledge_base",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.ALWAYS
        )
    ),
    
    "learning.learning_goals": ProjectionRule(
        section=ProjectionSection.CONTEXT,
        key="learning_objectives",
        meta=FieldMeta(
            expose_to_llm=True,
            redaction=RedactionLevel.COARSE,
            scope=ExposureScope.SESSION
        )
    )
}


# ============================================================================
# SCOPE PRECEDENCE SYSTEM
# ============================================================================

SCOPE_PRECEDENCE = [
    "task:",      # Highest priority - specific task contexts
    "agent:",     # Agent-specific projections
    "always",     # Always visible
    "session"     # Lowest priority - session-only
]

def get_scope_priority(scope: str) -> int:
    """Get scope priority (lower number = higher priority)"""
    for i, prefix in enumerate(SCOPE_PRECEDENCE):
        if scope.startswith(prefix) or scope == prefix.rstrip(':'):
            return i
    return len(SCOPE_PRECEDENCE)  # Unknown scopes get lowest priority

def resolve_scope_conflicts(field_scopes: List[str], requested_scope: str) -> Optional[str]:
    """Resolve scope conflicts using precedence rules"""
    applicable_scopes = []
    
    for scope in field_scopes:
        if scope_matches(scope, requested_scope):
            applicable_scopes.append(scope)
    
    if not applicable_scopes:
        return None
    
    # Return highest priority scope
    return min(applicable_scopes, key=get_scope_priority)

def scope_matches(field_scope: str, requested_scope: str) -> bool:
    """Check if field scope matches requested projection scope"""
    if field_scope == "always":
        return True
    
    if field_scope == requested_scope:
        return True
    
    # Handle wildcard matching
    if field_scope.endswith("/*"):
        base_scope = field_scope[:-2]
        if requested_scope.startswith(base_scope):
            return True
    
    # Session scope matches session or always requests
    if field_scope == "session" and requested_scope in ["session", "always"]:
        return True
    
    return False


# ============================================================================
# DSL RULE ENGINE
# ============================================================================

def evaluate_dsl_rule(rule: str, profile_data: Any) -> Optional[Dict[str, Any]]:
    """Evaluate DSL rule and return projection mapping if conditions are met"""
    try:
        # Parse rule: "condition -> target"
        if " -> " not in rule:
            return None
        
        condition, target = rule.split(" -> ", 1)
        condition = condition.strip()
        target = target.strip()
        
        # Evaluate condition
        if not evaluate_condition(condition, profile_data):
            return None
        
        # Parse target
        return parse_target(target)
        
    except Exception as e:
        print(f"DSL rule evaluation failed: {rule} - {e}")
        return None

def evaluate_condition(condition: str, profile_data: Any) -> bool:
    """Evaluate condition expression"""
    # Regex match: field ~ /pattern/flags
    regex_match = re.match(r'(.+?)\s*~\s*/(.+?)/(.*)', condition)
    if regex_match:
        field_path, pattern, flags = regex_match.groups()
        field_value = get_nested_value(profile_data, field_path.strip())
        
        if field_value is None:
            return False
        
        # Convert field value to string for regex matching
        field_str = str(field_value) if not isinstance(field_value, list) else " ".join(map(str, field_value))
        
        # Apply regex flags
        regex_flags = 0
        if 'i' in flags:
            regex_flags |= re.IGNORECASE
        
        return bool(re.search(pattern, field_str, regex_flags))
    
    # Comparison operators
    for op in [">=", "<=", "==", "!=", ">", "<"]:
        if op in condition:
            left, right = condition.split(op, 1)
            left_val = get_nested_value(profile_data, left.strip())
            right_val = parse_value(right.strip())
            
            if left_val is None:
                return False
            
            # Apply comparison
            if op == ">=":
                return left_val >= right_val
            elif op == "<=":
                return left_val <= right_val
            elif op == "==":
                return left_val == right_val
            elif op == "!=":
                return left_val != right_val
            elif op == ">":
                return left_val > right_val
            elif op == "<":
                return left_val < right_val
    
    # Simple field existence check
    field_value = get_nested_value(profile_data, condition.strip())
    return bool(field_value)

def parse_target(target: str) -> Dict[str, Any]:
    """Parse target expression into projection mapping"""
    # Handle assignment: "key = value"
    if " = " in target:
        key, value = target.split(" = ", 1)
        return {key.strip(): parse_value(value.strip())}
    
    # Handle simple flag: "capabilities.flags.gpu_acceleration"
    return {target: True}

def parse_value(value_str: str) -> Any:
    """Parse string value to appropriate Python type"""
    value_str = value_str.strip()
    
    # String literals
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    
    # Boolean literals
    if value_str.lower() == "true":
        return True
    elif value_str.lower() == "false":
        return False
    
    # Numeric literals
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass
    
    # Return as string if nothing else matches
    return value_str

def get_nested_value(obj: Any, path: str) -> Any:
    """Get nested value using dot notation"""
    try:
        parts = path.split('.')
        current = obj
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    except Exception:
        return None


# ============================================================================
# PROJECTION GENERATION
# ============================================================================

def generate_projection_from_registry(
    profile: Any,
    scope: str = "session",
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate projection using canonical registry rules"""
    
    projection = {
        "who": {},
        "work_env": {},
        "prefs": {},
        "capabilities": {
            "flags": {},
            "lists": {}
        },
        "context": {},
        "_meta": {
            "projection_id": f"proj_{hash(str(profile) + scope + str(agent_id)) % 100000:05d}",
            "scope": scope,
            "agent_id": agent_id,
            "registry_version": "1.0.0",
            "generated_at": "2025-08-09T14:30:00Z"
        }
    }
    
    # Apply all registry rules
    for field_path, rule in PROJECTION_REGISTRY.items():
        field_value = get_nested_value(profile, field_path)
        if field_value is None:
            continue
        
        # Check scope compatibility
        if not scope_matches(rule.meta.scope, scope):
            continue
        
        # Apply DSL rules first
        if rule.dsl_rules:
            for dsl_rule in rule.dsl_rules:
                dsl_result = evaluate_dsl_rule(dsl_rule, profile)
                if dsl_result:
                    apply_projection_mapping(projection, dsl_result)
        
        # Apply direct mapping
        if rule.section and rule.key:
            # Apply redaction
            redacted_value = apply_redaction(field_value, rule.meta.redaction)
            if redacted_value is not None:
                projection[rule.section][rule.key] = redacted_value
    
    return projection

def apply_projection_mapping(projection: Dict[str, Any], mapping: Dict[str, Any]) -> None:
    """Apply projection mapping to output structure"""
    for key, value in mapping.items():
        parts = key.split('.')
        current = projection
        
        # Navigate to parent
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        
        # Set value
        current[parts[-1]] = value

def apply_redaction(value: Any, redaction: Any) -> Any:
    """Apply redaction rules to field value. Supports both string and RedactionLevel."""
    # Normalize redaction to string value
    red = getattr(redaction, "value", redaction)
    if red == "none":
        return value
    elif red == "coarse":
        # Apply coarse filtering - preserve general structure
        if isinstance(value, dict):
            return {k: v for k, v in value.items() 
                   if not any(sensitive in k.lower() for sensitive in ['path', 'key', 'token', 'password'])}
        elif isinstance(value, list):
            return [item for item in value if isinstance(item, str) and 
                   not any(sensitive in item.lower() for sensitive in ['path', 'key', 'token', 'password'])]
        return value
    elif red == "masked":
        return "***"
    elif red == "hidden":
        return None
    
    return value


# ============================================================================
# REGISTRY VALIDATION AND TESTING
# ============================================================================

def compute_projection_hash(projection: Dict[str, Any]) -> str:
    """Compute deterministic hash of projection for validation"""
    # Remove metadata for hash computation
    clean_projection = {k: v for k, v in projection.items() if not k.startswith('_')}
    
    # Sort keys for deterministic serialization
    canonical_json = json.dumps(clean_projection, sort_keys=True, separators=(',', ':'))
    
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]

def validate_registry() -> List[str]:
    """Validate projection registry for completeness and consistency"""
    errors = []
    
    # Check for duplicate targets
    targets = {}
    for field_path, rule in PROJECTION_REGISTRY.items():
        if rule.section and rule.key:
            target = f"{rule.section}.{rule.key}"
            if target in targets:
                errors.append(f"Duplicate target {target} for fields {targets[target]} and {field_path}")
            else:
                targets[target] = field_path
    
    # Validate DSL rules
    for field_path, rule in PROJECTION_REGISTRY.items():
        for dsl_rule in rule.dsl_rules:
            if " -> " not in dsl_rule:
                errors.append(f"Invalid DSL rule in {field_path}: {dsl_rule}")
    
    # Check scope consistency
    for field_path, rule in PROJECTION_REGISTRY.items():
        if rule.meta.scope not in ["session", "always"] and not any(
            rule.meta.scope.startswith(prefix) for prefix in ["task:", "agent:"]
        ):
            errors.append(f"Invalid scope {rule.meta.scope} for field {field_path}")
    
    return errors

def test_projection_determinism(profile: Any, iterations: int = 5) -> bool:
    """Test that projections are deterministic"""
    first_hash = None
    
    for i in range(iterations):
        projection = generate_projection_from_registry(profile, "session")
        current_hash = compute_projection_hash(projection)
        
        if first_hash is None:
            first_hash = current_hash
        elif first_hash != current_hash:
            return False
    
    return True


# ============================================================================
# REGISTRY MANAGEMENT
# ============================================================================

def get_field_projection_rule(field_path: str) -> Optional[ProjectionRule]:
    """Get projection rule for specific field"""
    return PROJECTION_REGISTRY.get(field_path)

def update_field_projection_rule(field_path: str, rule: ProjectionRule) -> bool:
    """Update projection rule for field"""
    try:
        PROJECTION_REGISTRY[field_path] = rule
        return True
    except Exception:
        return False

def get_capability_fields() -> Dict[str, List[str]]:
    """Get mapping of capabilities to source fields"""
    capability_fields = {}
    
    for field_path, rule in PROJECTION_REGISTRY.items():
        if rule.section == ProjectionSection.CAPABILITIES:
            capability = rule.key
            if capability not in capability_fields:
                capability_fields[capability] = []
            capability_fields[capability].append(field_path)
    
    return capability_fields

def export_registry_schema() -> Dict[str, Any]:
    """Export registry as JSON schema for documentation"""
    schema = {
        "version": "1.0.0",
        "scope_precedence": SCOPE_PRECEDENCE,
        "capability_flags": [flag.value for flag in CapabilityFlag],
        "projection_sections": [section.value for section in ProjectionSection],
        "field_mappings": {}
    }
    
    for field_path, rule in PROJECTION_REGISTRY.items():
        schema["field_mappings"][field_path] = {
            "section": rule.section,
            "key": rule.key,
            "dsl_rules": rule.dsl_rules,
            # Normalize redaction to string for export
            "scope": getattr(rule.meta.scope, "value", rule.meta.scope),
            "redaction": getattr(rule.meta.redaction, "value", rule.meta.redaction),
            "expose_to_llm": rule.meta.expose_to_llm
        }
    
    return schema


if __name__ == "__main__":
    # Validate registry on import
    validation_errors = validate_registry()
    if validation_errors:
        print("Registry validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("✓ Projection registry validation passed")
    
    # Export schema for documentation
    schema = export_registry_schema()
    print(f"✓ Registry contains {len(schema['field_mappings'])} field mappings")
    print(f"✓ Registry supports {len(schema['capability_flags'])} capability flags")