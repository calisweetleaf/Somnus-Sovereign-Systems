#!/usr/bin/env python3
"""
Somnus Policy Management System
===============================

A centralized, industrial-grade system for managing, enforcing, and dynamically
adapting security policies across the Somnus Sovereign Systems architecture.

Core Features:
- Unified policy definition for network, process, file, and user activities.
- Versioned and auditable policy storage with YAML and SQLite backends.
- High-performance policy enforcement engine for real-time decision making.
- Seamless integration with Blue Team for enforcement and Purple Team for adaptation.
- API for programmatic policy management and updates.
"""

import asyncio
import ipaddress
import json
import logging
import re
import sqlite3
import time
import yaml
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

# --- Policy Data Models ---

class PolicyType(Enum):
    """Defines the scope of a security policy."""
    NETWORK = "network"
    PROCESS = "process"
    FILE_SYSTEM = "file_system"
    USER_ACCESS = "user_access"
    RESOURCE_USAGE = "resource_usage"
    DATA_EXFILTRATION = "data_exfiltration"
    AGENT_BEHAVIOR = "agent_behavior"

class PolicyAction(Enum):
    """Defines the action to take when a policy rule is matched."""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    ALERT = "alert"
    QUARANTINE = "quarantine"
    RATE_LIMIT = "rate_limit"
    REQUIRE_MFA = "require_mfa"
    AUDIT = "audit"

@dataclass
class PolicyCondition:
    """
    A single condition within a policy rule.
    e.g., "ip.source == '192.168.1.100'"
    """
    field: str  # e.g., 'ip.source', 'process.name', 'file.path'
    operator: str  # e.g., '==', '!=', 'in', 'not_in', 'matches', '>', '<'
    value: Any

@dataclass
class PolicyRule:
    """A single rule within a security policy."""
    rule_id: str
    description: str
    conditions: List[PolicyCondition]
    action: PolicyAction
    priority: int  # Lower number means higher priority
    enabled: bool = True

@dataclass
class SecurityPolicy:
    """A complete security policy for a specific domain."""
    policy_id: str
    name: str
    type: PolicyType
    version: int
    description: str
    rules: List[PolicyRule]
    enabled: bool = True
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Policy Storage ---

class PolicyDatabase:
    """Manages the storage and retrieval of policies in a SQLite database."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Main policy table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    description TEXT,
                    rules TEXT, -- JSON representation of rules
                    enabled BOOLEAN,
                    last_updated TEXT,
                    metadata TEXT -- JSON representation of metadata
                )
            """)
            # Policy version history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS policy_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    policy_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    policy_data TEXT, -- Full JSON of the policy at this version
                    change_reason TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_policy(self, policy: SecurityPolicy, change_reason: str = "System update") -> None:
        """Saves a policy to the database, creating a new version."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check for existing policy to increment version
            cursor.execute("SELECT version FROM policies WHERE policy_id = ?", (policy.policy_id,))
            result = cursor.fetchone()
            if result:
                policy.version = result[0] + 1
            else:
                policy.version = 1

            policy.last_updated = datetime.utcnow()
            
            rules_json = json.dumps([asdict(r) for r in policy.rules])
            metadata_json = json.dumps(policy.metadata)
            
            # Insert or replace the main policy record
            cursor.execute("""
                INSERT OR REPLACE INTO policies 
                (policy_id, name, type, version, description, rules, enabled, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                policy.policy_id, policy.name, policy.type.value, policy.version,
                policy.description, rules_json, policy.enabled,
                policy.last_updated.isoformat(), metadata_json
            ))

            # Add to history
            policy_data_json = json.dumps(asdict(policy), default=str)
            cursor.execute("""
                INSERT INTO policy_history (policy_id, version, policy_data, change_reason, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                policy.policy_id, policy.version, policy_data_json,
                change_reason, policy.last_updated.isoformat()
            ))
            
            conn.commit()
            logger.info(f"Saved policy '{policy.name}' (ID: {policy.policy_id}), version {policy.version}")

    def get_policy(self, policy_id: str, version: Optional[int] = None) -> Optional[SecurityPolicy]:
        """Retrieves a specific policy, optionally at a specific version."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if version is None:
                cursor.execute("SELECT * FROM policies WHERE policy_id = ?", (policy_id,))
                row = cursor.fetchone()
            else:
                cursor.execute("SELECT policy_data FROM policy_history WHERE policy_id = ? AND version = ?", (policy_id, version))
                history_row = cursor.fetchone()
                if history_row:
                    policy_data = json.loads(history_row[0])
                    # Reconstruct the row format for parsing
                    row = (
                        policy_data['policy_id'], policy_data['name'], policy_data['type'],
                        policy_data['version'], policy_data['description'], json.dumps(policy_data['rules']),
                        policy_data['enabled'], policy_data['last_updated'], json.dumps(policy_data['metadata'])
                    )
                else:
                    row = None

            if not row:
                return None
            
            return self._row_to_policy(row)

    def get_all_policies(self, policy_type: Optional[PolicyType] = None) -> List[SecurityPolicy]:
        """Retrieves all active policies, optionally filtered by type."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if policy_type:
                cursor.execute("SELECT * FROM policies WHERE type = ? AND enabled = 1", (policy_type.value,))
            else:
                cursor.execute("SELECT * FROM policies WHERE enabled = 1")
            
            rows = cursor.fetchall()
            return [self._row_to_policy(row) for row in rows]

    def _row_to_policy(self, row: tuple) -> SecurityPolicy:
        """Converts a database row to a SecurityPolicy object."""
        rules_data = json.loads(row[5])
        rules = [
            PolicyRule(
                rule_id=r['rule_id'],
                description=r['description'],
                conditions=[PolicyCondition(**c) for c in r['conditions']],
                action=PolicyAction(r['action']),
                priority=r['priority'],
                enabled=r['enabled']
            ) for r in rules_data
        ]
        
        policy_type_str = row[2]
        if isinstance(policy_type_str, PolicyType):
            policy_type = policy_type_str
        else:
            policy_type = PolicyType(policy_type_str)

        return SecurityPolicy(
            policy_id=row[0],
            name=row[1],
            type=policy_type,
            version=row[3],
            description=row[4],
            rules=rules,
            enabled=bool(row[6]),
            last_updated=datetime.fromisoformat(row[7]),
            metadata=json.loads(row[8])
        )

# --- Policy Enforcement Engine ---

class PolicyEnforcer:
    """Evaluates context against policies to make decisions."""

    def __init__(self, policy_db: PolicyDatabase):
        self.policy_db = policy_db
        self.policy_cache: Dict[PolicyType, List[SecurityPolicy]] = {}
        self.last_cache_update: float = 0.0
        self.cache_ttl: int = 60  # seconds

    async def _refresh_cache(self):
        """Refreshes the policy cache from the database."""
        if time.time() - self.last_cache_update > self.cache_ttl:
            logger.debug("Refreshing policy cache.")
            self.policy_cache.clear()
            for p_type in PolicyType:
                self.policy_cache[p_type] = self.policy_db.get_all_policies(p_type)
                # Sort rules by priority within each policy
                for policy in self.policy_cache[p_type]:
                    policy.rules.sort(key=lambda r: r.priority)
            self.last_cache_update = time.time()

    async def check(self, policy_type: PolicyType, context: Dict[str, Any]) -> Tuple[PolicyAction, Optional[PolicyRule]]:
        """
        Checks a given context against all policies of a certain type.

        Args:
            policy_type: The type of policy to check against (e.g., NETWORK).
            context: A dictionary of the current context (e.g., {'ip.source': '...', 'port.dest': ...}).

        Returns:
            A tuple of (PolicyAction, MatchedPolicyRule).
            Defaults to ALLOW if no rule is matched.
        """
        await self._refresh_cache()
        
        policies = self.policy_cache.get(policy_type, [])
        
        for policy in policies:
            if not policy.enabled:
                continue
            
            for rule in policy.rules:
                if not rule.enabled:
                    continue
                
                if self._evaluate_rule(rule, context):
                    logger.debug(f"Context matched rule '{rule.rule_id}' in policy '{policy.policy_id}'. Action: {rule.action.value}")
                    return rule.action, rule
        
        # Default action if no rule matches
        return PolicyAction.ALLOW, None

    def _evaluate_rule(self, rule: PolicyRule, context: Dict[str, Any]) -> bool:
        """Evaluates if a context matches all conditions of a rule."""
        for condition in rule.conditions:
            # Pass the whole context to the condition evaluator
            if not self._evaluate_condition(condition, context):
                return False  # One condition fails, so the rule doesn't match
        
        return True # All conditions passed

    def _evaluate_condition(self, condition: PolicyCondition, context: Dict[str, Any]) -> bool:
        """Evaluates a single condition against the full context."""
        op = condition.operator
        val = condition.value
        field = condition.field

        # Special handling for subnet checks which need to look at the 'ip.source' field
        if op in ['in_subnet', 'not_in_subnet']:
            # Use 'ip.source' as the default IP field for subnet checks
            ip_addr_str = context.get('ip.source')
            if not ip_addr_str:
                return False
            try:
                ip_addr = ipaddress.ip_address(ip_addr_str)
                # The 'value' for a subnet check should be a list of CIDR strings
                is_in_subnet = any(ip_addr in ipaddress.ip_network(subnet) for subnet in val)
                return is_in_subnet if op == 'in_subnet' else not is_in_subnet
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid IP address or subnet format for condition {condition}: {e}")
                return False

        context_value = context.get(field)
        if context_value is None:
            return False  # Condition field not in context, so no match

        try:
            if op == '==': return context_value == val
            if op == '!=': return context_value != val
            if op == 'in': return context_value in val
            if op == 'not_in': return context_value not in val
            if op == '>': return float(context_value) > float(val)
            if op == '<': return float(context_value) < float(val)
            if op == '>=': return float(context_value) >= float(val)
            if op == '<=': return float(context_value) <= float(val)
            if op == 'matches': return bool(re.search(val, str(context_value)))
            if op == 'not_matches': return not bool(re.search(val, str(context_value)))
            if op == 'starts_with': return str(context_value).startswith(val)
            if op == 'ends_with': return str(context_value).endswith(val)
        except (ValueError, TypeError) as e:
            logger.warning(f"Type error during condition evaluation for {condition} with value {context_value}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Could not evaluate condition {condition} with value {context_value}: {e}")
            return False
            
        logger.warning(f"Unsupported operator '{op}' in condition.")
        return False

# --- Main Policy Management System ---

class PolicyManagementSystem:
    """The main interface for managing and interacting with security policies."""

    def __init__(self, db_path: str, default_policy_dir: Optional[str] = None):
        self.db = PolicyDatabase(db_path)
        self.enforcer = PolicyEnforcer(self.db)
        if default_policy_dir:
            self.load_default_policies(default_policy_dir)

    def load_default_policies(self, policy_dir: str):
        """Loads default policies from a directory of YAML files."""
        logger.info(f"Loading default policies from '{policy_dir}'...")
        policy_path = Path(policy_dir)
        if not policy_path.exists():
            logger.warning(f"Default policy directory not found: {policy_dir}")
            return

        for yaml_file in policy_path.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    policy_data = yaml.safe_load(f)
                
                rules = [
                    PolicyRule(
                        rule_id=r.pop('rule_id'),
                        description=r.pop('description'),
                        conditions=[PolicyCondition(**c) for c in r.pop('conditions')],
                        action=PolicyAction(r.pop('action')),
                        priority=r.pop('priority'),
                        enabled=r.pop('enabled', True)
                    ) for r in policy_data.pop('rules')
                ]

                policy = SecurityPolicy(
                    policy_id=policy_data.pop('policy_id'),
                    name=policy_data.pop('name'),
                    type=PolicyType(policy_data.pop('type')),
                    description=policy_data.pop('description'),
                    rules=rules,
                    enabled=policy_data.pop('enabled', True),
                    metadata=policy_data,  # Remaining fields go to metadata
                    version=0 # Will be set on save
                )
                
                # Save only if it's a new policy or a higher version (not applicable here, just save)
                existing = self.db.get_policy(policy.policy_id)
                if not existing:
                    self.db.save_policy(policy, change_reason="Initial load from default policies.")

            except Exception as e:
                logger.error(f"Failed to load policy from {yaml_file}: {e}", exc_info=True)

    async def get_policy(self, policy_id: str, version: Optional[int] = None) -> Optional[SecurityPolicy]:
        return self.db.get_policy(policy_id, version)

    async def get_all_policies(self, policy_type: Optional[PolicyType] = None) -> List[SecurityPolicy]:
        return self.db.get_all_policies(policy_type)

    async def update_policy(self, policy_id: str, updates: Dict[str, Any], change_reason: str) -> Optional[SecurityPolicy]:
        """Updates a policy with new values and saves it as a new version."""
        policy = await self.get_policy(policy_id)
        if not policy:
            logger.error(f"Cannot update non-existent policy: {policy_id}")
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(policy, key):
                # Handle special cases like enums
                if key == 'type':
                    setattr(policy, key, PolicyType(value))
                elif key == 'rules':
                    # Full replacement of rules
                    new_rules = [PolicyRule(**r) for r in value]
                    setattr(policy, key, new_rules)
                else:
                    setattr(policy, key, value)
        
        self.db.save_policy(policy, change_reason=change_reason)
        return policy

    async def check_compliance(self, policy_type: PolicyType, context: Dict[str, Any]) -> Tuple[PolicyAction, Optional[PolicyRule]]:
        """Provides a simple interface to the enforcer."""
        return await self.enforcer.check(policy_type, context)

# --- Example Usage ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a directory for default policies
    default_policies_dir = Path("./default_policies")
    default_policies_dir.mkdir(exist_ok=True)

    # Create a sample default network policy file
    sample_network_policy = {
        'policy_id': 'net_base_01',
        'name': 'Basic Network Policy',
        'type': 'network',
        'description': 'Default rules for inbound and outbound traffic.',
        'rules': [
            {
                'rule_id': 'allow_ssh_internal',
                'description': 'Allow SSH from internal management network.',
                'conditions': [
                    # This now uses the special 'in_subnet' operator
                    {'field': 'ip.source', 'operator': 'in_subnet', 'value': ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16']},
                    {'field': 'port.dest', 'operator': '==', 'value': 22},
                    {'field': 'protocol', 'operator': '==', 'value': 'TCP'}
                ],
                'action': 'allow',
                'priority': 10,
                'enabled': True
            },
            {
                'rule_id': 'deny_all_ssh_external',
                'description': 'Deny all external SSH access.',
                'conditions': [
                    {'field': 'port.dest', 'operator': '==', 'value': 22}
                ],
                'action': 'deny',
                'priority': 100,
                'enabled': True
            },
            {
                'rule_id': 'log_http_traffic',
                'description': 'Log all outbound HTTP traffic.',
                'conditions': [
                    {'field': 'port.dest', 'operator': '==', 'value': 80}
                ],
                'action': 'log',
                'priority': 500,
                'enabled': True
            }
        ]
    }
    with open(default_policies_dir / "network.yaml", 'w') as f:
        yaml.dump(sample_network_policy, f)

    # Initialize the system
    pms = PolicyManagementSystem(db_path="policies.db", default_policy_dir=str(default_policies_dir))
    
    # --- Enforcement Example ---
    print("\n--- Enforcement Check ---")
    ssh_context_internal = {
        'ip.source': '10.0.1.5',
        'port.dest': 22,
        'protocol': 'TCP'
    }
    action, rule = await pms.check_compliance(PolicyType.NETWORK, ssh_context_internal)
    print(f"Internal SSH check (from {ssh_context_internal['ip.source']}) -> Action: {action.value}, Matched Rule: {rule.rule_id if rule else 'None'}")

    ssh_context_external = {
        'ip.source': '8.8.8.8',
        'port.dest': 22,
        'protocol': 'TCP'
    }
    action, rule = await pms.check_compliance(PolicyType.NETWORK, ssh_context_external)
    print(f"External SSH check (from {ssh_context_external['ip.source']}) -> Action: {action.value}, Matched Rule: {rule.rule_id if rule else 'None'}")

    # --- Management Example ---
    print("\n--- Policy Management ---")
    policy = await pms.get_policy('net_base_01')
    
    if policy:
        print(f"Retrieved policy '{policy.name}' version {policy.version}")

        # Update the policy
        new_rules_data = [asdict(r) for r in policy.rules]
        new_rules_data.append({
            'rule_id': 'deny_telnet',
            'description': 'Explicitly deny all telnet traffic.',
            'conditions': [{'field': 'port.dest', 'operator': '==', 'value': 23}],
            'action': 'deny',
            'priority': 99,
            'enabled': True
        })

        updated_policy = await pms.update_policy(
            policy_id='net_base_01',
            updates={'rules': new_rules_data},
            change_reason="Added rule to block Telnet."
        )

        if updated_policy:
            print(f"Policy updated to version {updated_policy.version}. New rule count: {len(updated_policy.rules)}")
        else:
            print("Failed to update policy.")
    else:
        print("Could not retrieve policy 'net_base_01' for update.")

if __name__ == "__main__":
    asyncio.run(main())
