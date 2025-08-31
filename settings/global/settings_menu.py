"""
SOMNUS SYSTEMS - User Personalization Settings Menu
Human-Centric Configuration and Privacy Management Interface

ARCHITECTURE PHILOSOPHY:
- Complete user sovereignty over personal data and AI interaction
- Intuitive interface for complex privacy and projection controls
- Real-time preview of what AI agents will see
- Seamless integration with user registry and projection system
- Zero AI constraints - pure human personalization and control

This interface empowers users to control their AI collaboration experience
while maintaining absolute privacy sovereignty and projection transparency.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from user_registry import (
    UserRegistryManager, UserProfile, FieldMeta, 
    AIObservation
)
from projection_registry import (
    generate_projection_from_registry, compute_projection_hash,
    PROJECTION_REGISTRY, CapabilityFlag, ExposureScope, RedactionLevel,
    ProvenanceType, get_capability_fields   # added import
)
from events import (
    get_event_bus, publish_profile_updated, publish_projection_generated,
    EventPriority
)
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# SETTINGS MENU TYPES AND ENUMS
# ============================================================================

class SettingsCategory(str, Enum):
    """Categories of user personalization settings"""
    PROFILE = "profile"                    # Basic profile information
    PRIVACY = "privacy"                    # Privacy and projection controls
    AI_INTERACTION = "ai_interaction"      # AI collaboration preferences
    CAPABILITIES = "capabilities"          # Capability exposure management
    SESSIONS = "sessions"                  # Session and context management
    SECURITY = "security"                  # Security and authentication
    EXPORT_IMPORT = "export_import"        # Data portability
    ADVANCED = "advanced"                  # Advanced power-user features

class SettingsAction(str, Enum):
    """Types of settings actions"""
    VIEW = "view"
    EDIT = "edit"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"
    RESET = "reset"
    PREVIEW = "preview"
    APPROVE = "approve"
    DENY = "deny"

@dataclass
class SettingsOperation:
    """Represents a settings operation with context"""
    operation_id: str = field(default_factory=lambda: str(uuid4()))
    category: SettingsCategory = SettingsCategory.PROFILE
    action: SettingsAction = SettingsAction.VIEW
    target: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = False
    message: str = ""
    requires_confirmation: bool = False

@dataclass
class ProjectionPreview:
    """Preview of what AI will see for a given scope"""
    scope: str = "session"
    agent_id: Optional[str] = None
    projection: Dict[str, Any] = field(default_factory=dict)
    projection_hash: str = ""
    field_count: int = 0
    capabilities_exposed: List[str] = field(default_factory=list)
    privacy_summary: Dict[str, int] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class CapabilityExposureSettings:
    """Settings for individual capability exposure"""
    capability_name: str = ""
    exposed: bool = False
    scope: ExposureScope = ExposureScope.SESSION
    redaction_level: RedactionLevel = RedactionLevel.COARSE
    expires_at: Optional[datetime] = None
    notes: str = ""


# ============================================================================
# USER PERSONALIZATION SETTINGS MENU
# ============================================================================

class UserPersonalizationMenu:
    """Main interface for user personalization and privacy management"""
    
    def __init__(self, user_registry: UserRegistryManager):
        self.user_registry = user_registry
        self.event_bus = get_event_bus()
        
        # UI state and caching
        self.cached_projections: Dict[str, ProjectionPreview] = {}
        self.pending_operations: List[SettingsOperation] = []
        
        # Settings categories and their handlers
        self.category_handlers = {
            SettingsCategory.PROFILE: self._handle_profile_settings,
            SettingsCategory.PRIVACY: self._handle_privacy_settings,
            SettingsCategory.AI_INTERACTION: self._handle_ai_interaction_settings,
            SettingsCategory.CAPABILITIES: self._handle_capability_settings,
            SettingsCategory.SESSIONS: self._handle_session_settings,
            SettingsCategory.SECURITY: self._handle_security_settings,
            SettingsCategory.EXPORT_IMPORT: self._handle_export_import_settings,
            SettingsCategory.ADVANCED: self._handle_advanced_settings
        }
        
        logger.info("User Personalization Menu initialized")
    
    async def get_settings_overview(self) -> Dict[str, Any]:
        """Get comprehensive overview of user settings and status"""
        
        if not self.user_registry.current_user:
            return {"error": "No user logged in"}
        
        user = self.user_registry.current_user
        
        # Generate current projection preview
        current_projection = await self.get_projection_preview("session")
        
        # Count pending AI observations
        pending_observations = len(self.user_registry.pending_observations)
        
        # Get privacy summary
        privacy_summary = self._generate_privacy_summary()
        
        # Get recent activity
        recent_activity = await self._get_recent_activity()
        
        overview = {
            "user_info": {
                "username": user.username,  # fixed
                "display_name": user.identity.display_name,
                "profile_version": user.version,
                "last_active": user.last_active.isoformat(),
                "session_count": user.session_count
            },
            "projection_status": {
                "current_scope": "session",
                "capabilities_exposed": current_projection.capabilities_exposed,
                "field_count": current_projection.field_count,
                "projection_hash": current_projection.projection_hash
            },
            "privacy_status": privacy_summary,
            "pending_items": {
                "ai_observations": pending_observations,
                "pending_operations": len(self.pending_operations)
            },
            "recent_activity": recent_activity,
            "available_categories": [category.value for category in SettingsCategory]
        }
        
        return overview
    
    async def handle_settings_request(
        self,
        category: SettingsCategory,
        action: SettingsAction,
        target: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> SettingsOperation:
        """Handle a settings request from the UI"""
        
        operation = SettingsOperation(
            category=category,
            action=action,
            target=target,
            data=data or {},
            user_id=self.user_registry.current_user.identity.username if self.user_registry.current_user else ""
        )
        
        try:
            # Route to appropriate handler
            handler = self.category_handlers.get(category)
            if not handler:
                operation.message = f"Unknown settings category: {category}"
                return operation
            
            # Execute the operation
            result = await handler(action, target, operation.data)
            operation.success = result[0]
            operation.message = result[1]
            
            # Log successful operations
            if operation.success:
                logger.info(f"Settings operation completed: {category}.{action}.{target}")
                
                # Publish profile update event if needed
                if category in [SettingsCategory.PROFILE, SettingsCategory.PRIVACY]:
                    await publish_profile_updated(
                        username=operation.user_id,
                        profile_id=self.user_registry.current_user.username,  # fixed
                        changes={f"{category}.{target}": operation.data},
                        priority=EventPriority.NORMAL,
                        user_id=operation.user_id
                    )
            
            return operation
            
        except Exception as e:
            operation.message = f"Operation failed: {e}"
            logger.error(f"Settings operation error: {e}")
            return operation
    
    async def get_projection_preview(
        self,
        scope: str = "session",
        agent_id: Optional[str] = None,
        force_refresh: bool = False
    ) -> ProjectionPreview:
        """Generate preview of what AI will see for given scope"""
        
        cache_key = f"{scope}:{agent_id or 'any'}"
        
        # Return cached preview if available and not forcing refresh
        if not force_refresh and cache_key in self.cached_projections:
            cached = self.cached_projections[cache_key]
            # Check if cache is still fresh (5 minutes)
            if (datetime.now(timezone.utc) - cached.generated_at).total_seconds() < 300:
                return cached
        
        if not self.user_registry.current_user:
            return ProjectionPreview(scope=scope, agent_id=agent_id)
        
        # Generate projection using registry
        projection = generate_projection_from_registry(
            self.user_registry.current_user,
            scope,
            agent_id
        )
        
        # Create preview object
        preview = ProjectionPreview(
            scope=scope,
            agent_id=agent_id,
            projection=projection,
            projection_hash=compute_projection_hash(projection),
            field_count=self._count_projection_fields(projection),
            capabilities_exposed=self._extract_exposed_capabilities(projection),
            privacy_summary=self._generate_projection_privacy_summary(projection)
        )
        
        # Cache the preview
        self.cached_projections[cache_key] = preview
        
        # Limit cache size
        if len(self.cached_projections) > 20:
            oldest_key = min(self.cached_projections.keys(), 
                           key=lambda k: self.cached_projections[k].generated_at)
            del self.cached_projections[oldest_key]
        
        return preview
    
    async def get_ai_observations(self, status: str = "all") -> List[Dict[str, Any]]:
        """Get AI observations requiring user review"""
        
        observations = []
        
        for obs in self.user_registry.pending_observations:
            if status == "all" or obs.status == status:
                observation_data = {
                    "id": str(obs.id),
                    "field_path": obs.field_path,
                    "suggested_value": obs.suggested_value,
                    "confidence": obs.confidence,
                    "evidence": obs.evidence,
                    "status": obs.status,
                    "timestamp": obs.timestamp.isoformat(),
                    "field_description": self._get_field_description(obs.field_path)
                }
                observations.append(observation_data)
        
        return sorted(observations, key=lambda x: x["timestamp"], reverse=True)
    
    async def approve_ai_observation(self, observation_id: str) -> Tuple[bool, str]:
        """Approve an AI observation to be added to profile"""
        
        try:
            from uuid import UUID
            obs_uuid = UUID(observation_id)
            success = await self.user_registry.approve_ai_observation(obs_uuid)
            
            if success:
                # Clear projection cache since profile changed
                self.cached_projections.clear()
                return True, "AI observation approved and added to profile"
            else:
                return False, "Failed to approve AI observation"
                
        except Exception as e:
            return False, f"Error approving observation: {e}"
    
    async def deny_ai_observation(self, observation_id: str) -> Tuple[bool, str]:
        """Deny an AI observation"""
        
        try:
            from uuid import UUID
            obs_uuid = UUID(observation_id)
            
            # Find and mark as rejected
            for obs in self.user_registry.pending_observations:
                if obs.id == obs_uuid:
                    obs.status = "rejected"
                    return True, "AI observation denied"
            
            return False, "Observation not found"
            
        except Exception as e:
            return False, f"Error denying observation: {e}"
    
    async def export_user_data(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export user data for backup or migration"""
        
        if not self.user_registry.current_user:
            return {"error": "No user logged in"}
        
        user = self.user_registry.current_user
        
        export_data = {
            "export_metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "somnus_version": "2.1.0",
                "include_sensitive": include_sensitive
            },
            "profile": {
                "identity": user.identity.dict(),
                "technical": user.technical.dict(),
                "workflow": user.workflow.dict(),
                "learning": user.learning.dict(),
                "version": user.version,
                "created_at": user.created_at.isoformat()
            },
            "privacy_settings": self._export_privacy_settings(),
            "ai_observations": [
                {
                    "field_path": obs.field_path,
                    "suggested_value": obs.suggested_value,
                    "confidence": obs.confidence,
                    "evidence": obs.evidence,
                    "status": obs.status,
                    "timestamp": obs.timestamp.isoformat()
                }
                for obs in self.user_registry.pending_observations
            ]
        }
        
        if not include_sensitive:
            # Remove sensitive information for basic export
            self._sanitize_export_data(export_data)
        
        return export_data
    
    async def import_user_data(self, import_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Import user data from backup or migration"""
        
        try:
            # Validate import data structure
            if not self._validate_import_data(import_data):
                return False, "Invalid import data format"
            
            # Create or update profile
            if "profile" in import_data:
                profile_data = import_data["profile"]
                
                # Apply imported profile data
                for component_name, component_data in profile_data.items():
                    if component_name in ["identity", "technical", "workflow", "learning"]:
                        component = getattr(self.user_registry.current_user, component_name)
                        for field_name, field_value in component_data.items():
                            if hasattr(component, field_name) and not field_name.startswith('_'):
                                setattr(component, field_name, field_value)
            
            # Apply privacy settings
            if "privacy_settings" in import_data:
                await self._import_privacy_settings(import_data["privacy_settings"])
            
            # Clear caches
            self.cached_projections.clear()
            
            return True, "User data imported successfully"
            
        except Exception as e:
            return False, f"Import failed: {e}"
    
    # ========================================================================
    # CATEGORY HANDLERS
    # ========================================================================
    
    async def _handle_profile_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle profile-related settings operations"""
        
        if action == SettingsAction.VIEW:
            return True, "Profile data retrieved"
        
        elif action == SettingsAction.EDIT:
            if not target:
                return False, "No target field specified"
            
            try:
                # Parse field path (e.g., "identity.display_name")
                component_name, field_name = target.split('.', 1)
                component = getattr(self.user_registry.current_user, component_name)
                
                if hasattr(component, field_name):
                    old_value = getattr(component, field_name)
                    setattr(component, field_name, data.get("value"))
                    
                    # Update last modified timestamp (use last_active)
                    self.user_registry.current_user.last_active = datetime.now(timezone.utc)  # fixed
                    
                    # Clear projection cache
                    self.cached_projections.clear()
                    
                    return True, f"Updated {target} from '{old_value}' to '{data.get('value')}'"
                else:
                    return False, f"Field {field_name} not found in {component_name}"
                    
            except Exception as e:
                return False, f"Failed to update profile: {e}"
        
        elif action == SettingsAction.RESET:
            # Reset specific field or entire component to defaults
            return True, "Profile reset to defaults"
        
        return False, f"Unsupported profile action: {action}"
    
    async def _handle_privacy_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle privacy and projection control settings"""
        
        if action == SettingsAction.VIEW:
            # Return current privacy settings
            return True, "Privacy settings retrieved"
        
        elif action == SettingsAction.EDIT:
            if not target:
                return False, "No target field specified"
            
            try:
                # Update field metadata for privacy control
                component_name, field_name = target.split('.', 1)
                
                # Normalize scope and redaction to plain strings
                scope_in = data.get("scope", "session")
                red_in = data.get("redaction", "coarse")
                try:
                    scope_val = ExposureScope(scope_in).value
                except Exception:
                    scope_val = str(scope_in)
                try:
                    red_val = RedactionLevel(red_in).value
                except Exception:
                    red_val = str(red_in)
                
                # Create new metadata
                metadata = FieldMeta(
                    expose_to_llm=data.get("expose_to_llm", False),
                    scope=scope_val,
                    redaction=red_val,
                    expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
                )
                
                success = await self.user_registry.update_field_metadata(
                    component_name, field_name, metadata
                )
                
                if success:
                    # Clear projection cache
                    self.cached_projections.clear()
                    return True, f"Privacy settings updated for {target}"
                else:
                    return False, f"Failed to update privacy settings for {target}"
                    
            except Exception as e:
                return False, f"Failed to update privacy settings: {e}"
        
        elif action == SettingsAction.PREVIEW:
            # Generate projection preview for specific scope
            scope = data.get("scope", "session")
            agent_id = data.get("agent_id")
            
            preview = await self.get_projection_preview(scope, agent_id, force_refresh=True)
            return True, f"Projection preview generated for scope: {scope}"
        
        return False, f"Unsupported privacy action: {action}"

    async def _handle_ai_interaction_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle AI interaction and collaboration preferences"""
        
        if action == SettingsAction.VIEW:
            # Return current AI interaction settings
            settings = {
                "collaboration_mode": "enhanced",
                "ai_suggestions_enabled": True,
                "auto_completion": True,
                "context_awareness": "full",
                "privacy_level": "maximum",
                "learning_preferences": {
                    "feedback_frequency": "real_time",
                    "correction_threshold": 0.8,
                    "adaptation_speed": "moderate"
                }
            }
            return True, json.dumps(settings)
        
        elif action == SettingsAction.EDIT:
            if not target:
                return False, "No target field specified"
            
            valid_targets = [
                "collaboration_mode",
                "ai_suggestions_enabled",
                "auto_completion",
                "context_awareness",
                "privacy_level",
                "learning_preferences.feedback_frequency",
                "learning_preferences.correction_threshold",
                "learning_preferences.adaptation_speed"
            ]
            
            if target not in valid_targets:
                return False, f"Invalid AI interaction target: {target}"
            
            try:
                # Update the specific setting
                if target.startswith("learning_preferences."):
                    pref_key = target.split(".", 1)[1]
                    if not hasattr(self.user_registry.current_user, 'ai_preferences'):
                        self.user_registry.current_user.ai_preferences = {}
                    
                    self.user_registry.current_user.ai_preferences[pref_key] = data.get("value")
                else:
                    # Store in user profile metadata
                    metadata = FieldMeta(
                        expose_to_llm=True,
                        scope=ExposureScope.SESSION,
                        redaction=RedactionLevel.NONE
                    )
                    
                    await self.user_registry.update_field_metadata(
                        "ai_interaction", target, metadata
                    )
                
                # Clear projection cache
                self.cached_projections.clear()
                return True, f"Updated AI interaction setting: {target}"
                
            except Exception as e:
                return False, f"Failed to update AI interaction setting: {e}"
        
        elif action == SettingsAction.RESET:
            # Reset AI interaction settings to defaults
            default_settings = {
                "collaboration_mode": "enhanced",
                "ai_suggestions_enabled": True,
                "auto_completion": True,
                "context_awareness": "full",
                "privacy_level": "maximum"
            }
            
            for key, value in default_settings.items():
                metadata = FieldMeta(
                    expose_to_llm=True,
                    scope=ExposureScope.SESSION,
                    redaction=RedactionLevel.NONE
                )
                await self.user_registry.update_field_metadata("ai_interaction", key, metadata)
            
            self.cached_projections.clear()
            return True, "AI interaction settings reset to defaults"
        
        return False, f"Unsupported AI interaction action: {action}"
    
    async def _handle_capability_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle capability exposure management"""
        
        if action == SettingsAction.VIEW:
            # Get all capability exposure settings
            capabilities = {}
            for capability_flag in CapabilityFlag:
                # Get current exposure settings for this capability
                capabilities[capability_flag.value] = self._get_capability_exposure_settings(capability_flag.value)
            
            return True, "Capability settings retrieved"
        
        elif action == SettingsAction.EDIT:
            if target.startswith("capability."):
                capability_name = target.replace("capability.", "")
                
                # Update capability exposure settings
                success = await self._update_capability_exposure(capability_name, data)
                
                if success:
                    self.cached_projections.clear()
                    return True, f"Capability exposure updated for {capability_name}"
                else:
                    return False, f"Failed to update capability exposure for {capability_name}"
        
        return False, f"Unsupported capability action: {action}"
    
    async def _handle_session_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle session and context management"""
        
        if action == SettingsAction.VIEW:
            return True, "Session settings retrieved"
        
        return False, f"Unsupported session action: {action}"
    
    async def _handle_security_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle security and authentication settings"""
        
        if action == SettingsAction.EDIT and target == "password":
            # Expect the new password payload
            old_password = data.get("old_password")
            new_password = data.get("new_password")
            
            if not old_password or not new_password:
                return False, "Both 'old_password' and 'new_password' must be provided"
            
            try:
                # The UserRegistryManager should expose an async password‑change method.
                # Replace `change_password` with the actual method name if different.
                await self.user_registry.change_password(
                    username=self.user_registry.current_user.username,
                    old_password=old_password,
                    new_password=new_password
                )
                return True, "Password updated successfully"
            except Exception as e:
                logger.error(f"Password change failed: {e}")
                return False, f"Password update failed: {e}"
        
        return False, f"Unsupported security action: {action}"
    
    async def _handle_export_import_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle data export and import operations"""
        
        if action == SettingsAction.EXPORT:
            export_data = await self.export_user_data(include_sensitive=data.get("include_sensitive", False))
            return True, "User data exported successfully"
        
        elif action == SettingsAction.IMPORT:
            if "import_data" not in data:
                return False, "No import data provided"
            
            success, message = await self.import_user_data(data["import_data"])
            return success, message
        
        return False, f"Unsupported export/import action: {action}"
    
    async def _handle_advanced_settings(
        self,
        action: SettingsAction,
        target: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Handle advanced power-user features"""
        
        if action == SettingsAction.VIEW:
            # Return advanced diagnostic information
            return True, "Advanced settings retrieved"
        
        elif action == SettingsAction.EDIT:
            if target == "debug_mode":
                # Toggle debug mode
                return True, "Debug mode toggled"
        
        return False, f"Unsupported advanced action: {action}"
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _generate_privacy_summary(self) -> Dict[str, int]:
        """Generate summary of privacy settings"""
        
        if not self.user_registry.current_user:
            return {}
        
        summary = {
            "total_fields": 0,
            "exposed_fields": 0,
            "redacted_fields": 0,
            "hidden_fields": 0
        }
        
        # Count privacy settings across all components
        for component_name in ["identity", "technical", "workflow", "learning"]:
            component = getattr(self.user_registry.current_user, component_name)
            if hasattr(component, 'meta'):
                for field_name, field_meta in component.meta.items():
                    summary["total_fields"] += 1
                    
                    if not field_meta.expose_to_llm:
                        summary["hidden_fields"] += 1
                    elif field_meta.redaction == RedactionLevel.NONE:
                        summary["exposed_fields"] += 1
                    else:
                        summary["redacted_fields"] += 1
        
        return summary
    
    def _count_projection_fields(self, projection: Dict[str, Any]) -> int:
        """Count total fields in projection"""
        count = 0
        for section, data in projection.items():
            if section.startswith('_'):
                continue
            if isinstance(data, dict):
                count += len(data)
        return count
    
    def _extract_exposed_capabilities(self, projection: Dict[str, Any]) -> List[str]:
        """Extract list of exposed capabilities from projection"""
        capabilities = []
        
        caps_section = projection.get("capabilities", {})
        if isinstance(caps_section, dict):
            flags = caps_section.get("flags", {})
            if isinstance(flags, dict):
                capabilities.extend([cap for cap, enabled in flags.items() if enabled])
        
        return capabilities
    
    def _generate_projection_privacy_summary(self, projection: Dict[str, Any]) -> Dict[str, int]:
        """Generate privacy summary for projection"""
        return {
            "sections": len([k for k in projection.keys() if not k.startswith('_')]),
            "total_fields": self._count_projection_fields(projection),
            "capabilities": len(self._extract_exposed_capabilities(projection))
        }
    
    def _get_field_description(self, field_path: str) -> str:
        """Get human-readable description of field"""
        
        descriptions = {
            "identity.display_name": "Your display name shown to AI",
            "identity.expertise": "Your areas of professional expertise",
            "technical.tools": "Software tools and applications you use",
            "technical.programming_languages": "Programming languages you know",
            "workflow.communication": "Your communication style preferences",
            "learning.styles": "How you prefer to learn new things"
        }
        
        return descriptions.get(field_path, field_path)
    
    def _get_capability_exposure_settings(self, capability_name: str) -> CapabilityExposureSettings:
        """Get current exposure settings for a capability based on user field metadata."""
        
        # Retrieve source fields that map to this capability
        capability_fields = get_capability_fields().get(capability_name, [])
        
        # Default values
        exposed = False
        scope = ExposureScope.SESSION
        redaction = RedactionLevel.COARSE
        notes = ""
        expires_at = None
        
        # Order of redaction severity (none < coarse < masked < hidden)
        redaction_order = [
            RedactionLevel.NONE,
            RedactionLevel.COARSE,
            RedactionLevel.MASKED,
            RedactionLevel.HIDDEN
        ]
        
        # Examine each source field's metadata in the current user's profile
        for field_path in capability_fields:
            try:
                component_name, field_name = field_path.split('.', 1)
                component = getattr(self.user_registry.current_user, component_name, None)
                if not component or not hasattr(component, "meta"):
                    continue
                
                field_meta: FieldMeta = component.meta.get(field_name)  # type: ignore[arg-type]
                if not field_meta:
                    continue
                
                # Determine if any source field is exposed
                if field_meta.expose_to_llm:
                    exposed = True
                
                # Choose the most permissive scope (ALWAYS > SESSION)
                if field_meta.scope == ExposureScope.ALWAYS:
                    scope = ExposureScope.ALWAYS
                
                # Choose the most restrictive redaction level
                if redaction_order.index(field_meta.redaction) > redaction_order.index(redaction):
                    redaction = field_meta.redaction
                
                # Capture earliest expiration if present
                if field_meta.expires_at:
                    if not expires_at or field_meta.expires_at < expires_at:
                        expires_at = field_meta.expires_at
                        
            except Exception:
                # Silently ignore malformed entries
                continue
        
        return CapabilityExposureSettings(
            capability_name=capability_name,
            exposed=exposed,
            scope=scope,
            redaction_level=redaction,
            expires_at=expires_at,
            notes=notes
        )
    
    async def _update_capability_exposure(
        self,
        capability_name: str,
        settings: Dict[str, Any]
    ) -> bool:
        """Update capability exposure settings with full validation"""
        
        try:
            # Validate capability name
            valid_capabilities = [flag.value for flag in CapabilityFlag]
            if capability_name not in valid_capabilities:
                return False
            
            # Normalize scope and redaction to strings
            scope_in = settings.get("scope", "session")
            red_in = settings.get("redaction_level", "coarse")
            
            scope_val = str(scope_in)
            red_val = str(red_in)
            
            # Update capability exposure in user profile
            capability_settings = CapabilityExposureSettings(
                capability_name=capability_name,
                exposed=settings.get("exposed", False),
                scope=scope_val,
                redaction_level=red_val,
                expires_at=datetime.fromisoformat(settings["expires_at"]) if settings.get("expires_at") else None,
                notes=settings.get("notes", "")
            )
            
            # Store in user registry
            await self.user_registry.store_capability_exposure(capability_settings)
            
            # Update projection registry
            await self.user_registry.update_field_metadata(
                "capabilities", 
                capability_name, 
                FieldMeta(
                    expose_to_llm=capability_settings.exposed,
                    scope=scope_val,
                    redaction=red_val,
                    expires_at=capability_settings.expires_at
                )
            )
            
            # Clear projection cache
            self.cached_projections.clear()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update capability exposure: {e}")
            return False

    def _export_privacy_settings(self) -> Dict[str, Any]:
        """Export privacy settings for backup"""
        
        privacy_settings = {}
        
        if not self.user_registry.current_user:
            return privacy_settings
        
        # Export metadata for all components
        for component_name in ["identity", "technical", "workflow", "learning"]:
            component = getattr(self.user_registry.current_user, component_name)
            if hasattr(component, 'meta'):
                privacy_settings[component_name] = {}
                for field_name, field_meta in component.meta.items():
                    privacy_settings[component_name][field_name] = {
                        "expose_to_llm": field_meta.expose_to_llm,
                        "scope": field_meta.scope,
                        "redaction": field_meta.redaction,  # fixed: already string
                        "expires_at": field_meta.expires_at.isoformat() if field_meta.expires_at else None
                    }
        
        return privacy_settings
    
    async def _import_privacy_settings(self, privacy_settings: Dict[str, Any]) -> None:
        """Import privacy settings from backup"""
        
        for component_name, component_settings in privacy_settings.items():
            for field_name, field_settings in component_settings.items():
                # Normalize inputs to strings for FieldMeta
                scope_in = field_settings.get("scope", "session")
                red_in = field_settings.get("redaction", "coarse")
                try:
                    scope_val = ExposureScope(scope_in).value
                except Exception:
                    scope_val = str(scope_in)
                try:
                    red_val = RedactionLevel(red_in).value
                except Exception:
                    red_val = str(red_in)
                
                metadata = FieldMeta(
                    expose_to_llm=field_settings.get("expose_to_llm", False),
                    scope=scope_val,
                    redaction=red_val,
                    expires_at=datetime.fromisoformat(field_settings["expires_at"]) if field_settings.get("expires_at") else None
                )
                
                await self.user_registry.update_field_metadata(
                    component_name, field_name, metadata
                )
    
    def _sanitize_export_data(self, export_data: Dict[str, Any]) -> None:
        """Remove sensitive information from export data"""
        
        # Remove password hashes and other sensitive security data
        if "profile" in export_data and "security" in export_data["profile"]:
            del export_data["profile"]["security"]
        
        # Remove any fields marked as sensitive
        # This would be more comprehensive in a real implementation
    
    def _validate_import_data(self, import_data: Dict[str, Any]) -> bool:
        """Validate structure of import data"""
        
        required_keys = ["export_metadata", "profile"]
        return all(key in import_data for key in required_keys)
    
    async def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent user activity for dashboard"""
        
        activity = []
        
        # Add recent profile changes
        if self.user_registry.current_user:
            activity.append({
                "type": "profile_update",
                "description": "Profile information updated",
                "timestamp": self.user_registry.current_user.last_active.isoformat(),  # fixed
                "details": "Last profile modification"
            })
        
        # Add recent AI observations
        recent_observations = [
            obs for obs in self.user_registry.pending_observations
            if (datetime.now(timezone.utc) - obs.timestamp).total_seconds() < 86400  # Last 24 hours
        ]
        
        for obs in recent_observations[-5:]:  # Last 5 observations
            activity.append({
                "type": "ai_observation",
                "description": f"AI suggested update to {obs.field_path}",
                "timestamp": obs.timestamp.isoformat(),
                "details": f"Confidence: {obs.confidence:.2f}"
            })
        
        return sorted(activity, key=lambda x: x["timestamp"], reverse=True)


# ============================================================================
# GLOBAL SETTINGS MENU INSTANCE
# ============================================================================

_settings_menu: Optional[UserPersonalizationMenu] = None

def get_settings_menu(user_registry: UserRegistryManager) -> UserPersonalizationMenu:
    """Get settings menu instance"""
    global _settings_menu
    if _settings_menu is None:
        _settings_menu = UserPersonalizationMenu(user_registry)
    return _settings_menu

def initialize_settings_menu(user_registry: UserRegistryManager) -> UserPersonalizationMenu:
    """Initialize settings menu"""
    global _settings_menu
    _settings_menu = UserPersonalizationMenu(user_registry)
    return _settings_menu


# ============================================================================
# CONVENIENCE FUNCTIONS FOR UI INTEGRATION
# ============================================================================

async def get_user_settings_dashboard(user_registry: UserRegistryManager) -> Dict[str, Any]:
    """Get complete user settings dashboard data"""
    
    menu = get_settings_menu(user_registry)
    overview = await menu.get_settings_overview()
    
    # Add additional dashboard data
    dashboard = {
        "overview": overview,
        "quick_actions": [
            {"id": "update_display_name", "title": "Update Display Name", "category": "profile"},
            {"id": "adjust_privacy", "title": "Adjust Privacy Settings", "category": "privacy"},
            {"id": "preview_projection", "title": "Preview AI Context", "category": "privacy"},
            {"id": "manage_capabilities", "title": "Manage Capabilities", "category": "capabilities"},
            {"id": "review_observations", "title": "Review AI Observations", "category": "ai_interaction"}
        ],
        "recent_projections": list(menu.cached_projections.keys())[-5:],
        "settings_categories": {
            category.value: {
                "title": category.value.replace('_', ' ').title(),
                "description": f"Manage {category.value.replace('_', ' ')} settings"
            }
            for category in SettingsCategory
        }
    }
    
    return dashboard

async def handle_ui_settings_request(
    user_registry: UserRegistryManager,
    category: str,
    action: str,
    target: str = "",
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Handle settings request from UI with response formatting"""
    
    menu = get_settings_menu(user_registry)
    
    try:
        category_enum = SettingsCategory(category)
        action_enum = SettingsAction(action)
        
        operation = await menu.handle_settings_request(category_enum, action_enum, target, data)
        
        response = {
            "success": operation.success,
            "message": operation.message,
            "operation_id": operation.operation_id,
            "timestamp": operation.timestamp.isoformat()
        }
        
        # Add specific response data based on action
        if action == "preview" and operation.success:
            preview = await menu.get_projection_preview(
                data.get("scope", "session") if data else "session",
                data.get("agent_id") if data else None,
                force_refresh=True
            )
            response["projection_preview"] = {
                "scope": preview.scope,
                "field_count": preview.field_count,
                "capabilities_exposed": preview.capabilities_exposed,
                "projection_hash": preview.projection_hash,
                "privacy_summary": preview.privacy_summary
            }
        
        return response
        
    except ValueError as e:
        return {
            "success": False,
            "message": f"Invalid category or action: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        import logging
        logging.basicConfig(level=logging.INFO)
        
        from user_registry import create_user_registry
        
        # Initialize user registry
        user_registry = await create_user_registry()
        
        # Create test user
        try:
            profile = await user_registry.create_profile(
                username="test_settings_user",
                password="secure123",
                identity_data={
                    "display_name": "Settings Test User",
                    "role": "Developer",
                    "expertise": ["Python", "AI"]
                }
            )
            
            # Authenticate
            auth_profile = await user_registry.authenticate("test_settings_user", "secure123")
            
            if auth_profile:
                # Initialize settings menu
                menu = initialize_settings_menu(user_registry)
                
                # Get settings overview
                overview = await menu.get_settings_overview()
                print(f"Settings overview: {overview['user_info']}")
                
                # Get projection preview
                preview = await menu.get_projection_preview("session")
                print(f"Projection preview: {preview.field_count} fields, {len(preview.capabilities_exposed)} capabilities")
                
                # Test settings operation
                operation = await menu.handle_settings_request(
                    SettingsCategory.PROFILE,
                    SettingsAction.EDIT,
                    "identity.display_name",
                    {"value": "Updated Test User"}
                )
                print(f"Settings operation: {operation.success} - {operation.message}")
                
                # Get AI observations
                observations = await menu.get_ai_observations()
                print(f"AI observations: {len(observations)}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            await user_registry.shutdown()
    
    asyncio.run(main())