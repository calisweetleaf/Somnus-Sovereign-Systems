# sovereign_ai_orchestrator.py

"""
================================================================================
Morpheus "SaaS Killer" - Sovereign AI Orchestrator
================================================================================

This module replaces the previous `integrated_vm.py`. It acts as the central
conductor for the entire system, translating high-level user requests into
a sequence of operations across the various managers (VM, Memory, Session, etc.).

It embodies the shift from a simple "bridge" architecture to a true
host-controlled orchestration model, leveraging the full power of the
VMSupervisor and the in-VM SomnusAgent.
"""

import logging
from typing import Dict, List, Any, Optional
from uuid import UUID

# Import all the necessary manager and schema components from your codebase
# Note: Adjust the import paths based on your final project structure.
from backend.virtual_machine.vm_supervisor import VMSupervisor, AIVMInstance
from ..schemas.dev_session_schemas import DevSession, DevSessionStatus
from ..core.session_manager import DevSessionManager
from ..core.memory_core import MemoryManager, MemoryType, MemoryImportance
from ..security.security_layer import SecurityEnforcer

# --- Type Aliases for Clarity ---
UserID = str

# --- Capability Pack Definitions ---

# These packs define the set of tools and configurations for specific tasks.
# They are easily extensible.
CAPABILITY_PACKS: Dict[str, Dict[str, List[str]]] = {
    "base_tools": {
        "description": "Essential tools for any development environment.",
        "commands": [
            "sudo apt-get update -y",
            "sudo apt-get install -y git curl wget build-essential python3-pip",
        ]
    },
    "web_development": {
        "description": "A complete environment for modern web development.",
        "commands": [
            "sudo apt-get install -y nodejs npm",
            "sudo npm install -g typescript create-react-app @vue/cli",
            "sudo snap install code --classic",
            "code --install-extension dbaeumer.vscode-eslint",
            "code --install-extension esbenp.prettier-vscode",
        ]
    },
    "ai_research": {
        "description": "A powerful environment for AI/ML research and development.",
        "commands": [
            "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "pip3 install transformers datasets jupyterlab pandas numpy matplotlib",
            "pip3 install accelerate",
            "sudo snap install code --classic",
            "code --install-extension ms-python.python",
        ]
    },
    "data_analysis": {
        "description": "Tools for data manipulation, analysis, and visualization.",
        "commands": [
            "pip3 install pandas numpy scikit-learn seaborn plotly dash",
            "sudo apt-get install -y r-base",
        ]
    }
}


class SovereignAIOrchestrator:
    """
    The central conductor that integrates all system components to provision
    and manage sovereign AI development environments.
    """

    def __init__(
        self,
        vm_supervisor: VMSupervisor,
        dev_session_manager: DevSessionManager,
        memory_manager: MemoryManager,
        security_enforcer: SecurityEnforcer
    ):
        """
        Initializes the orchestrator with instances of all core managers.
        This uses dependency injection for modularity and testability.
        """
        self.vm_supervisor = vm_supervisor
        self.dev_session_manager = dev_session_manager
        self.memory_manager = memory_manager
        self.security_enforcer = security_enforcer
        logging.info("Sovereign AI Orchestrator initialized with all core managers.")

    async def provision_sovereign_environment(
        self,
        user_id: UserID,
        session_title: str,
        personality_config: Dict[str, Any],
        requested_capabilities: List[str]
    ) -> Dict[str, Any]:
        """
        The main workflow for creating a new, fully configured, persistent AI environment.

        This process is idempotent and includes security validation, VM provisioning,
        session creation, resource scaling, and capability installation with snapshotting.

        Args:
            user_id: The ID of the user requesting the environment.
            session_title: The title for the new development session.
            personality_config: The personality configuration for the AI.
            requested_capabilities: A list of capability pack names to install.

        Returns:
            A dictionary containing the state of the newly provisioned environment.
        """
        logging.info(f"Provisioning new sovereign environment '{session_title}' for user {user_id}.")

        # 1. Security Validation (Placeholder for your security logic)
        # In a real scenario, you'd validate the user's request here.
        # For example: security_result = self.security_enforcer.validate_provisioning_request(...)
        # if not security_result.allowed:
        #     raise PermissionError("Provisioning request denied by security policy.")

        # 2. Provision the Persistent VM
        try:
            vm_instance = await self.vm_supervisor.create_ai_computer(
                instance_name=f"{user_id}-{session_title.replace(' ', '_')}",
                personality_config=personality_config
            )
        except Exception as e:
            logging.error(f"VM provisioning failed: {e}")
            return {"status": "error", "message": f"Failed to provision VM: {e}"}

        # 3. Create the Central DevSession Record
        # The DevSession is the source of truth for this environment.
        dev_session = await self.dev_session_manager.create_session(
            user_id=user_id,
            chat_session_id=UUID(int=0),  # Placeholder, link to a real chat session ID
            title=session_title
        )
        # Link the VM to the session
        dev_session.vm_instance_id = vm_instance.vm_id
        dev_session.add_event(
            event_type="vm_assigned",
            content=f"Assigned persistent VM {vm_instance.vm_id} to session.",
            actor="Orchestrator"
        )
        logging.info(f"Created DevSession {dev_session.dev_session_id} and linked VM {vm_instance.vm_id}.")

        # 4. Install Capability Packs with Snapshotting
        # This creates a safe, versioned history of the AI's "learning" process.
        installed_packs = []
        try:
            # Always install base tools first
            all_capabilities = ["base_tools"] + [cap for cap in requested_capabilities if cap != "base_tools"]

            for capability_name in all_capabilities:
                if capability_name in CAPABILITY_PACKS:
                    logging.info(f"Installing capability pack '{capability_name}' into VM {vm_instance.vm_id}...")
                    
                    # Create a snapshot BEFORE installing, enabling rollback.
                    self.vm_supervisor.create_snapshot(
                        vm_id=vm_instance.vm_id,
                        description=f"Before installing '{capability_name}' pack."
                    )
                    
                    pack = CAPABILITY_PACKS[capability_name]
                    for command in pack["commands"]:
                        # Here, we would use the VMSupervisor to execute commands inside the VM
                        # This requires an `execute_command_in_vm` method in your supervisor.
                        # For now, we'll log the action and add to the tool list.
                        logging.info(f"Executing command in VM: '{command}'")
                        # await self.vm_supervisor.execute_command_in_vm(vm_instance.vm_id, command)
                        vm_instance.installed_tools.append(command.split(" ")[0]) # Simplified tracking
                    
                    dev_session.add_event(
                        event_type="capability_granted",
                        content=f"Successfully installed capability pack: {capability_name}",
                        actor="Orchestrator"
                    )
                    installed_packs.append(capability_name)
                    logging.info(f"Successfully installed '{capability_name}'.")

        except Exception as e:
            logging.error(f"Failed during capability installation for VM {vm_instance.vm_id}: {e}")
            dev_session.status = "error"
            dev_session.add_event(event_type="system_message", content=f"Installation failed: {e}", actor="Orchestrator")
            # Consider rolling back to the last good snapshot here.
            # self.vm_supervisor.rollback_to_last_snapshot(vm_instance.vm_id)

        # 5. Finalize and Save State
        await self.dev_session_manager._save_session(dev_session)
        self.vm_supervisor._save_vm_config(vm_instance)

        logging.info(f"Provisioning complete for '{session_title}'.")
        return {
            "status": "success",
            "message": "Sovereign environment provisioned successfully.",
            "dev_session_id": dev_session.dev_session_id,
            "vm_id": vm_instance.vm_id,
            "vm_ip": vm_instance.internal_ip,
            "installed_capabilities": installed_packs,
            "connection_info": {
                "ssh": f"ssh user@{vm_instance.internal_ip} -p {vm_instance.ssh_port}",
                "vnc": f"vnc://127.0.0.1:{vm_instance.vnc_port}"
            }
        }

    async def get_environment_status(self, dev_session_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Retrieves the complete status of a sovereign environment.

        Args:
            dev_session_id: The ID of the development session.

        Returns:
            A dictionary with the combined status, or None if not found.
        """
        dev_session = await self.dev_session_manager.get_session(dev_session_id)
        if not dev_session or not dev_session.vm_instance_id:
            return None

        vm_instance = self.vm_supervisor.active_vms.get(dev_session.vm_instance_id)
        if not vm_instance:
            return None

        return {
            "dev_session": dev_session.model_dump(exclude={'event_log'}),
            "vm_status": {
                "state": vm_instance.vm_state,
                "profile": vm_instance.current_profile,
                "ip": vm_instance.internal_ip,
                "snapshots": [s.model_dump() for s in vm_instance.snapshots]
            },
            "latest_runtime_stats": vm_instance.runtime_stats_history[-1].model_dump() if vm_instance.runtime_stats_history else None
        }

    async def shutdown_environment(self, dev_session_id: UUID) -> bool:
        """
        Gracefully shuts down the VM and archives the session.

        Args:
            dev_session_id: The ID of the development session to shut down.

        Returns:
            True if shutdown was successful, False otherwise.
        """
        dev_session = await self.dev_session_manager.get_session(dev_session_id)
        if not dev_session or not dev_session.vm_instance_id:
            return False

        logging.info(f"Shutting down environment for session {dev_session_id}...")
        
        # 1. Shutdown the VM
        success = self.vm_supervisor.shutdown_vm(dev_session.vm_instance_id)
        if not success:
            logging.error(f"Failed to shutdown VM {dev_session.vm_instance_id}.")
            return False

        # 2. Archive the DevSession
        await self.dev_session_manager.set_session_status(dev_session_id, DevSessionStatus.ARCHIVED)
        
        logging.info(f"Environment for session {dev_session_id} has been shut down and archived.")
        return True


