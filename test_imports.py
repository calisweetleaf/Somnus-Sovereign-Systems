import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    # Test importing from the backend.virtual_machine package
    from backend.virtual_machine.ai_orchestrator import SovereignAIOrchestrator
    print("Successfully imported SovereignAIOrchestrator")
except ImportError as e:
    print(f"Failed to import SovereignAIOrchestrator: {e}")

try:
    # Test importing from the backend.virtual_machine package
    from backend.virtual_machine.vm_supervisor import VMSupervisor
    print("Successfully imported VMSupervisor")
except ImportError as e:
    print(f"Failed to import VMSupervisor: {e}")
    
try:
    # Test importing from the backend.virtual_machine package
    from backend.virtual_machine.somnus_agent import AGENT_PORT
    print("Successfully imported somnus_agent")
except ImportError as e:
    print(f"Failed to import somnus_agent: {e}")