import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure the backend modules can be imported
backend_path = project_root / "backend"
if backend_path.exists():
    sys.path.insert(0, str(backend_path))
