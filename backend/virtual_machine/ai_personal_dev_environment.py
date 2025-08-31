"""
AI Personal Development Environment
Production-ready framework for AI to build persistent development capabilities
Integrated with custom VM supervisor for full environment control
"""

import asyncio
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from uuid import UUID

# Import our enhanced orchestrator
from .enhanced_ai_action_orchestrator import (
    AIActionOrchestrator,
    ArtifactExecutionResult,
    VMActionResult
)

# Import advanced AI shell for proper command execution
from .advanced_ai_shell import (
    AdvancedAIShell,
    ExecutionContext,
    ContainerSpec,
    create_ai_shell
)

@dataclass
class AIPreferences:
    preferred_ides: List[str]
    coding_styles: Dict[str, Any]
    favorite_tools: List[str]
    automation_preferences: Dict[str, Any]
    learning_history: List[Dict[str, Any]]
    efficiency_metrics: Dict[str, float]
    project_templates: Dict[str, List[str]]
    vm_profiles: Dict[str, Dict[str, Any]]

@dataclass
class ProjectEnvironment:
    name: str
    required_tools: List[str]
    virtual_env: Optional[str]
    ide_config: Dict[str, Any]
    startup_commands: List[str]
    directory_structure: List[str]
    recommended_vm_profile: str = "idle"

@dataclass
class AIDevelopmentSession:
    session_id: str
    project_type: str
    project_path: str
    workspace_ready: bool
    ide_session: Dict[str, Any]
    virtual_environment: Dict[str, Any]
    available_templates: List[str]
    installed_tools: List[str]
    startup_results: List[Dict[str, Any]]
    efficiency_rating: float
    setup_time_seconds: float
    associated_vm_id: Optional[str] = None

class AIDevelopmentEnvironment:
    """AI's personal development setup that grows and evolves over time"""

    def __init__(self, orchestrator: AIActionOrchestrator, base_path: str = "/home/ai"):
        self.orchestrator = orchestrator
        self.base_path = Path(base_path)
        self.workspace_path = self.base_path / "workspace"
        self.tools_path = self.base_path / "tools"
        self.projects_path = self.base_path / "projects"
        self.config_path = self.base_path / "config"
        self.cache_path = self.base_path / "cache"
        self.logs_path = self.base_path / "logs"
        self.preferences = self._load_ai_preferences()
        self.installed_tools: List[str] = []
        self.custom_scripts: List[str] = []
        self.personal_libraries: List[str] = []
        self.project_environments: Dict[str, ProjectEnvironment] = {}
        self.active_sessions: Dict[str, AIDevelopmentSession] = {}
        # Initialize AI shell for proper command execution
        self.ai_shell: Optional[AdvancedAIShell] = None
        asyncio.create_task(self._ensure_directory_structure())
        asyncio.create_task(self._initialize_ai_shell())

    async def _initialize_ai_shell(self):
        """Initialize the AI shell for command execution"""
        try:
            self.ai_shell = await create_ai_shell()
        except Exception as e:
            # Fallback to None if shell initialization fails
            # We'll handle this gracefully in _execute_command
            self.ai_shell = None

    async def _ensure_directory_structure(self):
        directories = [
            self.workspace_path,
            self.tools_path,
            self.projects_path,
            self.config_path,
            self.cache_path,
            self.logs_path,
            self.tools_path / "libraries",
            self.tools_path / "automation",
            self.tools_path / "templates",
            self.cache_path / "downloads",
            self.cache_path / "web_requests",
            self.base_path / "venvs",
        ]
        # Create directories using the AI shell's VM execution capability
        for directory in directories:
            await self._execute_command(f"mkdir -p {directory}")

    def _load_ai_preferences(self) -> AIPreferences:
        preferences_file = self.config_path / "ai_preferences.json"
        default_preferences = AIPreferences(
            preferred_ides=["code", "vim", "jupyter"],
            coding_styles={
                "python": {"line_length": 88, "formatter": "black"},
                "javascript": {"formatter": "prettier", "semi": True},
                "general": {"indentation": "spaces", "tab_size": 4},
            },
            favorite_tools=[],
            automation_preferences={"auto_backup": True, "auto_format": True, "auto_test": False},
            learning_history=[],
            efficiency_metrics={},
            project_templates={},
            vm_profiles={
                "idle": {"vcpus": 1, "memory_gb": 4, "description": "Low power state"},
                "coding": {"vcpus": 4, "memory_gb": 8, "description": "Optimized for compilation"},
                "research": {"vcpus": 2, "memory_gb": 6, "description": "Balanced for browsing"},
                "media_creation": {"vcpus": 6, "memory_gb": 16, "gpu_enabled": True, "description": "High-power for generation"}
            }
        )
        try:
            if preferences_file.exists():
                with open(preferences_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge with defaults to ensure all fields are present
                    for field in default_preferences.__dataclass_fields__:
                        if field not in data:
                            data[field] = getattr(default_preferences, field)
                    return AIPreferences(**data)
        except Exception as e:
            self._log(f"Failed to load preferences: {e}", level="ERROR")
        return default_preferences

    async def _save_ai_preferences(self):
        preferences_file = self.config_path / "ai_preferences.json"
        try:
            with open(preferences_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.preferences), f, indent=2, default=str)
        except Exception as e:
            self._log(f"Failed to save preferences: {e}", level="ERROR")

    def _sanitize_command(self, cmd: str) -> str:
        allowed = re.compile(r"[^\\w\\s\\-_=:/.,@+%()\\[\\]{}']+")
        cmd = allowed.sub(" ", cmd)
        return re.sub(r"\\s+", " ", cmd).strip()

    def _sanitize_path(self, p: Path) -> str:
        return re.sub(r"[^\\w\\-_/\\.:", "_", str(p))

    async def setup_development_session(
        self, 
        project_type: str, 
        project_name: Optional[str] = None, 
        create_new: bool = False,
        create_dedicated_vm: bool = False
    ) -> AIDevelopmentSession:
        session_start = datetime.now()
        session_id = hashlib.md5(f"{project_type}_{datetime.now()}".encode()).hexdigest()[:8]
        
        # Get or create project environment
        env_config = await self._get_or_create_project_environment(project_type)
        
        # Create dedicated VM if requested
        vm_id = None
        if create_dedicated_vm:
            vm_result = await self._create_dedicated_vm_for_project(env_config, project_type)
            if vm_result.success and vm_result.data:
                vm_id = vm_result.data.get("vm_id")
        
        # Check and install required tools
        missing_tools = await self._check_required_tools(env_config.required_tools)
        if missing_tools:
            await self._install_tools(missing_tools, project_type)
            
        # Create project workspace
        if project_name and create_new:
            project_path = await self._create_project_workspace(project_name, env_config)
        else:
            project_path = self.workspace_path / project_type
            await self._execute_command(f"mkdir -p {project_path}")
            
        # Setup IDE session
        ide_session = await self._setup_ide_session(env_config, project_path)
        
        # Setup virtual environment
        venv_info = await self._setup_virtual_environment(env_config, project_path)
        
        # Execute startup commands
        startup_results = await self._execute_startup_commands(env_config.startup_commands, project_path)
        
        # Calculate metrics
        setup_time = (datetime.now() - session_start).total_seconds()
        efficiency_rating = self._calculate_efficiency_rating(project_type, setup_time)
        await self._update_learning_history(project_type, setup_time, efficiency_rating)
        templates = self.preferences.project_templates.get(project_type, [])
        
        # Create session object
        session = AIDevelopmentSession(
            session_id=session_id,
            project_type=project_type,
            project_path=str(project_path),
            workspace_ready=True,
            ide_session=ide_session,
            virtual_environment=venv_info,
            available_templates=templates,
            installed_tools=env_config.required_tools,
            startup_results=startup_results,
            efficiency_rating=efficiency_rating,
            setup_time_seconds=setup_time,
            associated_vm_id=vm_id
        )
        
        # Store active session
        self.active_sessions[session_id] = session
        
        return session

    async def _create_dedicated_vm_for_project(self, env_config: ProjectEnvironment, project_type: str) -> VMActionResult:
        """Create a dedicated VM for a project with appropriate resources"""
        vm_name = f"{project_type}_dev_environment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        personality_config = {
            "purpose": f"Development environment for {project_type} projects",
            "preferred_tools": env_config.required_tools
        }
        
        # Create the VM
        result = self.orchestrator.create_sovereign_ai_computer(vm_name, personality_config)
        
        if result.success and result.data:
            vm_id = result.data.get("vm_id")
            if vm_id:
                # Scale to appropriate profile
                self.orchestrator.scale_ai_computer_resources(vm_id, env_config.recommended_vm_profile)
                
        return result

    def _calculate_efficiency_rating(self, project_type: str, setup_time: float) -> float:
        historical_times = [entry["setup_time"] for entry in self.preferences.learning_history if entry["project_type"] == project_type]
        if not historical_times:
            return 1.0
        average_time = sum(historical_times) / len(historical_times)
        if average_time == 0:
            return 1.0
        return max(0.0, min(2.0, (average_time / setup_time)))

    async def _update_learning_history(self, project_type: str, setup_time: float, efficiency_rating: float, errors: Optional[List[str]] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "project_type": project_type,
            "setup_time": setup_time,
            "efficiency_rating": efficiency_rating,
            "errors": errors or [],
        }
        self.preferences.learning_history.append(entry)
        await self._save_ai_preferences()

    def _log(self, message: str, level: str = "INFO", context: Dict = None):
        log_file = self.logs_path / f"ai_dev_env_{datetime.now().strftime('%Y%m%d')}.log"
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()} [{level}] {message}\\n")
            # Log to host system as well
            if hasattr(self.orchestrator.client, "log_event_to_session"):
                self.orchestrator.client.log_event_to_session(
                    event_type=f"ai_dev_env_log_{level.lower()}",
                    content=message,
                    metadata={"level": level, "context": context or {}, "timestamp": datetime.now().isoformat()},
                )
        except Exception:
            pass

    async def _get_or_create_project_environment(self, project_type: str) -> ProjectEnvironment:
        if project_type in self.project_environments:
            return self.project_environments[project_type]
            
        default_environments: Dict[str, ProjectEnvironment] = {
            "web_frontend": ProjectEnvironment(
                name="web_frontend",
                required_tools=["node", "npm", "git", "code"],
                virtual_env=None,
                ide_config={"extensions": ["ms-vscode.vscode-typescript-next", "esbenp.prettier-vscode"]},
                startup_commands=["npm install", "npm audit fix"],
                directory_structure=["src/", "public/", "tests/", "docs/", "package.json", "README.md"],
                recommended_vm_profile="coding"
            ),
            "web_backend": ProjectEnvironment(
                name="web_backend",
                required_tools=["python3", "pip", "git", "code"],
                virtual_env="backend_env",
                ide_config={"extensions": ["ms-python.python", "ms-python.flake8"]},
                startup_commands=["pip install -r requirements.txt"],
                directory_structure=["src/", "tests/", "docs/", "requirements.txt", "README.md", ".env.example"],
                recommended_vm_profile="coding"
            ),
            "ai_research": ProjectEnvironment(
                name="ai_research",
                required_tools=["python3", "pip", "jupyter", "git", "code"],
                virtual_env="ai_research_env",
                ide_config={"extensions": ["ms-python.python", "ms-toolsai.jupyter"]},
                startup_commands=[
                    "pip install jupyter pandas numpy matplotlib seaborn",
                    "pip install torch transformers datasets",
                    "pip install scikit-learn plotly",
                ],
                directory_structure=["notebooks/", "data/", "models/", "results/", "papers/", "README.md"],
                recommended_vm_profile="research"
            ),
            "data_science": ProjectEnvironment(
                name="data_science",
                required_tools=["python3", "pip", "jupyter", "git", "code"],
                virtual_env="data_science_env",
                ide_config={"extensions": ["ms-python.python", "ms-toolsai.jupyter"]},
                startup_commands=[
                    "pip install pandas numpy matplotlib seaborn plotly",
                    "pip install scikit-learn scipy statsmodels",
                    "pip install dash streamlit",
                ],
                directory_structure=["data/raw/", "data/processed/", "notebooks/", "src/", "reports/", "README.md"],
                recommended_vm_profile="research"
            ),
            "mobile_app": ProjectEnvironment(
                name="mobile_app",
                required_tools=["node", "npm", "git", "code", "android-studio"],
                virtual_env=None,
                ide_config={"extensions": ["ms-vscode.vscode-react-native"]},
                startup_commands=["npm install -g react-native-cli", "npm install"],
                directory_structure=["src/", "assets/", "tests/", "android/", "ios/", "package.json"],
                recommended_vm_profile="coding"
            ),
            "devops": ProjectEnvironment(
                name="devops",
                required_tools=["docker", "kubectl", "terraform", "git", "code"],
                virtual_env=None,
                ide_config={"extensions": ["ms-vscode.docker", "hashicorp.terraform"]},
                startup_commands=["docker --version", "kubectl version --client"],
                directory_structure=["terraform/", "k8s/", "docker/", "scripts/", "docs/", "README.md"],
                recommended_vm_profile="coding"
            ),
            "general": ProjectEnvironment(
                name="general",
                required_tools=["git", "code"],
                virtual_env=None,
                ide_config={},
                startup_commands=[],
                directory_structure=["src/", "docs/", "tests/", "README.md"],
                recommended_vm_profile="idle"
            ),
        }
        
        if project_type in default_environments:
            env = default_environments[project_type]
            self.project_environments[project_type] = env
            return env
            
        # Create custom environment based on existing files
        custom_env = ProjectEnvironment(
            name=project_type,
            required_tools=["git", "code"],
            virtual_env=f"{project_type}_env",
            ide_config={},
            startup_commands=[],
            directory_structure=["src/", "docs/", "README.md"],
            recommended_vm_profile="idle"
        )
        
        ls_res = await self._execute_command(f"ls -1 {self.workspace_path}")
        if ls_res.success and ls_res.exit_code == 0:
            file_list = [line.strip() for line in ls_res.output.splitlines()]
            if any(f.endswith(('.js', '.jsx', '.ts', '.tsx')) for f in file_list):
                if "node" not in custom_env.required_tools:
                    custom_env.required_tools.extend(["node", "npm"])
                custom_env.ide_config["extensions"] = ["ms-vscode.vscode-typescript-next", "esbenp.prettier-vscode"]
                custom_env.startup_commands.append("npm install")
                custom_env.recommended_vm_profile = "coding"
            if any(f.endswith('.py') for f in file_list):
                for t in ("python3", "pip"):
                    if t not in custom_env.required_tools:
                        custom_env.required_tools.append(t)
                custom_env.ide_config["extensions"] = ["ms-python.python", "ms-python.flake8"]
                custom_env.startup_commands.append("pip install -r requirements.txt")
                custom_env.recommended_vm_profile = "coding"
            if "Dockerfile" in file_list or "docker-compose.yml" in file_list:
                if "docker" not in custom_env.required_tools:
                    custom_env.required_tools.append("docker")
                custom_env.recommended_vm_profile = "coding"
                    
        self.project_environments[project_type] = custom_env
        return custom_env

    async def _check_required_tools(self, required_tools: List[str]) -> List[str]:
        missing: List[str] = []
        for tool in required_tools:
            res = await self._execute_command(f"command -v {tool} || which {tool}")
            if not res.success or res.exit_code != 0:
                missing.append(tool)
        return missing

    async def _check_path_exists(self, path: Path) -> bool:
        res = await self._execute_command(f"test -e {self._sanitize_path(path)}")
        return res.success and res.exit_code == 0

    async def _install_tools(self, tools: List[str], project_type: str):
        # Detect operating system - Windows 11 native app runs in WSL2 context
        os_type_res = await self._execute_command("uname -s")
        os_type = (os_type_res.output.strip() if os_type_res.success and os_type_res.exit_code == 0 else "Linux")
        
        # Check for Windows environment
        if "Microsoft" in os_type or "WSL" in os_type:
            os_type = "Windows"
        
        installation_commands: Dict[str, Dict[str, List[str]]] = {
            "Linux": {
                "node": [
                    "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -",
                    "sudo apt-get install -y nodejs",
                ],
                "npm": ["sudo apt-get install -y npm"],
                "python3": ["sudo apt-get install -y python3 python3-pip"],
                "pip": ["sudo apt-get install -y python3-pip"],
                "git": ["sudo apt-get install -y git"],
                "code": [
                    "wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg",
                    "sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/",
                    "sudo sh -c 'echo \"deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main\" > /etc/apt/sources.list.d/vscode.list'",
                    "sudo apt update",
                    "sudo apt install -y code",
                ],
                "jupyter": ["pip3 install jupyter jupyterlab"],
                "docker": [
                    "sudo apt-get update",
                    "sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release",
                    "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg",
                    "echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable\" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null",
                    "sudo apt-get update",
                    "sudo apt-get install -y docker-ce docker-ce-cli containerd.io",
                ],
                "kubectl": [
                    "curl -LO \"https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl\"",
                    "sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl",
                ],
                "terraform": [
                    "wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg",
                    "echo \"deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/hashicorp.list",
                    "sudo apt update && sudo apt install -y terraform",
                ],
            },
            "Windows": {
                # Windows 11 with WSL2/Docker Desktop support
                "node": [
                    "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -",
                    "sudo apt-get install -y nodejs",
                ],
                "npm": ["sudo apt-get install -y npm"],
                "python3": ["sudo apt-get install -y python3 python3-pip"],
                "pip": ["sudo apt-get install -y python3-pip"],
                "git": ["sudo apt-get install -y git"],
                "code": [
                    # For WSL2, install code-server or use Windows VS Code via PATH
                    "wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg",
                    "sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/",
                    "sudo sh -c 'echo \"deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main\" > /etc/apt/sources.list.d/vscode.list'",
                    "sudo apt update",
                    "sudo apt install -y code",
                ],
                "jupyter": ["pip3 install jupyter jupyterlab"],
                "docker": [
                    # On Windows 11, Docker Desktop handles Docker - just ensure Docker CLI is available
                    "echo 'Docker Desktop for Windows should be installed. Checking Docker CLI availability...'",
                    "which docker || echo 'Please ensure Docker Desktop is running and WSL2 integration is enabled'",
                ],
                "kubectl": [
                    "curl -LO \"https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl\"",
                    "sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl",
                ],
                "terraform": [
                    "wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg",
                    "echo \"deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main\" | sudo tee /etc/apt/sources.list.d/hashicorp.list",
                    "sudo apt update && sudo apt install -y terraform",
                ],
            },
            "Darwin": {
                # macOS support
                "node": ["brew install node"],
                "npm": ["brew install npm"],
                "python3": ["brew install python@3.11"],
                "pip": ["python3 -m ensurepip --upgrade"],
                "git": ["brew install git"],
                "code": ["brew install --cask visual-studio-code"],
                "jupyter": ["pip3 install jupyter jupyterlab"],
                "docker": ["brew install --cask docker"],
                "kubectl": ["brew install kubectl"],
                "terraform": ["brew tap hashicorp/tap && brew install hashicorp/tap/terraform"],
            },
        }
        
        commands_for_os = installation_commands.get(os_type, installation_commands["Linux"])
        for tool in tools:
            cmds = commands_for_os.get(tool)
            if not cmds:
                self._log(f"No installer configured for {tool} on {os_type}", level="WARNING")
                continue
            self._log(f"Installing {tool} for {project_type} project on {os_type}")
            ok = True
            for cmd in cmds:
                res = await self._execute_command(cmd)
                if not res.success or res.exit_code != 0:
                    ok = False
                    error_msg = res.error if res.error else (res.output if not res.success else "Unknown error")
                    self._log(f"Failed to install {tool}: {error_msg}", level="ERROR")
                    break
            if ok:
                self.installed_tools.append(tool)
                self._log(f"Successfully installed {tool}")
                await self._create_tool_shortcuts(tool)

    async def _create_tool_shortcuts(self, tool: str):
        shortcuts = {
            "git": [
                "alias gs='git status'",
                "alias ga='git add .'",
                "alias gc='git commit -m'",
                "alias gp='git push'",
                "alias gl='git log --oneline -10'",
            ],
            "code": ["alias c='code .'", "alias cn='code -n'", "alias cr='code -r'"],
            "docker": [
                "alias d='docker'",
                "alias dc='docker-compose'",
                "alias dps='docker ps'",
                "alias di='docker images'",
            ],
            "kubectl": [
                "alias k='kubectl'",
                "alias kgp='kubectl get pods'",
                "alias kgs='kubectl get services'",
                "alias kgd='kubectl get deployments'",
            ],
        }
        if tool in shortcuts:
            bashrc_path = f"{self.base_path}/.bashrc"
            for alias in shortcuts[tool]:
                await self._execute_command(f'echo "{alias}" >> {bashrc_path}')

    async def _create_project_workspace(self, project_name: str, env_config: ProjectEnvironment) -> Path:
        project_path = self.projects_path / project_name
        await self._execute_command(f"mkdir -p {project_path}")
        for item in env_config.directory_structure:
            item_path = project_path / item
            if item.endswith('/'):
                await self._execute_command(f"mkdir -p {item_path}")
            else:
                await self._execute_command(f"mkdir -p {item_path.parent} && touch {item_path}")
        await self._execute_command(f"cd {project_path} && git init")
        gitignore_content = self._generate_gitignore(env_config.name)
        await self._write_file(project_path / ".gitignore", gitignore_content)
        return project_path

    def _generate_gitignore(self, project_type: str) -> str:
        base_gitignore = """
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage
coverage/
*.lcov
.nyc_output

# node
node_modules/
.npm
.eslintcache
*.tgz
.yarn-integrity

# env
.env
.env.*

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
""".strip()
        project_specific = {
            "ai_research": """
__pycache__/
*.py[cod]
*.so
build/
dist/
.eggs/
*.egg-info/
.ipynb_checkpoints
.python-version
models/
checkpoints/
            """,
            "web_frontend": """
build/
dist/
node_modules/
.env*
            """,
            "devops": """
*.tfstate
*.tfstate.*
.terraform/
.terraform.lock.hcl
kubeconfig
Dockerfile.local
            """,
        }
        if project_type in project_specific:
            return base_gitignore + "\\n" + project_specific[project_type]
        return base_gitignore

    async def _setup_ide_session(self, env_config: ProjectEnvironment, project_path: Path) -> Dict[str, Any]:
        ide_info: Dict[str, Any] = {
            "primary_ide": self.preferences.preferred_ides[0] if self.preferences.preferred_ides else "code",
            "extensions_installed": [],
            "workspace_config": {},
            "success": False,
        }
        primary_ide = ide_info["primary_ide"]
        if primary_ide == "code" and env_config.ide_config.get("extensions"):
            for extension in env_config.ide_config["extensions"]:
                res = await self._execute_command(f"code --install-extension {extension}")
                if res.success and res.exit_code == 0:
                    ide_info["extensions_installed"].append(extension)
        if primary_ide == "code":
            workspace_config = {
                "folders": [{"path": str(project_path)}],
                "settings": {**self.preferences.coding_styles.get("general", {}), **env_config.ide_config.get("settings", {})},
                "extensions": {"recommendations": env_config.ide_config.get("extensions", [])},
            }
            workspace_file = project_path / f"{project_path.name}.code-workspace"
            await self._write_file(workspace_file, json.dumps(workspace_config, indent=2))
            launch = await self._execute_command(f"code {project_path}")
            ide_info["workspace_config"] = workspace_config
            ide_info["success"] = launch.success and launch.exit_code == 0
        return ide_info

    async def _setup_virtual_environment(self, env_config: ProjectEnvironment, project_path: Path) -> Dict[str, Any]:
        venv_info: Dict[str, Any] = {"virtual_env_name": env_config.virtual_env, "virtual_env_path": None, "active": False, "success": False}
        if not env_config.virtual_env:
            venv_info["success"] = True
            return venv_info
        venv_path = self.base_path / "venvs" / env_config.virtual_env
        venv_info["virtual_env_path"] = str(venv_path)
        if not await self._check_path_exists(venv_path):
            res = await self._execute_command(f"python3 -m venv {venv_path}")
            if not res.success or res.exit_code != 0:
                return venv_info
        activation_script = f"""#!/bin/bash
source {venv_path}/bin/activate
export VIRTUAL_ENV_NAME=\\"{env_config.virtual_env}\\"
echo \\"Activated virtual environment: {env_config.virtual_env}\\"\\n""".strip()
        script_path = project_path / "activate_env.sh"
        await self._write_file(script_path, activation_script)
        await self._execute_command(f"chmod +x {script_path}")
        venv_info["active"] = True
        venv_info["success"] = True
        return venv_info

    async def _execute_startup_commands(self, commands: List[str], project_path: Path) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for cmd in commands:
            sanitized = self._sanitize_command(cmd)
            res = await self._execute_command(f"cd {project_path} && {sanitized}")
            results.append({
                "command": cmd,
                "success": res.success,
                "output": res.output,
                "error": res.error,
                "exit_code": res.exit_code
            })
            if not res.success or res.exit_code != 0:
                self._log(f"Startup command failed: {cmd}", level="WARNING")
        return results

    async def create_personal_library(self, library_name: str, description: str, library_type: str = "python") -> str:
        library_path = self.tools_path / "libraries" / library_name
        await self._execute_command(f"mkdir -p {library_path}")
        if library_type == "python":
            library_structure = {
                "__init__.py": f'"""AI Personal Library: {description}"""',
                "core.py": await self._generate_core_utilities(),
                "automation.py": await self._generate_automation_utilities(),
                "research.py": await self._generate_research_utilities(),
                "development.py": await self._generate_development_utilities(),
                "README.md": f"# {library_name}\\n\\n{description}\\n\\nPersonal AI library for enhanced development workflow.",
                "setup.py": (
                    f"from setuptools import setup, find_packages\n\n"
                    f"setup(name='{library_name}', version='1.0.0', description='{description}', packages=find_packages(), install_requires=[], python_requires='>=3.8')\n"
                ),
            }
        else:
            library_structure = {"README.md": f"# {library_name}\\n\\n{description}\\n"}
        for filename, content in library_structure.items():
            await self._write_file(library_path / filename, content)
        if library_type == "python":
            await self._execute_command(f"cd {library_path} && pip install -e .")
        self.personal_libraries.append(library_name)
        await self._save_ai_preferences()
        
        # Create artifact for the library
        if hasattr(self.orchestrator, "create_and_setup_artifact"):
            self.orchestrator.create_and_setup_artifact(
                title=f"AI Personal Library: {library_name}",
                content=f"AI-generated personal library: {library_name}",
                artifact_type="text/python"
            )
        return str(library_path)

    async def _generate_core_utilities(self) -> str:
        content = '''
"""Core AI Utilities - Essential functions for development"""

from __future__ import annotations
import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

class FileUtils:
    @staticmethod
    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

class JSONUtils:
    @staticmethod
    def safe_loads(s: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(s)
        except Exception:
            return None

    @staticmethod
    def safe_dumps(obj: Any, indent: int = 2) -> str:
        try:
            return json.dumps(obj, indent=indent, default=str)
        except Exception:
            return json.dumps({"error": "serialization_failed"})

class Timing:
    @staticmethod
    def timeit(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
        t0 = time.time()
        result = fn(*args, **kwargs)
        t1 = time.time()
        return {"result": result, "elapsed": t1 - t0}
'''
        return content

    async def _generate_automation_utilities(self) -> str:
        content = '''
"""AI Automation Utilities"""

from __future__ import annotations
import shutil
import os
import time
from pathlib import Path
from typing import List

class AutomationHelper:
    @staticmethod
    def backup_directory(source_dir: Path, destination_dir: Path) -> str:
        source_dir = Path(source_dir)
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.rglob('*'):
            if item.is_file():
                rel = item.relative_to(source_dir)
                target = destination_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        return f"Backup complete: {source_dir} -> {destination_dir}"

    @staticmethod
    def clean_directory(target_dir: Path, older_than_days: int) -> int:
        cutoff = time.time() - (older_than_days * 86400)
        removed = 0
        for item in Path(target_dir).rglob('*'):
            try:
                if item.is_file() and item.stat().st_mtime < cutoff:
                    item.unlink()
                    removed += 1
            except Exception:
                continue
        return removed
'''
        return content

    async def _generate_research_utilities(self) -> str:
        content = '''
"""AI Research Utilities"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None  # Fallback if not installed

class ResearchHelper:
    @staticmethod
    def parse_pdf(pdf_path: Path) -> str:
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 not available")
        reader = PyPDF2.PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        citations = [l for l in lines if any(tag in l.lower() for tag in ["doi:", "arxiv:", "http", "www."])]
        return citations
'''
        return content

    async def _generate_development_utilities(self) -> str:
        content = '''
"""AI Development Utilities"""

from __future__ import annotations
from pathlib import Path

class DevelopmentHelper:
    @staticmethod
    def create_flask_app(app_name: str, path: Path) -> str:
        app_path = Path(path) / app_name
        app_path.mkdir(parents=True, exist_ok=True)
        (app_path / "app.py").write_text(
            """from flask import Flask\\napp = Flask(__name__)\\n\\n@app.route('/')\\ndef hello_world():\\n    return 'Hello, World!'\\n\\nif __name__ == '__main__':\\n    app.run(host='0.0.0.0', port=8000)\\n"""
        )
        (app_path / "requirements.txt").write_text("Flask\\n")
        (app_path / "README.md").write_text(f"# {app_name}\\n\\nBasic Flask application.\\n")
        return f"Created Flask app at {app_path}"
'''
        return content

    # --- Enhanced Execution Methods ---
    
    async def _execute_command(self, command: str, context: ExecutionContext = ExecutionContext.VM_NATIVE) -> ArtifactExecutionResult:
        """
        Execute a command using the advanced AI shell system.
        This routes commands through the proper execution context (VM native, container, etc.)
        """
        try:
            sanitized_cmd = self._sanitize_command(command)
            
            # Use AI shell if available, fallback to orchestrator if not
            if self.ai_shell is not None:
                # Execute using the advanced AI shell with proper context routing
                result = await self.ai_shell.execute_command(
                    sanitized_cmd,
                    context=context
                )
                
                return ArtifactExecutionResult(
                    success=result.return_code == 0,
                    exit_code=result.return_code,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time=result.execution_time
                )
            else:
                # Fallback: try to use orchestrator's VM execution if AI shell not available
                try:
                    vm_result = await self.orchestrator.execute_in_vm(sanitized_cmd)
                    return ArtifactExecutionResult(
                        success=vm_result.success,
                        exit_code=0 if vm_result.success else -1,
                        output=vm_result.output or "",
                        error=vm_result.error or "",
                        execution_time=vm_result.execution_time or 0.0
                    )
                except Exception as orchestrator_error:
                    # Final fallback: limited subprocess execution for critical operations
                    import subprocess
                    import time
                    
                    start_time = time.time()
                    try:
                        result = subprocess.run(
                            sanitized_cmd, 
                            shell=True, 
                            capture_output=True, 
                            text=True, 
                            timeout=30
                        )
                        execution_time = time.time() - start_time
                        return ArtifactExecutionResult(
                            success=result.returncode == 0,
                            exit_code=result.returncode,
                            output=result.stdout,
                            error=result.stderr,
                            execution_time=execution_time
                        )
                    except subprocess.TimeoutExpired:
                        execution_time = time.time() - start_time
                        return ArtifactExecutionResult(
                            success=False,
                            exit_code=124,  # Standard timeout exit code
                            output="",
                            error="Command timed out",
                            execution_time=execution_time
                        )
                        
        except Exception as e:
            return ArtifactExecutionResult(
                success=False,
                exit_code=-1,
                output="",
                error=f"Command execution failed: {str(e)}",
                execution_time=0.0
            )

    async def _write_file(self, path: Path, content: str) -> bool:
        """
        Write content to a file in the current environment.
        """
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            self._log(f"Failed to write file {path}: {e}", level="ERROR")
            return False


class AILogger:
    def __init__(self, orchestrator: AIActionOrchestrator, log_dir: str = "/home/ai/logs"):
        self.orchestrator = orchestrator
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]

    def log(self, message: str, level: str = "INFO", context: Dict = None):
        timestamp = datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, "session_id": self.session_id, "level": level, "message": message, "context": context or {}}
        log_file = self.log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\\n")
            # Log to host system as well
            if hasattr(self.orchestrator.client, "log_event_to_session"):
                self.orchestrator.client.log_event_to_session(
                    event_type=f"ai_logger_{level.lower()}",
                    content=message,
                    metadata={"context": context or {}, "session_id": self.session_id}
                )
        except Exception:
            pass


class SmartCache:
    def __init__(self, cache_dir: str = "/home/ai/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get(self, key: str, ttl_hours: int = 24) -> Optional[Any]:
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            cached_time = datetime.fromisoformat(cache_data["timestamp"])
            if (datetime.now() - cached_time).total_seconds() > ttl_hours * 3600:
                cache_file.unlink(missing_ok=True)
                return None
            return cache_data["value"]
        except Exception:
            return None

    def set(self, key: str, value: Any):
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
        cache_data = {"timestamp": datetime.now().isoformat(), "key": key, "value": value}
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, default=str, indent=2)
        except Exception:
            pass


class ProjectHelper:
    @staticmethod
    def create_directory_structure(base_path: Path, structure: List[str]):
        for item in structure:
            item_path = base_path / item
            if item.endswith('/'):
                item_path.mkdir(parents=True, exist_ok=True)
            else:
                item_path.parent.mkdir(parents=True, exist_ok=True)
                item_path.touch(exist_ok=True)

    @staticmethod
    def extract_code_blocks(text: str) -> List[Dict[str, str]]:
        import re
        pattern = re.compile(r"```([\\w+-]*)\\n([\\s\\S]*?)```", re.MULTILINE)
        blocks: List[Dict[str, str]] = []
        for match in pattern.finditer(text):
            lang = match.group(1) or ""
            code = match.group(2)
            blocks.append({"language": lang, "code": code})
        return blocks


# Example usage function
async def create_ai_developer_environment(orchestrator: AIActionOrchestrator) -> AIDevelopmentEnvironment:
    """
    Factory function to create a fully initialized AI development environment.
    """
    # Create the development environment
    dev_env = AIDevelopmentEnvironment(orchestrator)
    
    # Return the initialized environment
    return dev_env