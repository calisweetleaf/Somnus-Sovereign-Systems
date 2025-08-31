# somnus_agent.py (v3 - Digital Twin Edition)

"""
================================================================================
Somnus Digital Twin Agent (v3 - Production Ready)
================================================================================

This script is the in-guest counterpart to the VMSupervisor. It has been
redesigned from a simple agent to a stateful "Digital Twin."

Key Architectural Changes:
- **Stateful, Proactive Monitoring:** A background thread continuously monitors
  system resources, processes, and logs, maintaining a live, in-memory model
  of the VM's runtime state.
- **High-Performance Endpoints:** API endpoints now read from this pre-computed
  in-memory state, making them extremely fast and responsive.
- **Lightweight & Focused:** The sentence-transformer model has been removed.
  The agent's sole responsibility is to gather and provide raw, high-quality
  data to the host-side supervisor, which now handles all "intelligent"
  analysis.
- **Digital Twin Concept:** The agent acts as a persistent, real-time "clone"
  of the VM's operational status, remembering recent events and providing a
  coherent picture of its health over time.
"""

import logging
import os
import json
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Any, List, Optional
from collections import deque

import psutil
from flask import Flask, jsonify, request

# --- Agent Configuration ---
# For cross-platform compatibility, we'll check for a config in the current
# directory first, then fall back to a standard Linux path.
CONFIG_CANDIDATES = [
    Path("./agent_config.json"),
    Path("/etc/somnus/agent_config.json")
]
AGENT_PORT = 9901 # Port must match SomnusVMAgentClient in supervisor

# --- The Digital Twin's State ---

class DigitalTwinState:
    """
    A thread-safe class that holds the live operational state of the VM.
    This object is continuously updated by a background monitoring thread.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc)
        self.last_updated_at: Optional[datetime] = None

        # Overall system stats
        self.overall_cpu_percent: float = 0.0
        self.overall_memory_percent: float = 0.0

        # Per-process detailed stats
        self.process_stats: Dict[str, Dict[str, Any]] = {}

        # Log analysis stats
        self.plugin_faults: int = 0
        self.error_frequency: Dict[str, int] = {}

        # Event history (e.g., process crashes)
        self.event_history = deque(maxlen=50)

    def update_system_stats(self):
        """Updates overall CPU and memory stats."""
        with self._lock:
            self.overall_cpu_percent = psutil.cpu_percent()
            self.overall_memory_percent = psutil.virtual_memory().percent
            self.last_updated_at = datetime.now(timezone.utc)

    def update_process_stat(self, process_name: str, data: Dict[str, Any]):
        """Updates the state for a single monitored process."""
        with self._lock:
            if process_name not in self.process_stats:
                self.process_stats[process_name] = {}
            self.process_stats[process_name].update(data)

    def record_event(self, event_type: str, message: str, metadata: Dict = None):
        """Records a significant event in the agent's history."""
        with self._lock:
            self.event_history.appendleft({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "message": message,
                "metadata": metadata or {}
            })

    def increment_fault_counter(self, fault_type: str, category: Optional[str] = None):
        """Increments counters for plugin faults or categorized errors."""
        with self._lock:
            if fault_type == "plugin":
                self.plugin_faults += 1
            elif fault_type == "error" and category:
                self.error_frequency[category] = self.error_frequency.get(category, 0) + 1

    def get_full_stats(self) -> Dict[str, Any]:
        """
        Returns a snapshot of the current state, formatted to match the
        VMRuntimeStats schema expected by the supervisor.
        """
        with self._lock:
            return {
                "timestamp": self.last_updated_at.isoformat() if self.last_updated_at else datetime.now(timezone.utc).isoformat(),
                "overall_cpu_percent": self.overall_cpu_percent,
                "overall_memory_percent": self.overall_memory_percent,
                "process_stats": self.process_stats.copy(),
                "plugin_faults": self.plugin_faults,
                "error_frequency": self.error_frequency.copy()
            }

# --- Global State and App Initialization ---

twin_state = DigitalTwinState()
app = Flask(__name__)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SomnusTwinAgent] - %(levelname)s - %(message)s'
)

# --- Core Helper Functions ---

@lru_cache(maxsize=1)
def load_agent_config() -> Dict:
    """Loads the agent's configuration from a JSON file."""
    config_path = None
    for path in CONFIG_CANDIDATES:
        if path.exists():
            config_path = path
            break

    if not config_path:
        logging.error(f"Agent config file not found in any candidate location.")
        return {}

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load or parse agent config from {config_path}: {e}")
        return {}

# --- Background Monitoring Thread ---

def _monitor_loop(stop_event: threading.Event):
    """
    The main loop for the background thread that continuously updates the
    DigitalTwinState.
    """
    logging.info("Digital Twin background monitor started.")
    log_file_positions = {}
    
    # Known error categories for keyword matching
    error_categories = {
        "CUDA_ERROR": "CUDA out of memory, GPU error, nvidia-smi failed",
        "NETWORK_FAILURE": "Connection timed out, failed to resolve host, network is unreachable",
        "FILE_NOT_FOUND": "No such file or directory, cannot find path specified",
        "PERMISSION_DENIED": "Permission denied, access is denied, operation not permitted",
        "AUTHENTICATION_ERROR": "Authentication failed, invalid credentials, API key error",
        "DEPENDENCY_MISSING": "ModuleNotFoundError, ImportError, package not found",
        "RUNTIME_ERROR": "TypeError, ValueError, IndexError, NoneType object has no attribute"
    }

    while not stop_event.is_set():
        config = load_agent_config()
        monitored_processes = config.get("monitored_processes", [])
        log_files = config.get("log_files", {})
        
        # 1. Update overall system stats
        twin_state.update_system_stats()

        # 2. Update process stats
        running_procs_this_cycle = set()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                for p_name in monitored_processes:
                    if p_name in cmdline:
                        running_procs_this_cycle.add(p_name)
                        twin_state.update_process_stat(p_name, {
                            "pid": proc.info['pid'],
                            "status": "running",
                            "cpu_percent": proc.info['cpu_percent'],
                            "memory_mb": proc.info['memory_info'].rss / (1024 * 1024)
                        })
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Check for processes that were running but are now gone
        previous_running_procs = {p for p, d in twin_state.process_stats.items() if d.get("status") == "running"}
        stopped_procs = previous_running_procs - running_procs_this_cycle
        for p_name in stopped_procs:
            twin_state.update_process_stat(p_name, {"status": "stopped", "pid": None})
            twin_state.record_event("PROCESS_STOPPED", f"Monitored process '{p_name}' is no longer running.")

        # 3. Analyze log files
        for name, path_str in log_files.items():
            try:
                log_path = Path(path_str)
                if not log_path.exists():
                    continue

                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    last_pos = log_file_positions.get(path_str, 0)
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    log_file_positions[path_str] = f.tell()

                    for line in new_lines:
                        if "[PluginFault]" in line:
                            twin_state.increment_fault_counter("plugin")
                        
                        if "ERROR" in line or "Traceback" in line or "Exception" in line:
                            line_l = line.lower()
                            best_match = None
                            for cat, phrases in error_categories.items():
                                for phrase in [p.strip().lower() for p in phrases.split(',')]:
                                    if phrase and phrase in line_l:
                                        best_match = cat
                                        break
                                if best_match:
                                    break
                            if best_match:
                                twin_state.increment_fault_counter("error", best_match)

            except Exception as e:
                logging.warning(f"Could not process log file {path_str}: {e}")

        time.sleep(config.get("monitor_interval_seconds", 5))

# --- API Endpoints ---

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check the agent's health and a summary of the VM's status."""
    config = load_agent_config()
    monitored_procs = config.get("monitored_processes", [])
    
    running_count = 0
    state_snapshot = twin_state.process_stats
    for p_name in monitored_procs:
        if state_snapshot.get(p_name, {}).get("status") == "running":
            running_count += 1
            
    return jsonify({
        "agent_status": "ok",
        "digital_twin_state_last_updated": twin_state.last_updated_at.isoformat() if twin_state.last_updated_at else None,
        "monitored_processes_total": len(monitored_procs),
        "monitored_processes_running": running_count
    })

@app.route('/stats', methods=['GET'])
def get_runtime_stats():
    """
    Returns a detailed snapshot of runtime statistics, conforming to the
    VMRuntimeStats schema expected by the supervisor.
    """
    return jsonify(twin_state.get_full_stats())

@app.route('/soft-reboot', methods=['POST'])
def soft_reboot():
    """
    Terminates all monitored processes, relying on an external supervisor
    (like systemd or a startup script) to restart them.
    """
    logging.info("Soft reboot command received.")
    terminated_pids = []
    
    state_snapshot = twin_state.process_stats
    for process_name, data in state_snapshot.items():
        if data.get("status") == "running" and data.get("pid"):
            pid = data["pid"]
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                terminated_pids.append(pid)
                twin_state.record_event("SOFT_REBOOT_ACTION", f"Sent SIGTERM to process '{process_name}' (PID: {pid}).")
                logging.info(f"Sent SIGTERM to process {pid} ({process_name}).")
            except psutil.NoSuchProcess:
                logging.warning(f"Process {pid} ({process_name}) already terminated.")
            except Exception as e:
                logging.error(f"Could not terminate process {pid}: {e}")

    # The supervisor expects a specific response format
    return jsonify({"status": "rebooting", "terminated_pids": terminated_pids})

@app.route('/health/live', methods=['GET'])
def health_live():
    """Liveness probe - confirms the agent process is running."""
    return jsonify({"status": "live"})

@app.route('/health/ready', methods=['GET'])
def health_ready():
    """Readiness probe - confirms the agent has loaded config and the twin is updating."""
    config_loaded = bool(load_agent_config())
    twin_updated = twin_state.last_updated_at is not None
    ready = config_loaded and twin_updated
    return jsonify({
        "status": "ready" if ready else "not_ready",
        "config_loaded": config_loaded,
        "digital_twin_active": twin_updated
    }), (200 if ready else 503)

@app.route('/execute_command', methods=['POST'])
def execute_command():
    """
    Execute a command inside the VM and return stdout, stderr, and exit code.
    This endpoint is used by the vm_supervisor for capability pack installation
    and other system management tasks.
    """
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'command' in request body"
            }), 400
        
        command = data['command']
        timeout = data.get('timeout', 300)  # Default 5 minutes
        working_dir = data.get('working_dir', None)
        env_vars = data.get('env_vars', {})
        
        logging.info(f"Executing command: {command}")
        twin_state.record_event("COMMAND_EXECUTION", f"Executing: {command}")
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update(env_vars)
        
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
            env=env
        )
        
        # Log execution result
        twin_state.record_event("COMMAND_COMPLETED", 
            f"Command completed with exit code {result.returncode}")
        
        response_data = {
            "success": True,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": command
        }
        
        logging.info(f"Command executed. Exit code: {result.returncode}")
        return jsonify(response_data)
        
    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {timeout} seconds"
        logging.error(error_msg)
        twin_state.record_event("COMMAND_TIMEOUT", error_msg)
        return jsonify({
            "success": False,
            "error": error_msg,
            "exit_code": -1
        }), 408
        
    except Exception as e:
        error_msg = f"Command execution failed: {str(e)}"
        logging.error(error_msg)
        twin_state.record_event("COMMAND_ERROR", error_msg)
        return jsonify({
            "success": False,
            "error": error_msg,
            "exit_code": -1
        }), 500

# --- Production WSGI Configuration ---

def create_app():
    """Factory function for creating the Flask app with monitoring"""
    # Start the background monitoring thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=_monitor_loop, args=(stop_event,), daemon=True)
    monitor_thread.start()
    
    # Store the stop event and thread in app config for cleanup
    app.config['STOP_EVENT'] = stop_event
    app.config['MONITOR_THREAD'] = monitor_thread
    
    logging.info(f"Somnus Digital Twin Agent initialized on port {AGENT_PORT}.")
    logging.info("This agent runs inside the VM and provides data to the host supervisor.")
    
    return app


# --- Main Execution ---

if __name__ == "__main__":
    import sys
    import signal
    
    # Check if we should use production WSGI server
    use_production = os.environ.get('SOMNUS_PRODUCTION', 'false').lower() == 'true'
    
    if use_production:
        try:
            # Try to use Gunicorn for production
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()

                def load_config(self):
                    config = {key: value for key, value in self.options.items()
                              if key in self.cfg.settings and value is not None}
                    for key, value in config.items():
                        self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            # Production Gunicorn configuration
            options = {
                'bind': f'0.0.0.0:{AGENT_PORT}',
                'workers': 2,  # Keep lightweight for VM resources
                'worker_class': 'sync',
                'worker_connections': 100,
                'max_requests': 1000,
                'max_requests_jitter': 100,
                'timeout': 60,
                'keepalive': 2,
                'preload_app': True,
                'capture_output': True,
                'enable_stdio_inheritance': True,
                'access_log': '-',  # Log to stdout
                'error_log': '-',   # Log to stderr
                'log_level': 'info'
            }
            
            # Create app and run with Gunicorn
            application = create_app()
            
            # Setup signal handlers for graceful shutdown
            def signal_handler(sig, frame):
                logging.info("Received shutdown signal, cleaning up...")
                stop_event = application.config.get('STOP_EVENT')
                monitor_thread = application.config.get('MONITOR_THREAD')
                
                if stop_event:
                    stop_event.set()
                if monitor_thread and monitor_thread.is_alive():
                    monitor_thread.join(timeout=5)
                
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            logging.info("Starting Somnus Agent with Gunicorn production server")
            StandaloneApplication(application, options).run()
            
        except ImportError:
            logging.warning("Gunicorn not available, falling back to Flask development server")
            use_production = False
    
    if not use_production:
        # Development mode or Gunicorn not available
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=_monitor_loop, args=(stop_event,), daemon=True)
        monitor_thread.start()
        
        logging.info(f"Somnus Digital Twin Agent starting on port {AGENT_PORT} (development mode).")
        logging.info("This agent runs inside the VM and provides data to the host supervisor.")
        logging.info("Set SOMNUS_PRODUCTION=true environment variable for production WSGI server")
        
        try:
            # Flask development server
            app.run(host="0.0.0.0", port=AGENT_PORT, debug=False, threaded=True)
        finally:
            # Cleanup on exit
            stop_event.set()
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=5)