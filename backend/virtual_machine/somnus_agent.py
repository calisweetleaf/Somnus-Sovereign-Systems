# somnus_agent.py

import logging
import os
import signal
import json
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, Optional

# Core dependencies to be pre-installed in the base VM image.
import psutil
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Agent Configuration ---
AGENT_CONFIG_PATH = Path("/etc/somnus/agent_config.json")
AGENT_PORT = 9901

# --- Global Variables for Caching and Model ---
ST_MODEL: Optional[SentenceTransformer] = None
ERROR_EMBEDDINGS: Dict[str, np.ndarray] = {}

# Known error categories moved to global for reuse and fallback when model is disabled/unavailable
ERROR_CATEGORIES = {
    "CUDA_ERROR": "CUDA out of memory, GPU error, nvidia-smi failed",
    "NETWORK_FAILURE": "Connection timed out, failed to resolve host, network is unreachable",
    "FILE_NOT_FOUND": "No such file or directory, cannot find path specified",
    "PERMISSION_DENIED": "Permission denied, access is denied, operation not permitted",
    "AUTHENTICATION_ERROR": "Authentication failed, invalid credentials, API key error",
    "DEPENDENCY_MISSING": "ModuleNotFoundError, ImportError, package not found",
    "RUNTIME_ERROR": "TypeError, ValueError, IndexError, NoneType object has no attribute"
}

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight cosine similarity without sklearn."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

# lightweight auth guard (set 'agent_token' in /etc/somnus/agent_config.json)
@app.before_request
def _auth_guard():
    cfg = load_agent_config()
    token = cfg.get("agent_token")
    if token and request.headers.get("X-Agent-Token") != token:
        return jsonify({"error": "unauthorized"}), 403

# --- Flask App Initialization ---
app = Flask(__name__)

# Configure logging for the agent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SomnusAgent] - %(levelname)s - %(message)s'
)

# --- Core Helper Functions ---

@lru_cache(maxsize=1)
def load_agent_config() -> Dict:
    """
    Loads and caches the agent's configuration from the JSON file.
    This configuration dictates which processes and logs to monitor.
    """
    if not AGENT_CONFIG_PATH.exists():
        logging.error(f"Agent config file not found at {AGENT_CONFIG_PATH}")
        return {"monitored_processes": [], "log_files": {}}
    try:
        with open(AGENT_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load or parse agent config: {e}")
        return {"monitored_processes": [], "log_files": {}}

def initialize_embedding_model():
    """
    Loads the sentence-transformer model and pre-computes embeddings for known error categories.
    Honors 'embedding_enabled' and 'embedding_model_name' in config.
    """
    global ST_MODEL, ERROR_EMBEDDINGS
    try:
        config = load_agent_config()
        if not config.get("embedding_enabled", True):
            logging.info("Embedding model disabled via config.")
            ST_MODEL = None
            ERROR_EMBEDDINGS.clear()
            return

        model_name = config.get("embedding_model_name", "all-MiniLM-L6-v2")
        logging.info(f"Loading sentence-transformer model: {model_name}...")
        ST_MODEL = SentenceTransformer(model_name)
        logging.info("Embedding model loaded successfully.")

        logging.info("Pre-computing embeddings for error categories...")
        ERROR_EMBEDDINGS.clear()
        if ST_MODEL:
            for category, keywords in ERROR_CATEGORIES.items():
                ERROR_EMBEDDINGS[category] = ST_MODEL.encode(keywords)
        logging.info("Error embeddings are ready.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to initialize the embedding model: {e}")
        ST_MODEL = None
        ERROR_EMBEDDINGS.clear()

def find_monitored_processes() -> list[psutil.Process]:
    """
    Finds running instances of the core AI processes based on the config file.
    """
    config = load_agent_config()
    process_names = config.get("monitored_processes", [])
    if not process_names:
        return []

    found_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if any(p_name in cmdline for p_name in process_names):
                found_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return found_processes

# --- API Endpoints ---

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check the agent's health and the status of monitored processes."""
    monitored_procs = find_monitored_processes()
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitored_processes_running": len(monitored_procs) > 0,
        "pids": [p.pid for p in monitored_procs],
        "embedding_model_loaded": ST_MODEL is not None
    })

@app.route('/health/live', methods=['GET'])
def health_live():
    """Liveness probe - process is up."""
    return jsonify({"status": "live", "timestamp": datetime.now(timezone.utc).isoformat()})

@app.route('/health/ready', methods=['GET'])
def health_ready():
    """Readiness probe - config is loaded and (optionally) model is ready."""
    cfg_loaded = bool(load_agent_config())
    model_required = load_agent_config().get("embedding_enabled", True)
    ready = cfg_loaded and (ST_MODEL is not None or not model_required)
    return jsonify({
        "status": "ready" if ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_loaded": cfg_loaded,
        "embedding_required": model_required,
        "embedding_model_loaded": ST_MODEL is not None
    }), (200 if ready else 503)

@app.route('/config/reload', methods=['POST'])
def reload_config():
    """Reload agent config and reinitialize embeddings."""
    try:
        load_agent_config.cache_clear()
        initialize_embedding_model()
        return jsonify({"status": "reloaded"}), 200
    except Exception as e:
        logging.error(f"Config reload failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/soft-reboot', methods=['POST'])
def soft_reboot():
    """
    Endpoint to trigger a soft reboot. It gracefully terminates monitored
    processes, relying on a system-level supervisor (like systemd) to
    restart them.
    """
    logging.info("Soft reboot command received.")
    procs_to_restart = find_monitored_processes()
    terminated_pids = []

    for proc in procs_to_restart:
        try:
            proc.terminate()
            terminated_pids.append(proc.pid)
            logging.info(f"Sent SIGTERM to process {proc.pid} ({proc.name()}).")
        except psutil.NoSuchProcess:
            logging.warning(f"Process {proc.pid} already terminated.")
        except Exception as e:
            logging.error(f"Could not terminate process {proc.pid}: {e}")

    logging.info("Core AI processes terminated. A supervisor (e.g., systemd) should restart them.")
    return jsonify({"status": "reboot_signal_sent", "terminated_pids": terminated_pids})

@app.route('/stats/basic', methods=['GET'])
def get_basic_stats():
    """Endpoint to provide raw, overall system resource usage inside the VM."""
    return jsonify({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_cpu_percent": psutil.cpu_percent(),
        "overall_memory_percent": psutil.virtual_memory().percent
    })

@app.route('/stats/intelligent', methods=['GET'])
def get_intelligent_stats():
    """
    Endpoint to provide a qualitative analysis of the AI's recent activity
    by analyzing log files for errors and faults.
    Falls back to keyword matching when embeddings are unavailable.
    """
    config = load_agent_config()
    log_files = config.get("log_files", {})
    lines_to_read = config.get("log_lines_to_analyze", 200)

    plugin_faults = 0
    error_frequency = {category: 0 for category in ERROR_CATEGORIES.keys()}
    recent_errors = []

    for process_name, log_path_str in log_files.items():
        log_path = Path(log_path_str)
        if not log_path.exists():
            continue

        try:
            lines = []
            with open(log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                buffer = bytearray()
                while len(lines) < lines_to_read and f.tell() > 0:
                    f.seek(-1, os.SEEK_CUR)
                    byte = f.read(1)
                    if byte == b'\n':
                        lines.append(buffer.decode('utf-8', errors='ignore')[::-1])
                        buffer.clear()
                    else:
                        buffer.extend(byte)
                    f.seek(-1, os.SEEK_CUR)
                if buffer:
                    lines.append(buffer.decode('utf-8', errors='ignore')[::-1])

            for line in reversed(lines):
                if "[PluginFault]" in line:
                    plugin_faults += 1
                    continue

                if "ERROR" in line or "Traceback" in line or "Exception" in line:
                    if ST_MODEL and ERROR_EMBEDDINGS:
                        error_embedding = ST_MODEL.encode(line)
                        similarities = {
                            cat: _cosine_sim(error_embedding, emb)
                            for cat, emb in ERROR_EMBEDDINGS.items()
                        }
                        best_match = max(similarities, key=similarities.get)
                        score = similarities[best_match]
                        threshold = float(config.get("embedding_match_threshold", 0.4))
                        if score > threshold:
                            error_frequency[best_match] += 1
                            if len(recent_errors) < 10:
                                recent_errors.append({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "type": best_match,
                                    "message": line.strip(),
                                    "score": round(score, 3)
                                })
                    else:
                        # Keyword-based fallback (no embedding model)
                        line_l = line.lower()
                        best_match = None
                        for cat, phrases in ERROR_CATEGORIES.items():
                            for phrase in [p.strip().lower() for p in phrases.split(',')]:
                                if phrase and phrase in line_l:
                                    best_match = cat
                                    break
                            if best_match:
                                break
                        if best_match:
                            error_frequency[best_match] += 1
                            if len(recent_errors) < 10:
                                recent_errors.append({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "type": best_match,
                                    "message": line.strip()
                                })
        except Exception as e:
            logging.warning(f"Could not process log file {log_path}: {e}")

    return jsonify({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plugin_faults_total": plugin_faults,
        "error_frequency": error_frequency,
        "recent_errors": recent_errors,
        "embedding_used": ST_MODEL is not None
    })

# --- Flask App Initialization Hook ---
with app.app_context():
    initialize_embedding_model()

# Simple dev runner (use gunicorn/uwsgi in production)
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=AGENT_PORT)

