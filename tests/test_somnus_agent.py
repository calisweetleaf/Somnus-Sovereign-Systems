"""
Test suite for the Somnus Agent component of the Somnus Sovereign Systems.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import numpy as np
from flask import Flask
from flask.testing import FlaskClient

from backend.virtual_machine.somnus_agent import (
    app, load_agent_config, initialize_embedding_model, 
    find_monitored_processes, _cosine_sim
)


class TestSomnusAgent:
    """Test suite for the Somnus Agent Flask application."""

    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    @pytest.fixture
    def temp_config_path(self):
        """Create a temporary agent config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agent_config.json"
            config_data = {
                "monitored_processes": ["python", "node"],
                "log_files": {
                    "test_process": "/tmp/test.log"
                },
                "embedding_enabled": True,
                "embedding_model_name": "all-MiniLM-L6-v2"
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)
            
            with patch("backend.virtual_machine.somnus_agent.AGENT_CONFIG_PATH", config_path):
                yield config_path

    def test_health_live_endpoint(self, client):
        """Test the /health/live endpoint."""
        response = client.get('/health/live')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "live"
        assert "timestamp" in data

    def test_health_ready_endpoint(self, client):
        """Test the /health/ready endpoint."""
        response = client.get('/health/ready')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ready"
        assert "timestamp" in data
        assert "config_loaded" in data
        assert "embedding_model_loaded" in data

    def test_status_endpoint(self, client):
        """Test the /status endpoint."""
        response = client.get('/status')
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "monitored_processes_running" in data
        assert "pids" in data
        assert "embedding_model_loaded" in data

    def test_get_basic_stats_endpoint(self, client):
        """Test the /stats/basic endpoint."""
        response = client.get('/stats/basic')
        assert response.status_code == 200
        data = response.get_json()
        assert "timestamp" in data
        assert "overall_cpu_percent" in data
        assert "overall_memory_percent" in data

    def test_cosine_similarity(self):
        """Test the cosine similarity function."""
        # Test with identical vectors
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        similarity = _cosine_sim(a, b)
        assert abs(similarity - 1.0) < 1e-6

        # Test with orthogonal vectors
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        similarity = _cosine_sim(a, b)
        assert abs(similarity - 0.0) < 1e-6

        # Test with opposite vectors
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        similarity = _cosine_sim(a, b)
        assert abs(similarity - (-1.0)) < 1e-6

    @patch("backend.virtual_machine.somnus_agent.psutil.process_iter")
    def test_find_monitored_processes(self, mock_process_iter, temp_config_path):
        """Test finding monitored processes."""
        # Mock process iterator
        mock_process1 = Mock()
        mock_process1.info = {
            'pid': 1234,
            'name': 'python',
            'cmdline': ['python', 'test_script.py']
        }
        
        mock_process2 = Mock()
        mock_process2.info = {
            'pid': 5678,
            'name': 'node',
            'cmdline': ['node', 'server.js']
        }
        
        mock_process3 = Mock()
        mock_process3.info = {
            'pid': 9999,
            'name': 'bash',
            'cmdline': ['bash']
        }
        
        mock_process_iter.return_value = [mock_process1, mock_process2, mock_process3]
        
        # Load config and find processes
        load_agent_config.cache_clear()
        processes = find_monitored_processes()
        
        # Verify the correct processes were found
        assert len(processes) == 2
        assert processes[0].info['pid'] == 1234
        assert processes[1].info['pid'] == 5678

    @patch("backend.virtual_machine.somnus_agent.SentenceTransformer")
    def test_initialize_embedding_model(self, mock_sentence_transformer, temp_config_path):
        """Test initializing the embedding model."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_sentence_transformer.return_value = mock_model
        
        # Clear cache and initialize model
        load_agent_config.cache_clear()
        initialize_embedding_model()
        
        # Verify the model was loaded
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        
        # Verify embeddings were computed
        from backend.virtual_machine.somnus_agent import ERROR_EMBEDDINGS
        assert len(ERROR_EMBEDDINGS) > 0
        for category, embedding in ERROR_EMBEDDINGS.items():
            assert isinstance(embedding, np.ndarray)


class TestAgentConfig:
    """Test suite for agent configuration functions."""

    def test_load_agent_config_nonexistent(self):
        """Test loading config when file doesn't exist."""
        with patch("backend.virtual_machine.somnus_agent.AGENT_CONFIG_PATH", Path("/nonexistent/config.json")):
            load_agent_config.cache_clear()
            config = load_agent_config()
            assert config == {"monitored_processes": [], "log_files": {}}

    def test_load_agent_config_invalid_json(self, temp_config_path):
        """Test loading config with invalid JSON."""
        # Create invalid JSON file
        with open(temp_config_path, "w") as f:
            f.write("invalid json content")
        
        load_agent_config.cache_clear()
        config = load_agent_config()
        assert config == {"monitored_processes": [], "log_files": {}}


if __name__ == "__main__":
    pytest.main([__file__])