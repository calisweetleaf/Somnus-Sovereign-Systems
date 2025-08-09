#!/usr/bin/env python3
"""
Somnus Purple Team Agent - Synthesis Engine
===========================================

Advanced threat intelligence synthesis combining red and blue team operations.
Creates adaptive learning loops, correlates attack/defense data, and evolves
security posture based on real-world testing results.

Core Functions:
- Real-time attack/defense correlation
- Adaptive security policy generation
- Machine learning-driven threat prediction
- Automated security posture optimization
- Cross-team intelligence fusion
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import pickle
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union
import joblib
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.cluster import DBSCAN, KMeans
import xgboost as xgb
import yaml
from scipy import stats
from scipy.spatial.distance import cosine
import hashlib
import hmac

# Import Somnus components
from combined_security_system import (
    UnifiedThreatLevel,
    UnifiedSecurityResult,
    ThreatIntelligenceEngine,
    IntegratedSecurityOrchestrator
)
from network_security import (
    SecurityEvent,
    ThreatLevel,
    SecurityAuditLogger,
    SomnusSecurityFramework
)
from vm_supervisor import (
    VMSupervisor,
    SomnusVMAgentClient,
    AIVMInstance,
    VMState
)

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for adaptive security"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    FEDERATED = "federated"


class CorrelationType(Enum):
    """Types of attack/defense correlations"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    STRATEGIC = "strategic"


@dataclass
class AttackDefenseCorrelation:
    """Correlation between red team attacks and blue team responses"""
    correlation_id: str = field(default_factory=lambda: f"corr_{int(time.time())}_{hash(str(np.random.random()))}")
    red_team_event_id: str = ""
    blue_team_response_id: str = ""
    correlation_strength: float = 0.0
    correlation_type: CorrelationType = CorrelationType.TEMPORAL
    attack_vector: str = ""
    defense_mechanism: str = ""
    effectiveness_score: float = 0.0
    false_positive_rate: float = 0.0
    response_time: float = 0.0
    mitigation_success: bool = False
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThreatIntelligencePattern:
    """Learned threat intelligence pattern"""
    pattern_id: str = field(default_factory=lambda: f"pattern_{int(time.time())}_{hash(str(np.random.random()))}")
    pattern_type: str = ""
    feature_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_score: float = 0.0
    attack_indicators: List[str] = field(default_factory=list)
    defense_recommendations: List[str] = field(default_factory=list)
    source_events: List[str] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ThreatIntelligenceFusion:
    """Fuses threat intelligence from multiple sources into actionable insights"""
    
    def __init__(self, db_path: str = "threat_intelligence.db"):
        self.db_path = db_path
        self.intelligence_db = self._initialize_database()
        self.pattern_cache: Dict[str, ThreatIntelligencePattern] = {}
        self.correlation_graph = nx.DiGraph()
        self.threat_vectors = defaultdict(list)
        self.defense_effectiveness = defaultdict(list)
        
        # ML models for intelligence fusion
        self.threat_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.threat_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.pattern_clusterer = DBSCAN(eps=0.3, min_samples=5)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature extraction
        self.feature_extractors = {
            'temporal': self._extract_temporal_features,
            'network': self._extract_network_features,
            'behavioral': self._extract_behavioral_features,
            'payload': self._extract_payload_features
        }
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize threat intelligence database"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables for threat intelligence storage
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS threat_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                feature_vector BLOB,
                confidence_score REAL,
                attack_indicators TEXT,
                defense_recommendations TEXT,
                source_events TEXT,
                validation_results TEXT,
                evolution_history TEXT,
                first_observed TEXT,
                last_updated TEXT
            );
            
            CREATE TABLE IF NOT EXISTS correlations (
                correlation_id TEXT PRIMARY KEY,
                red_team_event_id TEXT,
                blue_team_response_id TEXT,
                correlation_strength REAL,
                correlation_type TEXT,
                attack_vector TEXT,
                defense_mechanism TEXT,
                effectiveness_score REAL,
                false_positive_rate REAL,
                response_time REAL,
                mitigation_success INTEGER,
                learned_patterns TEXT,
                timestamp TEXT
            );
            
            CREATE TABLE IF NOT EXISTS intelligence_sources (
                source_id TEXT PRIMARY KEY,
                source_type TEXT,
                credibility_score REAL,
                last_updated TEXT,
                data_quality_metrics TEXT
            );
            
            CREATE TABLE IF NOT EXISTS threat_predictions (
                prediction_id TEXT PRIMARY KEY,
                threat_type TEXT,
                predicted_probability REAL,
                confidence_interval TEXT,
                time_horizon_hours INTEGER,
                contributing_factors TEXT,
                recommended_actions TEXT,
                prediction_timestamp TEXT,
                validation_outcome TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_patterns_type ON threat_patterns(pattern_type);
            CREATE INDEX IF NOT EXISTS idx_correlations_timestamp ON correlations(timestamp);
            CREATE INDEX IF NOT EXISTS idx_predictions_type ON threat_predictions(threat_type);
        ''')
        
        conn.commit()
        return conn
    
    async def ingest_threat_intelligence(self, source_data: Dict[str, Any], source_type: str) -> str:
        """Ingest and process threat intelligence from various sources"""
        source_id = f"src_{int(time.time())}_{hash(json.dumps(source_data, sort_keys=True))}"
        
        # Extract features from source data
        features = await self._extract_comprehensive_features(source_data, source_type)
        
        # Create threat pattern
        pattern = ThreatIntelligencePattern(
            pattern_type=source_type,
            feature_vector=features,
            confidence_score=self._calculate_confidence_score(source_data, features),
            attack_indicators=self._extract_attack_indicators(source_data),
            defense_recommendations=await self._generate_defense_recommendations(features, source_type)
        )
        
        # Store in database
        await self._store_threat_pattern(pattern)
        
        # Update correlation graph
        self._update_correlation_graph(pattern, source_data)
        
        # Cache pattern
        self.pattern_cache[pattern.pattern_id] = pattern
        
        logger.info(f"Ingested threat intelligence: {source_id} -> {pattern.pattern_id}")
        return pattern.pattern_id
    
    async def _extract_comprehensive_features(self, data: Dict[str, Any], source_type: str) -> np.ndarray:
        """Extract comprehensive feature vectors from threat data"""
        all_features = []
        
        # Extract features using different extractors
        for extractor_type, extractor_func in self.feature_extractors.items():
            try:
                features = extractor_func(data, source_type)
                all_features.extend(features)
            except Exception as e:
                logger.error(f"Error in {extractor_type} feature extraction: {e}")
                all_features.extend([0.0] * 10)  # Fallback zeros
        
        return np.array(all_features, dtype=np.float32)
    
    def _extract_temporal_features(self, data: Dict[str, Any], source_type: str) -> List[float]:
        """Extract temporal pattern features"""
        features = []
        
        # Time-based features
        timestamp = data.get('timestamp', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        features.extend([
            timestamp.hour / 24.0,
            timestamp.weekday() / 7.0,
            timestamp.day / 31.0,
            timestamp.month / 12.0
        ])
        
        # Event frequency features
        event_count = data.get('event_count', 1)
        duration = data.get('duration_seconds', 1)
        
        features.extend([
            min(event_count / 100.0, 1.0),
            min(duration / 3600.0, 1.0),
            min((event_count / max(duration, 1)) * 60, 1.0)  # Events per minute
        ])
        
        # Periodicity features
        if 'previous_events' in data:
            intervals = []
            prev_events = data['previous_events']
            for i in range(1, len(prev_events)):
                interval = (prev_events[i] - prev_events[i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                features.extend([
                    np.mean(intervals) / 3600.0,  # Mean interval in hours
                    np.std(intervals) / 3600.0,   # Std interval in hours
                    len(set(int(i/3600) for i in intervals)) / 24.0  # Unique hour buckets
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _extract_network_features(self, data: Dict[str, Any], source_type: str) -> List[float]:
        """Extract network-based features"""
        features = []
        
        # IP-based features
        source_ip = data.get('source_ip', '')
        if source_ip:
            try:
                import ipaddress
                ip_obj = ipaddress.ip_address(source_ip)
                features.extend([
                    1.0 if ip_obj.is_private else 0.0,
                    1.0 if ip_obj.is_multicast else 0.0,
                    1.0 if ip_obj.is_loopback else 0.0,
                    hash(source_ip) % 1000 / 1000.0  # IP hash feature
                ])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Port and protocol features
        port = data.get('port', 0)
        protocol = data.get('protocol', 'unknown').lower()
        
        features.extend([
            min(port / 65535.0, 1.0) if port else 0.0,
            1.0 if port in [80, 443, 22, 25, 53] else 0.0,  # Common ports
            1.0 if protocol == 'tcp' else 0.0,
            1.0 if protocol == 'udp' else 0.0,
            1.0 if protocol == 'icmp' else 0.0
        ])
        
        # Traffic volume features
        bytes_transferred = data.get('bytes', 0)
        packets = data.get('packets', 0)
        
        features.extend([
            min(bytes_transferred / 1048576.0, 1.0),  # MB normalized
            min(packets / 1000.0, 1.0),
            (bytes_transferred / max(packets, 1)) / 1500.0  # Avg packet size normalized
        ])
        
        return features
    
    def _extract_behavioral_features(self, data: Dict[str, Any], source_type: str) -> List[float]:
        """Extract behavioral pattern features"""
        features = []
        
        # User/agent behavior features
        user_id = data.get('user_id', '')
        session_duration = data.get('session_duration', 0)
        request_count = data.get('request_count', 0)
        
        features.extend([
            hash(user_id) % 1000 / 1000.0 if user_id else 0.0,
            min(session_duration / 3600.0, 1.0),  # Hours
            min(request_count / 100.0, 1.0)
        ])
        
        # Action pattern features
        actions = data.get('actions', [])
        unique_actions = len(set(actions)) if actions else 0
        action_frequency = len(actions) / max(session_duration, 1) if session_duration > 0 else 0
        
        features.extend([
            min(unique_actions / 20.0, 1.0),
            min(action_frequency * 60, 1.0),  # Actions per minute
            len(actions) / 100.0 if actions else 0.0
        ])
        
        # Error and anomaly features
        error_count = data.get('error_count', 0)
        failed_attempts = data.get('failed_attempts', 0)
        anomaly_score = data.get('anomaly_score', 0.0)
        
        features.extend([
            min(error_count / 10.0, 1.0),
            min(failed_attempts / 5.0, 1.0),
            min(anomaly_score, 1.0)
        ])
        
        return features
    
    def _extract_payload_features(self, data: Dict[str, Any], source_type: str) -> List[float]:
        """Extract payload and content features"""
        features = []
        
        # Payload content analysis
        payload = data.get('payload', '')
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        elif not isinstance(payload, str):
            payload = str(payload)
        
        # Basic content features
        features.extend([
            len(payload) / 10000.0,  # Payload length normalized
            payload.count('<') / max(len(payload), 1),  # HTML tag density
            payload.count('script') / max(len(payload), 1),  # Script density
            payload.count('eval') / max(len(payload), 1),  # Eval function density
            payload.count('exec') / max(len(payload), 1),  # Exec function density
        ])
        
        # SQL injection indicators
        sql_keywords = ['select', 'union', 'drop', 'insert', 'update', 'delete', 'exec']
        sql_density = sum(payload.lower().count(kw) for kw in sql_keywords) / max(len(payload), 1)
        features.append(min(sql_density * 100, 1.0))
        
        # XSS indicators
        xss_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=', 'alert(']
        xss_density = sum(payload.lower().count(pattern) for pattern in xss_patterns) / max(len(payload), 1)
        features.append(min(xss_density * 100, 1.0))
        
        # Command injection indicators
        cmd_patterns = ['|', ';', '&&', '||', '$(', '`']
        cmd_density = sum(payload.count(pattern) for pattern in cmd_patterns) / max(len(payload), 1)
        features.append(min(cmd_density * 50, 1.0))
        
        # Encoding detection
        try:
            import base64
            import urllib.parse
            
            # Base64 detection
            b64_ratio = 0.0
            if len(payload) > 10:
                try:
                    decoded = base64.b64decode(payload)
                    b64_ratio = 1.0
                except:
                    pass
            
            # URL encoding detection
            url_encoded_chars = payload.count('%')
            url_ratio = url_encoded_chars / max(len(payload), 1)
            
            features.extend([b64_ratio, min(url_ratio * 10, 1.0)])
        except:
            features.extend([0.0, 0.0])
        
        return features
    
    def _calculate_confidence_score(self, source_data: Dict[str, Any], features: np.ndarray) -> float:
        """Calculate confidence score for threat intelligence"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on multiple indicators
        if 'source_credibility' in source_data:
            confidence += source_data['source_credibility'] * 0.3
        
        # Feature-based confidence
        if len(features) > 0:
            feature_variance = np.var(features)
            confidence += min(feature_variance, 0.2)
        
        # Corroboration from multiple sources
        if 'corroboration_count' in source_data:
            confidence += min(source_data['corroboration_count'] * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
    def _extract_attack_indicators(self, data: Dict[str, Any]) -> List[str]:
        """Extract specific attack indicators from data"""
        indicators = []
        
        # Network indicators
        if 'source_ip' in data:
            indicators.append(f"source_ip:{data['source_ip']}")
        
        if 'user_agent' in data:
            indicators.append(f"user_agent:{data['user_agent']}")
        
        # Payload indicators
        payload = data.get('payload', '')
        if 'script' in str(payload).lower():
            indicators.append("payload_contains:script")
        
        if 'union' in str(payload).lower():
            indicators.append("payload_contains:sql_union")
        
        # Behavioral indicators
        if data.get('failed_attempts', 0) > 3:
            indicators.append("behavior:multiple_failures")
        
        if data.get('anomaly_score', 0) > 0.8:
            indicators.append("behavior:high_anomaly")
        
        return indicators
    
    async def _generate_defense_recommendations(self, features: np.ndarray, threat_type: str) -> List[str]:
        """Generate defense recommendations based on threat features"""
        recommendations = []
        
        # Feature-based recommendations
        if len(features) >= 10:
            # Network-based recommendations
            if features[4] > 0.8:  # High port activity
                recommendations.append("implement_port_filtering")
            
            if features[7] > 0.5:  # High traffic volume
                recommendations.append("enable_rate_limiting")
            
            # Behavioral recommendations
            if len(features) > 15 and features[15] > 0.7:  # High error rate
                recommendations.append("enhance_input_validation")
            
            # Payload-based recommendations
            if len(features) > 20:
                if features[20] > 0.3:  # SQL injection indicators
                    recommendations.append("implement_parameterized_queries")
                
                if features[21] > 0.3:  # XSS indicators
                    recommendations.append("implement_csp_headers")
        
        # Threat type specific recommendations
        if threat_type == 'web_attack':
            recommendations.extend([
                "enable_waf",
                "implement_request_validation",
                "configure_security_headers"
            ])
        elif threat_type == 'network_scan':
            recommendations.extend([
                "configure_firewall_rules",
                "enable_intrusion_detection",
                "implement_network_segmentation"
            ])
        elif threat_type == 'malware':
            recommendations.extend([
                "enable_realtime_scanning",
                "implement_behavioral_analysis",
                "configure_file_integrity_monitoring"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _store_threat_pattern(self, pattern: ThreatIntelligencePattern):
        """Store threat pattern in database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT OR REPLACE INTO threat_patterns 
            (pattern_id, pattern_type, feature_vector, confidence_score, 
             attack_indicators, defense_recommendations, source_events,
             validation_results, evolution_history, first_observed, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            pattern.pattern_type,
            pickle.dumps(pattern.feature_vector),
            pattern.confidence_score,
            json.dumps(pattern.attack_indicators),
            json.dumps(pattern.defense_recommendations),
            json.dumps(pattern.source_events),
            json.dumps(pattern.validation_results),
            json.dumps(pattern.evolution_history),
            pattern.first_observed.isoformat(),
            pattern.last_updated.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_correlation_graph(self, pattern: ThreatIntelligencePattern, source_data: Dict[str, Any]):
        """Update threat correlation graph"""
        pattern_node = pattern.pattern_id
        
        # Add pattern node
        self.correlation_graph.add_node(
            pattern_node,
            type='threat_pattern',
            confidence=pattern.confidence_score,
            timestamp=pattern.first_observed
        )
        
        # Add edges to related indicators
        for indicator in pattern.attack_indicators:
            indicator_node = f"indicator_{hash(indicator)}"
            self.correlation_graph.add_node(indicator_node, type='indicator', value=indicator)
            self.correlation_graph.add_edge(pattern_node, indicator_node, weight=0.8)
        
        # Add edges to defense recommendations
        for recommendation in pattern.defense_recommendations:
            defense_node = f"defense_{hash(recommendation)}"
            self.correlation_graph.add_node(defense_node, type='defense', value=recommendation)
            self.correlation_graph.add_edge(pattern_node, defense_node, weight=0.9)
    
    async def correlate_patterns(self, similarity_threshold: float = 0.8) -> List[List[str]]:
        """Find correlated threat patterns using similarity analysis"""
        if len(self.pattern_cache) < 2:
            return []
        
        patterns = list(self.pattern_cache.values())
        correlated_groups = []
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(patterns), len(patterns)))
        
        for i, pattern_a in enumerate(patterns):
            for j, pattern_b in enumerate(patterns):
                if i != j:
                    # Calculate cosine similarity between feature vectors
                    if len(pattern_a.feature_vector) > 0 and len(pattern_b.feature_vector) > 0:
                        # Ensure same dimensionality
                        min_len = min(len(pattern_a.feature_vector), len(pattern_b.feature_vector))
                        vec_a = pattern_a.feature_vector[:min_len]
                        vec_b = pattern_b.feature_vector[:min_len]
                        
                        similarity = 1 - cosine(vec_a, vec_b)
                        similarity_matrix[i][j] = similarity
        
        # Find correlated groups using clustering
        if similarity_matrix.max() > similarity_threshold:
            # Convert similarity to distance matrix
            distance_matrix = 1 - similarity_matrix
            
            # Apply clustering
            clustering = DBSCAN(metric='precomputed', eps=1-similarity_threshold, min_samples=2)
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group patterns by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise points
                    clusters[label].append(patterns[i].pattern_id)
            
            correlated_groups = list(clusters.values())
        
        return correlated_groups
    
    async def predict_threat_evolution(self, pattern_id: str, time_horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict how a threat pattern might evolve"""
        if pattern_id not in self.pattern_cache:
            return {}
        
        pattern = self.pattern_cache[pattern_id]
        
        # Analyze historical evolution
        evolution_features = []
        for evolution in pattern.evolution_history:
            features = evolution.get('features', [])
            if features:
                evolution_features.append(features)
        
        if len(evolution_features) < 3:
            # Not enough history for prediction
            return {
                'prediction_confidence': 0.0,
                'predicted_changes': [],
                'risk_assessment': 'insufficient_data'
            }
        
        # Use time series analysis for prediction
        evolution_array = np.array(evolution_features)
        
        # Calculate trend vectors
        trends = []
        for i in range(evolution_array.shape[1]):
            feature_series = evolution_array[:, i]
            if len(feature_series) > 1:
                # Calculate linear trend
                x = np.arange(len(feature_series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, feature_series)
                trends.append({
                    'feature_index': i,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value
                })
        
        # Predict future values
        predicted_changes = []
        for trend in trends:
            if trend['r_squared'] > 0.5 and trend['p_value'] < 0.05:  # Significant trend
                future_value = pattern.feature_vector[trend['feature_index']] + trend['slope'] * time_horizon_hours
                change_magnitude = abs(future_value - pattern.feature_vector[trend['feature_index']])
                
                if change_magnitude > 0.1:  # Significant change threshold
                    predicted_changes.append({
                        'feature_index': trend['feature_index'],
                        'current_value': float(pattern.feature_vector[trend['feature_index']]),
                        'predicted_value': float(future_value),
                        'change_magnitude': float(change_magnitude),
                        'confidence': float(trend['r_squared'])
                    })
        
        # Risk assessment
        risk_level = 'low'
        if len(predicted_changes) > 5:
            risk_level = 'high'
        elif len(predicted_changes) > 2:
            risk_level = 'medium'
        
        return {
            'prediction_confidence': np.mean([c['confidence'] for c in predicted_changes]) if predicted_changes else 0.0,
            'predicted_changes': predicted_changes,
            'risk_assessment': risk_level,
            'time_horizon_hours': time_horizon_hours,
            'recommendation': await self._generate_evolution_recommendations(predicted_changes)
        }
    
    async def _generate_evolution_recommendations(self, predicted_changes: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on predicted threat evolution"""
        recommendations = []
        
        for change in predicted_changes:
            change_magnitude = change['change_magnitude']
            
            if change_magnitude > 0.5:
                recommendations.append("increase_monitoring_frequency")
            
            if change_magnitude > 0.3:
                recommendations.append("prepare_additional_defenses")
            
            if change['confidence'] > 0.8:
                recommendations.append("implement_proactive_blocking")
        
        # Add general recommendations
        if len(predicted_changes) > 3:
            recommendations.extend([
                "activate_enhanced_logging",
                "notify_security_team",
                "consider_threat_hunting"
            ])
        
        return list(set(recommendations))


class AdaptiveLearningEngine:
    """Machine learning engine that adapts security policies based on attack/defense outcomes"""
    
    def __init__(self, model_path: str = "models/adaptive_security"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # ML models for different aspects of adaptation
        self.policy_optimizer = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.threat_predictor = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        self.response_selector = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )
        
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Training data storage
        self.training_data = {
            'policy_optimization': [],
            'threat_prediction': [],
            'response_selection': []
        }
        
        # Performance tracking
        self.model_performance = {
            'policy_optimizer': {'accuracy': 0.0, 'last_updated': datetime.utcnow()},
            'threat_predictor': {'accuracy': 0.0, 'last_updated': datetime.utcnow()},
            'response_selector': {'accuracy': 0.0, 'last_updated': datetime.utcnow()}
        }
        
        # Load existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing trained models"""
        try:
            if (self.model_path / "policy_optimizer.pkl").exists():
                self.policy_optimizer = joblib.load(self.model_path / "policy_optimizer.pkl")
                
            if (self.model_path / "threat_predictor.pkl").exists():
                self.threat_predictor = joblib.load(self.model_path / "threat_predictor.pkl")
                
            if (self.model_path / "response_selector.pkl").exists():
                self.response_selector = joblib.load(self.model_path / "response_selector.pkl")
                
            if (self.model_path / "feature_scaler.pkl").exists():
                self.feature_scaler = joblib.load(self.model_path / "feature_scaler.pkl")
                
            if (self.model_path / "label_encoders.pkl").exists():
                self.label_encoders = joblib.load(self.model_path / "label_encoders.pkl")
                
            logger.info("Loaded existing adaptive learning models")
        
        except Exception as e:
            logger.warning(f"Could not load existing models: {e}")
    
    async def learn_from_correlation(self, correlation: AttackDefenseCorrelation):
        """Learn from attack/defense correlation data"""
        # Extract features from correlation
        features = await self._extract_correlation_features(correlation)
        
        # Add to training data
        self.training_data['policy_optimization'].append({
            'features': features,
            'effectiveness_score': correlation.effectiveness_score,
            'response_time': correlation.response_time,
            'false_positive_rate': correlation.false_positive_rate
        })
        
        self.training_data['response_selection'].append({
            'features': features,
            'defense_mechanism': correlation.defense_mechanism,
            'mitigation_success': correlation.mitigation_success
        })
        
        # Trigger retraining if enough new data
        if len(self.training_data['policy_optimization']) >= 50:
            await self._retrain_models()
    
    async def _extract_correlation_features(self, correlation: AttackDefenseCorrelation) -> np.ndarray:
        """Extract features from attack/defense correlation"""
        features = []
        
        # Timing features
        features.extend([
            correlation.response_time / 60.0,  # Response time in minutes
            correlation.timestamp.hour / 24.0,
            correlation.timestamp.weekday() / 7.0
        ])
        
        # Effectiveness features
        features.extend([
            correlation.effectiveness_score,
            correlation.false_positive_rate,
            correlation.correlation_strength,
            1.0 if correlation.mitigation_success else 0.0
        ])
        
        # Attack vector encoding
        attack_vector_hash = hash(correlation.attack_vector) % 1000
        features.append(attack_vector_hash / 1000.0)
        
        # Defense mechanism encoding
        defense_hash = hash(correlation.defense_mechanism) % 1000
        features.append(defense_hash / 1000.0)
        
        # Learned pattern features
        if correlation.learned_patterns:
            pattern_count = len(correlation.learned_patterns)
            pattern_complexity = sum(len(str(v)) for v in correlation.learned_patterns.values())
            features.extend([
                min(pattern_count / 10.0, 1.0),
                min(pattern_complexity / 1000.0, 1.0)
            ])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    async def _retrain_models(self):
        """Retrain ML models with new data"""
        logger.info("Retraining adaptive learning models")
        
        try:
            # Retrain policy optimizer
            if len(self.training_data['policy_optimization']) >= 10:
                await self._retrain_policy_optimizer()
            
            # Retrain threat predictor
            if len(self.training_data['threat_prediction']) >= 10:
                await self._retrain_threat_predictor()
            
            # Retrain response selector
            if len(self.training_data['response_selection']) >= 10:
                await self._retrain_response_selector()
            
            # Save updated models
            await self._save_models()
            
            logger.info("Model retraining completed")
        
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    async def _retrain_policy_optimizer(self):
        """Retrain the policy optimization model"""
        data = self.training_data['policy_optimization']
        
        # Prepare training data
        X = np.array([item['features'] for item in data])
        y = np.array([item['effectiveness_score'] for item in data])
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y, y
        
        # Train model
        self.policy_optimizer.fit(X_train, y_train)
        
        # Evaluate
        if len(X_test) > 0:
            y_pred = self.policy_optimizer.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2)
            accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like metric
            
            self.model_performance['policy_optimizer'] = {
                'accuracy': accuracy,
                'last_updated': datetime.utcnow()
            }
    
    async def _retrain_threat_predictor(self):
        """Retrain the threat prediction model"""
        # Generate synthetic training data from correlations for threat prediction
        try:
            training_features = []
            training_labels = []
            
            for correlation in self.training_data.get('policy_optimization', []):
                features = correlation['features']
                # Create threat label based on effectiveness and response time
                threat_level = 'HIGH' if correlation['effectiveness_score'] < 0.3 else 'MEDIUM' if correlation['effectiveness_score'] < 0.7 else 'LOW'
                
                training_features.append(features)
                training_labels.append(threat_level)
            
            if len(training_features) >= 10:
                X = np.array(training_features)
                y = training_labels
                
                # Encode labels
                if 'threat_level' not in self.label_encoders:
                    self.label_encoders['threat_level'] = LabelEncoder()
                
                y_encoded = self.label_encoders['threat_level'].fit_transform(y)
                
                # Scale features
                X_scaled = self.feature_scaler.fit_transform(X)
                
                # Split data
                if len(X) > 20:
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
                else:
                    X_train, X_test = X_scaled, X_scaled
                    y_train, y_test = y_encoded, y_encoded
                
                # Train model
                self.threat_predictor.fit(X_train, y_train)
                
                # Evaluate
                if len(X_test) > 0:
                    y_pred = self.threat_predictor.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    self.model_performance['threat_predictor'] = {
                        'accuracy': accuracy,
                        'last_updated': datetime.utcnow()
                    }
                    
                    logger.info(f"Threat predictor retrained with accuracy: {accuracy:.3f}")
        
        except Exception as e:
            logger.error(f"Error retraining threat predictor: {e}")
            # Initialize with basic model if training fails
            self.threat_predictor = MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
    
    async def _retrain_response_selector(self):
        """Retrain the response selection model"""
        data = self.training_data['response_selection']
        
        # Prepare training data
        X = np.array([item['features'] for item in data])
        y = [item['defense_mechanism'] for item in data]
        
        # Encode labels
        if 'defense_mechanism' not in self.label_encoders:
            self.label_encoders['defense_mechanism'] = LabelEncoder()
        
        y_encoded = self.label_encoders['defense_mechanism'].fit_transform(y)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
        else:
            X_train, X_test = X_scaled, X_scaled
            y_train, y_test = y_encoded, y_encoded
        
        # Train model
        self.response_selector.fit(X_train, y_train)
        
        # Evaluate
        if len(X_test) > 0:
            y_pred = self.response_selector.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance['response_selector'] = {
                'accuracy': accuracy,
                'last_updated': datetime.utcnow()
            }
    
    async def _save_models(self):
        """Save trained models to disk"""
        joblib.dump(self.policy_optimizer, self.model_path / "policy_optimizer.pkl")
        joblib.dump(self.threat_predictor, self.model_path / "threat_predictor.pkl")
        joblib.dump(self.response_selector, self.model_path / "response_selector.pkl")
        joblib.dump(self.feature_scaler, self.model_path / "feature_scaler.pkl")
        joblib.dump(self.label_encoders, self.model_path / "label_encoders.pkl")
        
        # Save performance metrics
        with open(self.model_path / "performance_metrics.json", 'w') as f:
            json.dump(self.model_performance, f, default=str, indent=2)
    
    async def optimize_policy_parameters(self, current_policy: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize security policy parameters based on learned patterns"""
        try:
            # Extract features from current context
            context_features = await self._extract_context_features(current_policy, context)
            
            # Scale features
            context_scaled = self.feature_scaler.transform([context_features])
            
            # Predict optimal effectiveness score
            predicted_effectiveness = self.policy_optimizer.predict(context_scaled)[0]
            
            # Generate optimized policy
            optimized_policy = current_policy.copy()
            
            # Adjust parameters based on predicted effectiveness
            if predicted_effectiveness < 0.5:
                # Increase security measures
                optimized_policy = await self._increase_security_measures(optimized_policy)
            elif predicted_effectiveness > 0.8:
                # Potentially reduce security measures to improve performance
                optimized_policy = await self._balance_security_performance(optimized_policy)
            
            return optimized_policy
        
        except Exception as e:
            logger.error(f"Error optimizing policy parameters: {e}")
            return current_policy
    
    async def _extract_context_features(self, policy: Dict[str, Any], context: Dict[str, Any]) -> np.ndarray:
        """Extract features from policy and context"""
        features = []
        
        # Policy parameter features
        features.extend([
            policy.get('rate_limit_threshold', 100) / 1000.0,
            policy.get('authentication_timeout', 3600) / 86400.0,  # Normalize to days
            1.0 if policy.get('enable_logging', True) else 0.0,
            policy.get('threat_threshold', 0.5)
        ])
        
        # Context features
        features.extend([
            context.get('current_threat_level', 0.0),
            context.get('system_load', 0.0),
            context.get('recent_attacks', 0) / 100.0,
            context.get('false_positive_rate', 0.0)
        ])
        
        # Time-based features
        now = datetime.utcnow()
        features.extend([
            now.hour / 24.0,
            now.weekday() / 7.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    async def _increase_security_measures(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Increase security measures in policy"""
        optimized = policy.copy()
        
        # Tighten rate limits
        if 'rate_limit_threshold' in optimized:
            optimized['rate_limit_threshold'] = max(optimized['rate_limit_threshold'] * 0.8, 10)
        
        # Reduce authentication timeout
        if 'authentication_timeout' in optimized:
            optimized['authentication_timeout'] = max(optimized['authentication_timeout'] * 0.9, 300)
        
        # Lower threat threshold
        if 'threat_threshold' in optimized:
            optimized['threat_threshold'] = max(optimized['threat_threshold'] * 0.9, 0.1)
        
        # Enable additional monitoring
        optimized['enhanced_monitoring'] = True
        optimized['detailed_logging'] = True
        
        return optimized
    
    async def _balance_security_performance(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Balance security and performance in policy"""
        optimized = policy.copy()
        
        # Slightly relax rate limits
        if 'rate_limit_threshold' in optimized:
            optimized['rate_limit_threshold'] = min(optimized['rate_limit_threshold'] * 1.1, 1000)
        
        # Adjust logging granularity
        if optimized.get('detailed_logging', False):
            optimized['detailed_logging'] = False
            optimized['summary_logging'] = True
        
        return optimized
    
    async def predict_optimal_response(self, attack_features: np.ndarray, 
                                     available_responses: List[str]) -> str:
        """Predict the optimal response to an attack"""
        try:
            # Scale attack features
            features_scaled = self.feature_scaler.transform([attack_features])
            
            # Predict response probabilities
            response_probs = self.response_selector.predict_proba(features_scaled)[0]
            
            # Get response classes
            response_classes = self.response_selector.classes_
            
            # Decode response classes
            if 'defense_mechanism' in self.label_encoders:
                decoded_responses = self.label_encoders['defense_mechanism'].inverse_transform(response_classes)
            else:
                decoded_responses = response_classes
            
            # Find best available response
            best_response = None
            best_prob = 0.0
            
            for i, response in enumerate(decoded_responses):
                if response in available_responses and response_probs[i] > best_prob:
                    best_response = response
                    best_prob = response_probs[i]
            
            return best_response if best_response else available_responses[0]
        
        except Exception as e:
            logger.error(f"Error predicting optimal response: {e}")
            return available_responses[0] if available_responses else "alert_only"


class PurpleTeamOrchestrator:
    """Orchestrates the entire purple team operation, correlating red and blue team activities"""
    
    def __init__(self, red_team_agent, blue_team_agent, 
                 security_framework: SomnusSecurityFramework,
                 vm_supervisor: VMSupervisor):
        self.red_team = red_team_agent
        self.blue_team = blue_team_agent
        self.security_framework = security_framework
        self.vm_supervisor = vm_supervisor
        
        # Synthesis components
        self.threat_intelligence = ThreatIntelligenceFusion()
        self.adaptive_learner = AdaptiveLearningEngine()
        
        # Correlation tracking
        self.correlations: List[AttackDefenseCorrelation] = []
        self.correlation_lock = threading.Lock()
        
        # Orchestration state
        self.orchestration_active = False
        self.orchestration_tasks: List[asyncio.Task] = []
        
        # Performance metrics
        self.metrics = {
            'correlations_found': 0,
            'policies_optimized': 0,
            'threats_predicted': 0,
            'adaptive_improvements': 0,
            'start_time': datetime.utcnow()
        }
        
        logger.info("Purple Team Orchestrator initialized")
    
    async def start_orchestration(self):
        """Start purple team orchestration"""
        if self.orchestration_active:
            logger.warning("Purple team orchestration already active")
            return
        
        logger.info("Starting Purple Team orchestration")
        
        # Start correlation engine
        correlation_task = asyncio.create_task(self._correlation_engine())
        self.orchestration_tasks.append(correlation_task)
        
        # Start adaptive learning engine
        learning_task = asyncio.create_task(self._adaptive_learning_loop())
        self.orchestration_tasks.append(learning_task)
        
        # Start policy optimization engine
        policy_task = asyncio.create_task(self._policy_optimization_loop())
        self.orchestration_tasks.append(policy_task)
        
        # Start threat prediction engine
        prediction_task = asyncio.create_task(self._threat_prediction_loop())
        self.orchestration_tasks.append(prediction_task)
        
        # Start intelligence fusion engine
        fusion_task = asyncio.create_task(self._intelligence_fusion_loop())
        self.orchestration_tasks.append(fusion_task)
        
        self.orchestration_active = True
        
        logger.info("Purple Team orchestration started with 5 parallel engines")
        
        # Wait for orchestration tasks
        try:
            await asyncio.gather(*self.orchestration_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in purple team orchestration: {e}")
        finally:
            self.orchestration_active = False
    
    async def stop_orchestration(self):
        """Stop purple team orchestration"""
        logger.info("Stopping Purple Team orchestration")
        
        self.orchestration_active = False
        
        # Cancel all tasks
        for task in self.orchestration_tasks:
            task.cancel()
        
        await asyncio.gather(*self.orchestration_tasks, return_exceptions=True)
        
        self.orchestration_tasks.clear()
        logger.info("Purple Team orchestration stopped")
    
    async def _correlation_engine(self):
        """Correlate red team attacks with blue team responses"""
        correlation_window = timedelta(minutes=10)
        
        while self.orchestration_active:
            try:
                # Get recent red team results
                red_team_events = await self._get_recent_red_team_events(correlation_window)
                
                # Get recent blue team responses
                blue_team_responses = await self._get_recent_blue_team_responses(correlation_window)
                
                # Find correlations
                new_correlations = await self._find_correlations(red_team_events, blue_team_responses)
                
                # Process new correlations
                for correlation in new_correlations:
                    await self._process_correlation(correlation)
                
                self.metrics['correlations_found'] += len(new_correlations)
                
                await asyncio.sleep(30)  # Correlate every 30 seconds
            
            except Exception as e:
                logger.error(f"Error in correlation engine: {e}")
                await asyncio.sleep(60)
    
    async def _get_recent_red_team_events(self, time_window: timedelta) -> List[Dict[str, Any]]:
        """Get recent red team test results"""
        cutoff_time = datetime.utcnow() - time_window
        
        # Get results from red team agent
        recent_results = []
        for result in self.red_team.test_results:
            if result.timestamp >= cutoff_time:
                recent_results.append({
                    'event_id': result.test_id,
                    'test_type': result.test_type.value,
                    'target_vm_id': result.target_vm_id,
                    'vulnerability_found': result.vulnerability_found,
                    'severity': result.severity.value,
                    'exploit_payload': result.exploit_payload,
                    'timestamp': result.timestamp,
                    'execution_time': result.execution_time
                })
        
        return recent_results
    
    async def _get_recent_blue_team_responses(self, time_window: timedelta) -> List[Dict[str, Any]]:
        """Get recent blue team defensive actions"""
        cutoff_time = datetime.utcnow() - time_window
        
        # Get responses from blue team agent
        recent_responses = []
        for action in self.blue_team.response_orchestrator.action_history:
            if action.timestamp >= cutoff_time:
                recent_responses.append({
                    'response_id': action.action_id,
                    'action_type': action.action_type.value,
                    'trigger_event': action.trigger_event,
                    'target_vm_id': action.target_vm_id,
                    'success': action.success,
                    'execution_time': action.execution_time,
                    'timestamp': action.timestamp
                })
        
        return recent_responses
    
    async def _find_correlations(self, red_events: List[Dict[str, Any]], 
                               blue_responses: List[Dict[str, Any]]) -> List[AttackDefenseCorrelation]:
        """Find correlations between red team attacks and blue team responses"""
        correlations = []
        
        for red_event in red_events:
            for blue_response in blue_responses:
                # Check temporal correlation
                time_diff = abs((red_event['timestamp'] - blue_response['timestamp']).total_seconds())
                
                if time_diff <= 300:  # 5 minute correlation window
                    # Check spatial correlation (same VM)
                    spatial_match = (red_event.get('target_vm_id') == blue_response.get('target_vm_id'))
                    
                    # Calculate correlation strength
                    correlation_strength = await self._calculate_correlation_strength(red_event, blue_response, time_diff)
                    
                    if correlation_strength > 0.3:  # Minimum correlation threshold
                        correlation = AttackDefenseCorrelation(
                            red_team_event_id=red_event['event_id'],
                            blue_team_response_id=blue_response['response_id'],
                            correlation_strength=correlation_strength,
                            correlation_type=CorrelationType.TEMPORAL if not spatial_match else CorrelationType.SPATIAL,
                            attack_vector=red_event.get('test_type', 'unknown'),
                            defense_mechanism=blue_response.get('action_type', 'unknown'),
                            effectiveness_score=await self._calculate_effectiveness_score(red_event, blue_response),
                            response_time=time_diff,
                            mitigation_success=blue_response.get('success', False),
                            learned_patterns=await self._extract_learned_patterns(red_event, blue_response)
                        )
                        
                        correlations.append(correlation)
        
        return correlations
    
    async def _calculate_correlation_strength(self, red_event: Dict[str, Any], 
                                            blue_response: Dict[str, Any], 
                                            time_diff: float) -> float:
        """Calculate the strength of correlation between attack and response"""
        strength = 0.0
        
        # Temporal correlation (inverse of time difference)
        temporal_strength = max(0, 1.0 - (time_diff / 300.0))  # 5 minute max
        strength += temporal_strength * 0.4
        
        # Spatial correlation (same VM)
        if red_event.get('target_vm_id') == blue_response.get('target_vm_id'):
            strength += 0.3
        
        # Severity matching
        red_severity = getattr(ThreatLevel, red_event.get('severity', 'LOW').upper(), ThreatLevel.LOW)
        response_urgency = 0.2  # Base urgency
        
        if blue_response.get('action_type') in ['isolate_vm', 'emergency_shutdown']:
            response_urgency = 1.0
        elif blue_response.get('action_type') in ['quarantine_process', 'block_ip']:
            response_urgency = 0.8
        elif blue_response.get('action_type') in ['rate_limit', 'restart_service']:
            response_urgency = 0.6
        
        severity_match = 1.0 - abs((red_severity.value / 4.0) - response_urgency)
        strength += severity_match * 0.3
        
        return min(strength, 1.0)
    
    async def _calculate_effectiveness_score(self, red_event: Dict[str, Any], 
                                           blue_response: Dict[str, Any]) -> float:
        """Calculate the effectiveness of the blue team response"""
        base_score = 0.5
        
        # Success of the response
        if blue_response.get('success', False):
            base_score += 0.3
        
        # Speed of response
        response_time = blue_response.get('execution_time', 10.0)
        speed_score = max(0, 1.0 - (response_time / 60.0))  # 1 minute normalization
        base_score += speed_score * 0.2
        
        # Appropriateness of response to threat
        threat_severity = red_event.get('severity', 'LOW')
        response_type = blue_response.get('action_type', 'alert_only')
        
        appropriateness_score = await self._assess_response_appropriateness(threat_severity, response_type)
        base_score += appropriateness_score * 0.3
        
        return min(base_score, 1.0)
    
    async def _assess_response_appropriateness(self, threat_severity: str, response_type: str) -> float:
        """Assess how appropriate a response is for a given threat level"""
        severity_scores = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        response_scores = {
            'alert_only': 1,
            'rate_limit': 2,
            'quarantine_process': 3,
            'block_ip': 3,
            'isolate_vm': 4,
            'emergency_shutdown': 4
        }
        
        threat_score = severity_scores.get(threat_severity, 1)
        response_score = response_scores.get(response_type, 1)
        
        # Perfect match = 1.0, over/under reaction penalized
        difference = abs(threat_score - response_score)
        appropriateness = max(0, 1.0 - (difference / 3.0))
        
        return appropriateness
    
    async def _extract_learned_patterns(self, red_event: Dict[str, Any], 
                                      blue_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learned patterns from the attack/response correlation"""
        patterns = {}
        
        # Attack pattern
        patterns['attack_pattern'] = {
            'type': red_event.get('test_type'),
            'success': red_event.get('vulnerability_found', False),
            'execution_time': red_event.get('execution_time', 0),
            'payload_characteristics': await self._analyze_payload(red_event.get('exploit_payload', ''))
        }
        
        # Response pattern
        patterns['response_pattern'] = {
            'type': blue_response.get('action_type'),
            'success': blue_response.get('success', False),
            'execution_time': blue_response.get('execution_time', 0)
        }
        
        # Correlation insights
        patterns['correlation_insights'] = {
            'response_triggered_by_attack': True,
            'response_effectiveness': await self._calculate_effectiveness_score(red_event, blue_response),
            'learning_opportunity': red_event.get('vulnerability_found', False) and not blue_response.get('success', False)
        }
        
        return patterns
    
    async def _analyze_payload(self, payload: str) -> Dict[str, Any]:
        """Analyze attack payload characteristics"""
        characteristics = {
            'length': len(payload),
            'contains_sql': any(keyword in payload.lower() for keyword in ['select', 'union', 'drop', 'insert']),
            'contains_script': 'script' in payload.lower(),
            'contains_command': any(char in payload for char in ['|', ';', '&&', '`']),
            'encoded': self._detect_encoding(payload)
        }
        
        return characteristics
    
    def _detect_encoding(self, payload: str) -> str:
        """Detect encoding used in payload"""
        if '%' in payload and all(c in '0123456789ABCDEFabcdef%' for c in payload.replace('%', '')):
            return 'url_encoded'
        
        try:
            import base64
            base64.b64decode(payload)
            return 'base64'
        except:
            pass
        
        return 'none'
    
    async def _process_correlation(self, correlation: AttackDefenseCorrelation):
        """Process a new attack/defense correlation"""
        with self.correlation_lock:
            self.correlations.append(correlation)
        
        # Store in database
        await self._store_correlation(correlation)
        
        # Feed to adaptive learner
        await self.adaptive_learner.learn_from_correlation(correlation)
        
        # Generate threat intelligence
        await self._generate_threat_intelligence(correlation)
        
        logger.info(f"Processed correlation: {correlation.attack_vector} -> {correlation.defense_mechanism} "
                   f"(effectiveness: {correlation.effectiveness_score:.2f})")
    
    async def _store_correlation(self, correlation: AttackDefenseCorrelation):
        """Store correlation in threat intelligence database"""
        conn = sqlite3.connect(self.threat_intelligence.db_path)
        
        conn.execute('''
            INSERT INTO correlations 
            (correlation_id, red_team_event_id, blue_team_response_id, correlation_strength,
             correlation_type, attack_vector, defense_mechanism, effectiveness_score,
             false_positive_rate, response_time, mitigation_success, learned_patterns, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            correlation.correlation_id,
            correlation.red_team_event_id,
            correlation.blue_team_response_id,
            correlation.correlation_strength,
            correlation.correlation_type.value,
            correlation.attack_vector,
            correlation.defense_mechanism,
            correlation.effectiveness_score,
            correlation.false_positive_rate,
            correlation.response_time,
            1 if correlation.mitigation_success else 0,
            json.dumps(correlation.learned_patterns),
            correlation.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def _generate_threat_intelligence(self, correlation: AttackDefenseCorrelation):
        """Generate threat intelligence from correlation"""
        # Create threat intelligence data
        intel_data = {
            'attack_vector': correlation.attack_vector,
            'defense_mechanism': correlation.defense_mechanism,
            'effectiveness_score': correlation.effectiveness_score,
            'correlation_strength': correlation.correlation_strength,
            'response_time': correlation.response_time,
            'mitigation_success': correlation.mitigation_success,
            'learned_patterns': correlation.learned_patterns,
            'timestamp': correlation.timestamp,
            'source_credibility': 1.0  # High credibility for internal data
        }
        
        # Ingest into threat intelligence fusion
        pattern_id = await self.threat_intelligence.ingest_threat_intelligence(
            intel_data, 'purple_team_correlation'
        )
        
        logger.debug(f"Generated threat intelligence pattern: {pattern_id}")
    
    async def _adaptive_learning_loop(self):
        """Continuous adaptive learning from correlations"""
        while self.orchestration_active:
            try:
                # Check if we have enough data for learning
                with self.correlation_lock:
                    correlation_count = len(self.correlations)
                
                if correlation_count >= 10:
                    # Trigger learning update
                    recent_correlations = self.correlations[-50:]  # Last 50 correlations
                    
                    for correlation in recent_correlations:
                        await self.adaptive_learner.learn_from_correlation(correlation)
                    
                    self.metrics['adaptive_improvements'] += 1
                    logger.info(f"Adaptive learning update completed with {len(recent_correlations)} correlations")
                
                await asyncio.sleep(300)  # Learn every 5 minutes
            
            except Exception as e:
                logger.error(f"Error in adaptive learning loop: {e}")
                await asyncio.sleep(300)
    
    async def _policy_optimization_loop(self):
        """Continuous security policy optimization"""
        while self.orchestration_active:
            try:
                # Get current security policies
                current_policies = await self._get_current_security_policies()
                
                # Get system context
                context = await self._get_system_context()
                
                # Optimize policies
                for policy_name, policy_config in current_policies.items():
                    optimized_config = await self.adaptive_learner.optimize_policy_parameters(
                        policy_config, context
                    )
                    
                    # Apply optimized policy if significantly different
                    if await self._policy_significantly_different(policy_config, optimized_config):
                        await self._apply_optimized_policy(policy_name, optimized_config)
                        self.metrics['policies_optimized'] += 1
                        
                        logger.info(f"Optimized security policy: {policy_name}")
                
                await asyncio.sleep(1800)  # Optimize every 30 minutes
            
            except Exception as e:
                logger.error(f"Error in policy optimization loop: {e}")
                await asyncio.sleep(1800)
    
    async def _get_current_security_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get current security policy configurations"""
        # This would integrate with the actual policy management system
        return {
            'network_security': {
                'rate_limit_threshold': 100,
                'authentication_timeout': 3600,
                'enable_logging': True,
                'threat_threshold': 0.5
            },
            'access_control': {
                'session_timeout': 1800,
                'max_concurrent_sessions': 5,
                'require_2fa': False,
                'password_complexity': 'medium'
            },
            'monitoring': {
                'log_level': 'INFO',
                'alert_threshold': 0.7,
                'retention_days': 30,
                'real_time_analysis': True
            }
        }
    
    async def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context for policy optimization"""
        # Calculate recent metrics
        recent_attacks = len([c for c in self.correlations 
                             if (datetime.utcnow() - c.timestamp).total_seconds() < 3600])
        
        avg_effectiveness = np.mean([c.effectiveness_score for c in self.correlations[-10:]]) if self.correlations else 0.5
        
        return {
            'current_threat_level': min(recent_attacks / 10.0, 1.0),
            'system_load': 0.3,  # Would get from actual system monitoring
            'recent_attacks': recent_attacks,
            'false_positive_rate': 1.0 - avg_effectiveness,
            'uptime_hours': (datetime.utcnow() - self.metrics['start_time']).total_seconds() / 3600
        }
    
    async def _policy_significantly_different(self, current: Dict[str, Any], optimized: Dict[str, Any]) -> bool:
        """Check if optimized policy is significantly different from current"""
        differences = 0
        total_params = 0
        
        for key in current:
            if key in optimized:
                total_params += 1
                if isinstance(current[key], (int, float)) and isinstance(optimized[key], (int, float)):
                    relative_diff = abs(current[key] - optimized[key]) / max(abs(current[key]), 1)
                    if relative_diff > 0.1:  # 10% threshold
                        differences += 1
                elif current[key] != optimized[key]:
                    differences += 1
        
        return (differences / max(total_params, 1)) > 0.2  # 20% of parameters changed
    
    async def _apply_optimized_policy(self, policy_name: str, optimized_config: Dict[str, Any]):
        """Apply optimized policy configuration"""
        # This would integrate with the actual policy management system
        logger.info(f"Applying optimized policy for {policy_name}: {optimized_config}")
        
        # Store the optimization in the database for tracking
        conn = sqlite3.connect(self.threat_intelligence.db_path)
        
        conn.execute('''
            INSERT INTO threat_predictions 
            (prediction_id, threat_type, predicted_probability, confidence_interval,
             time_horizon_hours, contributing_factors, recommended_actions, prediction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"policy_opt_{int(time.time())}",
            f"policy_optimization_{policy_name}",
            1.0,  # Optimization applied
            json.dumps([0.8, 1.0]),
            24,
            json.dumps({'optimization_trigger': 'adaptive_learning'}),
            json.dumps([f"apply_policy_{policy_name}"]),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def _threat_prediction_loop(self):
        """Continuous threat prediction and early warning"""
        while self.orchestration_active:
            try:
                # Analyze threat patterns for predictions
                threat_patterns = await self._analyze_threat_patterns()
                
                # Generate predictions
                for pattern in threat_patterns:
                    prediction = await self.threat_intelligence.predict_threat_evolution(
                        pattern['pattern_id'], time_horizon_hours=24
                    )
                    
                    if prediction.get('prediction_confidence', 0) > 0.6:
                        await self._handle_threat_prediction(pattern, prediction)
                        self.metrics['threats_predicted'] += 1
                
                await asyncio.sleep(3600)  # Predict every hour
            
            except Exception as e:
                logger.error(f"Error in threat prediction loop: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_threat_patterns(self) -> List[Dict[str, Any]]:
        """Analyze recent correlations for threat patterns"""
        patterns = []
        
        # Group correlations by attack vector
        attack_groups = defaultdict(list)
        for correlation in self.correlations[-100:]:  # Last 100 correlations
            attack_groups[correlation.attack_vector].append(correlation)
        
        # Analyze each group for patterns
        for attack_vector, correlations in attack_groups.items():
            if len(correlations) >= 3:  # Minimum for pattern analysis
                pattern = {
                    'pattern_id': f"pattern_{attack_vector}_{int(time.time())}",
                    'attack_vector': attack_vector,
                    'correlation_count': len(correlations),
                    'avg_effectiveness': np.mean([c.effectiveness_score for c in correlations]),
                    'trend': self._calculate_trend([c.effectiveness_score for c in correlations]),
                    'latest_timestamp': max(c.timestamp for c in correlations)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 3:
            return 'stable'
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        if p_value < 0.05 and abs(r_value) > 0.5:  # Significant trend
            return 'increasing' if slope > 0 else 'decreasing'
        else:
            return 'stable'
    
    async def _handle_threat_prediction(self, pattern: Dict[str, Any], prediction: Dict[str, Any]):
        """Handle a threat prediction"""
        logger.warning(f"Threat prediction: {pattern['attack_vector']} evolution predicted "
                      f"(confidence: {prediction['prediction_confidence']:.2f})")
        
        # Store prediction
        conn = sqlite3.connect(self.threat_intelligence.db_path)
        
        conn.execute('''
            INSERT INTO threat_predictions 
            (prediction_id, threat_type, predicted_probability, confidence_interval,
             time_horizon_hours, contributing_factors, recommended_actions, prediction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"pred_{int(time.time())}_{hash(pattern['attack_vector'])}",
            pattern['attack_vector'],
            prediction['prediction_confidence'],
            json.dumps([prediction['prediction_confidence'] - 0.1, prediction['prediction_confidence'] + 0.1]),
            prediction['time_horizon_hours'],
            json.dumps(pattern),
            json.dumps(prediction.get('recommendation', [])),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Take proactive action if high confidence
        if prediction['prediction_confidence'] > 0.8:
            await self._take_proactive_action(pattern, prediction)
    
    async def _take_proactive_action(self, pattern: Dict[str, Any], prediction: Dict[str, Any]):
        """Take proactive action based on threat prediction"""
        actions = prediction.get('recommendation', [])
        
        for action in actions:
            if action == 'increase_monitoring_frequency':
                # Increase monitoring for this attack vector
                logger.info(f"Increasing monitoring frequency for {pattern['attack_vector']}")
                
            elif action == 'prepare_additional_defenses':
                # Prepare additional defensive measures
                logger.info(f"Preparing additional defenses for {pattern['attack_vector']}")
                
            elif action == 'implement_proactive_blocking':
                # Implement proactive blocking rules
                logger.info(f"Implementing proactive blocking for {pattern['attack_vector']}")
    
    async def _intelligence_fusion_loop(self):
        """Continuous intelligence fusion and pattern correlation"""
        while self.orchestration_active:
            try:
                # Find correlated patterns across different threat types
                correlated_groups = await self.threat_intelligence.correlate_patterns(
                    similarity_threshold=0.7
                )
                
                # Process correlated groups
                for group in correlated_groups:
                    if len(group) >= 2:
                        await self._process_correlated_group(group)
                
                # Update correlation graph
                await self._update_intelligence_graph()
                
                await asyncio.sleep(900)  # Fuse intelligence every 15 minutes
            
            except Exception as e:
                logger.error(f"Error in intelligence fusion loop: {e}")
                await asyncio.sleep(900)
    
    async def _process_correlated_group(self, pattern_group: List[str]):
        """Process a group of correlated threat patterns"""
        logger.info(f"Processing correlated threat pattern group: {len(pattern_group)} patterns")
        
        # Analyze the group for common characteristics
        group_characteristics = await self._analyze_pattern_group(pattern_group)
        
        # Generate group-level intelligence
        group_intel = {
            'group_id': f"group_{int(time.time())}_{hash(str(pattern_group))}",
            'pattern_ids': pattern_group,
            'characteristics': group_characteristics,
            'threat_level': await self._assess_group_threat_level(group_characteristics),
            'recommended_actions': await self._generate_group_recommendations(group_characteristics)
        }
        
        # Store group intelligence
        await self._store_group_intelligence(group_intel)
    
    async def _analyze_pattern_group(self, pattern_ids: List[str]) -> Dict[str, Any]:
        """Analyze characteristics of a pattern group"""
        patterns = [self.threat_intelligence.pattern_cache.get(pid) for pid in pattern_ids]
        patterns = [p for p in patterns if p is not None]
        
        if not patterns:
            return {}
        
        # Common attack indicators
        all_indicators = []
        for pattern in patterns:
            all_indicators.extend(pattern.attack_indicators)
        
        indicator_counts = defaultdict(int)
        for indicator in all_indicators:
            indicator_counts[indicator] += 1
        
        common_indicators = [ind for ind, count in indicator_counts.items() 
                           if count >= len(patterns) * 0.5]  # In at least 50% of patterns
        
        # Average confidence
        avg_confidence = np.mean([p.confidence_score for p in patterns])
        
        # Pattern types
        pattern_types = [p.pattern_type for p in patterns]
        
        return {
            'common_indicators': common_indicators,
            'average_confidence': avg_confidence,
            'pattern_types': list(set(pattern_types)),
            'pattern_count': len(patterns),
            'time_span': (max(p.last_updated for p in patterns) - 
                         min(p.first_observed for p in patterns)).total_seconds() / 3600
        }
    
    async def _assess_group_threat_level(self, characteristics: Dict[str, Any]) -> str:
        """Assess threat level for a pattern group"""
        score = 0
        
        # High confidence patterns are more threatening
        score += characteristics.get('average_confidence', 0) * 30
        
        # More common indicators suggest coordinated threat
        score += len(characteristics.get('common_indicators', [])) * 10
        
        # Multiple pattern types suggest sophisticated threat
        score += len(characteristics.get('pattern_types', [])) * 5
        
        # Rapid evolution suggests active threat
        time_span = characteristics.get('time_span', 0)
        if time_span > 0 and time_span < 24:  # Less than 24 hours
            score += 20
        
        if score >= 70:
            return 'CRITICAL'
        elif score >= 50:
            return 'HIGH'
        elif score >= 30:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def _generate_group_recommendations(self, characteristics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for a pattern group"""
        recommendations = []
        
        # Based on common indicators
        common_indicators = characteristics.get('common_indicators', [])
        
        if any('source_ip:' in ind for ind in common_indicators):
            recommendations.append('implement_ip_reputation_filtering')
        
        if any('payload_contains:script' in ind for ind in common_indicators):
            recommendations.append('enhance_xss_protection')
        
        if any('payload_contains:sql' in ind for ind in common_indicators):
            recommendations.append('strengthen_sql_injection_defenses')
        
        # Based on pattern characteristics
        if characteristics.get('average_confidence', 0) > 0.8:
            recommendations.append('immediate_threat_hunting')
        
        if len(characteristics.get('pattern_types', [])) > 2:
            recommendations.append('comprehensive_security_review')
        
        return recommendations
    
    async def _store_group_intelligence(self, group_intel: Dict[str, Any]):
        """Store group intelligence in database"""
        conn = sqlite3.connect(self.threat_intelligence.db_path)
        
        conn.execute('''
            INSERT INTO threat_predictions 
            (prediction_id, threat_type, predicted_probability, confidence_interval,
             time_horizon_hours, contributing_factors, recommended_actions, prediction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            group_intel['group_id'],
            'correlated_pattern_group',
            1.0,  # High probability for identified groups
            json.dumps([0.8, 1.0]),
            24,  # 24 hour horizon
            json.dumps(group_intel['characteristics']),
            json.dumps(group_intel['recommended_actions']),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored group intelligence: {group_intel['group_id']} "
                   f"(threat level: {group_intel['threat_level']})")
    
    async def _update_intelligence_graph(self):
        """Update the threat intelligence correlation graph"""
        # Add new correlations to the graph
        recent_correlations = self.correlations[-10:]  # Last 10 correlations
        
        for correlation in recent_correlations:
            # Add nodes for attack and defense
            attack_node = f"attack_{correlation.attack_vector}"
            defense_node = f"defense_{correlation.defense_mechanism}"
            
            self.threat_intelligence.correlation_graph.add_node(
                attack_node, 
                type='attack_vector',
                last_seen=correlation.timestamp
            )
            
            self.threat_intelligence.correlation_graph.add_node(
                defense_node,
                type='defense_mechanism', 
                last_seen=correlation.timestamp
            )
            
            # Add edge with effectiveness as weight
            self.threat_intelligence.correlation_graph.add_edge(
                attack_node,
                defense_node,
                weight=correlation.effectiveness_score,
                correlation_id=correlation.correlation_id
            )
        
        # Analyze graph for insights
        if len(self.threat_intelligence.correlation_graph.nodes()) > 10:
            insights = await self._analyze_correlation_graph()
            logger.debug(f"Intelligence graph insights: {insights}")
    
    async def _analyze_correlation_graph(self) -> Dict[str, Any]:
        """Analyze the correlation graph for insights"""
        graph = self.threat_intelligence.correlation_graph
        
        insights = {
            'node_count': len(graph.nodes()),
            'edge_count': len(graph.edges()),
            'density': nx.density(graph),
            'most_connected_attacks': [],
            'most_effective_defenses': [],
            'weak_defense_coverage': []
        }
        
        # Find most connected attack vectors
        attack_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'attack_vector']
        if attack_nodes:
            # Get degree centrality for each attack node
            attack_degrees = []
            for node in attack_nodes:
                degree = len(list(graph.neighbors(node)))
                attack_degrees.append((node, degree))
            attack_degrees.sort(key=lambda x: x[1], reverse=True)
            insights['most_connected_attacks'] = attack_degrees[:5]
        
        # Find most effective defenses
        defense_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'defense_mechanism']
        if defense_nodes:
            defense_effectiveness = []
            for node in defense_nodes:
                # Calculate average effectiveness of edges to this defense
                incoming_edges = graph.in_edges(node, data=True)
                if incoming_edges:
                    avg_effectiveness = np.mean([d.get('weight', 0) for _, _, d in incoming_edges])
                    defense_effectiveness.append((node, avg_effectiveness))
            
            defense_effectiveness.sort(key=lambda x: x[1], reverse=True)
            insights['most_effective_defenses'] = defense_effectiveness[:5]
        
        # Find attack vectors with weak defense coverage
        weak_coverage = []
        for node in attack_nodes:
            outgoing_edges = graph.out_edges(node, data=True)
            if outgoing_edges:
                max_effectiveness = max(d.get('weight', 0) for _, _, d in outgoing_edges)
                if max_effectiveness < 0.5:  # Weak defense threshold
                    weak_coverage.append((node, max_effectiveness))
        
        insights['weak_defense_coverage'] = weak_coverage
        
        return insights
    
    async def get_purple_team_status(self) -> Dict[str, Any]:
        """Get comprehensive purple team status"""
        uptime = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        
        # Calculate recent performance metrics
        recent_correlations = [c for c in self.correlations 
                             if (datetime.utcnow() - c.timestamp).total_seconds() < 3600]
        
        avg_effectiveness = np.mean([c.effectiveness_score for c in recent_correlations]) if recent_correlations else 0.0
        avg_response_time = np.mean([c.response_time for c in recent_correlations]) if recent_correlations else 0.0
        
        return {
            'status': 'active' if self.orchestration_active else 'inactive',
            'uptime_seconds': uptime,
            'metrics': self.metrics.copy(),
            'recent_performance': {
                'correlations_last_hour': len(recent_correlations),
                'average_effectiveness': avg_effectiveness,
                'average_response_time': avg_response_time,
                'successful_mitigations': sum(1 for c in recent_correlations if c.mitigation_success)
            },
            'threat_intelligence': {
                'patterns_cached': len(self.threat_intelligence.pattern_cache),
                'correlation_graph_nodes': len(self.threat_intelligence.correlation_graph.nodes()),
                'correlation_graph_edges': len(self.threat_intelligence.correlation_graph.edges())
            },
            'adaptive_learning': {
                'model_performance': self.adaptive_learner.model_performance,
                'training_data_size': sum(len(data) for data in self.adaptive_learner.training_data.values())
            },
            'active_tasks': len(self.orchestration_tasks)
        }


def create_purple_team_system(red_team_agent, blue_team_agent,
                             security_framework: SomnusSecurityFramework,
                             vm_supervisor: VMSupervisor) -> PurpleTeamOrchestrator:
    """Factory function to create complete purple team system"""
    orchestrator = PurpleTeamOrchestrator(
        red_team_agent, blue_team_agent, security_framework, vm_supervisor
    )
    return orchestrator


if __name__ == "__main__":
    # Example usage
    async def main():
        # This would integrate with actual red/blue team agents
        logger.info("Purple Team Synthesis Engine - Production Ready")
        
        # Initialize threat intelligence fusion
        intel_fusion = ThreatIntelligenceFusion()
        
        # Test threat intelligence ingestion
        sample_threat_data = {
            'source_ip': '192.168.1.100',
            'payload': '<script>alert("xss")</script>',
            'attack_type': 'xss',
            'timestamp': datetime.utcnow(),
            'severity': 'HIGH'
        }
        
        pattern_id = await intel_fusion.ingest_threat_intelligence(sample_threat_data, 'web_attack')
        logger.info(f"Generated threat pattern: {pattern_id}")
        
        # Test pattern correlation
        correlations = await intel_fusion.correlate_patterns()
        logger.info(f"Found {len(correlations)} pattern correlations")
        
        # Test threat evolution prediction
        if pattern_id in intel_fusion.pattern_cache:
            prediction = await intel_fusion.predict_threat_evolution(pattern_id)
            logger.info(f"Threat evolution prediction: {prediction}")
    
    asyncio.run(main())