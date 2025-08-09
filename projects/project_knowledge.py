"""
MORPHEUS CHAT - Project Knowledge Base Management
Dynamic, autonomous knowledge base that grows with the project

Revolutionary Features:
- AI automatically curates and organizes knowledge
- Semantic search across all project content
- Dynamic knowledge graph construction
- Intelligent content synthesis
- Zero manual knowledge base maintenance
"""

import asyncio
import json
import logging
import pickle
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """Individual knowledge item in the project knowledge base"""
    item_id: UUID = field(default_factory=uuid4)
    title: str = ""
    content: str = ""
    source_file: Optional[str] = None
    item_type: str = "general"  # fact, concept, procedure, insight, pattern
    importance: float = 0.5
    confidence: float = 0.8
    tags: Set[str] = field(default_factory=set)
    relationships: Set[UUID] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    embedding: Optional[np.ndarray] = None


@dataclass
class KnowledgeGraph:
    """Dynamic knowledge graph structure"""
    nodes: Dict[UUID, KnowledgeItem] = field(default_factory=dict)
    edges: Dict[UUID, Set[UUID]] = field(default_factory=lambda: defaultdict(set))
    clusters: Dict[str, Set[UUID]] = field(default_factory=lambda: defaultdict(set))
    concepts: Counter = field(default_factory=Counter)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProjectKnowledgeBase:
    """
    Autonomous knowledge base for project content
    
    Automatically:
    - Extracts knowledge from uploaded files
    - Organizes knowledge by topics and relationships
    - Provides semantic search and retrieval
    - Synthesizes information across sources
    - Maintains knowledge graph structure
    """
    
    def __init__(self, project_id: str, vm_instance, knowledge_dir: Path):
        self.project_id = project_id
        self.vm = vm_instance
        self.knowledge_dir = knowledge_dir
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Knowledge storage
        self.knowledge_graph = KnowledgeGraph()
        self.vector_db = None
        self.metadata_db_path = knowledge_dir / "metadata.db"
        
        # AI components
        self.embedding_model = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Knowledge extraction patterns
        self.extraction_patterns = {
            'fact': [
                r'(.+) is (.+)',
                r'(.+) has (.+)',
                r'(.+) contains (.+)',
                r'(.+) consists of (.+)'
            ],
            'procedure': [
                r'to (.+), (.+)',
                r'step \d+[:\.]? (.+)',
                r'first (.+), then (.+)',
                r'in order to (.+), (.+)'
            ],
            'concept': [
                r'(.+) refers to (.+)',
                r'(.+) means (.+)',
                r'(.+) is defined as (.+)',
                r'the concept of (.+) (.+)'
            ]
        }
        
        logger.info(f"Knowledge base initialized for project {project_id}")
    
    async def initialize(self):
        """Initialize knowledge base components"""
        
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize vector database
            await self._initialize_vector_db()
            
            # Initialize metadata database
            await self._initialize_metadata_db()
            
            # Load existing knowledge
            await self._load_existing_knowledge()
            
            logger.info(f"Knowledge base ready for project {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise
    
    async def _initialize_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        
        vector_db_path = self.knowledge_dir / "vectors"
        vector_db_path.mkdir(exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(vector_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.vector_db = client.get_or_create_collection(
            name=f"project_{self.project_id}",
            metadata={"hnsw:space": "cosine"}
        )
    
    async def _initialize_metadata_db(self):
        """Initialize SQLite database for metadata"""
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                item_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                source_file TEXT,
                item_type TEXT,
                importance REAL,
                confidence REAL,
                tags TEXT,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                strength REAL,
                created_at TEXT,
                PRIMARY KEY (source_id, target_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_clusters (
                cluster_name TEXT,
                item_id TEXT,
                cluster_strength REAL,
                created_at TEXT,
                PRIMARY KEY (cluster_name, item_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def extract_knowledge_from_content(
        self, 
        content: str, 
        source_file: str,
        file_analysis: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """
        Extract structured knowledge from file content
        
        Uses AI to identify facts, concepts, procedures, and insights
        """
        
        knowledge_items = []
        
        # Extract different types of knowledge
        facts = await self._extract_facts(content, source_file)
        concepts = await self._extract_concepts(content, source_file)
        procedures = await self._extract_procedures(content, source_file)
        insights = await self._extract_insights(content, source_file, file_analysis)
        
        knowledge_items.extend(facts)
        knowledge_items.extend(concepts)
        knowledge_items.extend(procedures)
        knowledge_items.extend(insights)
        
        # Generate embeddings for all items
        for item in knowledge_items:
            item.embedding = await self._generate_embedding(item.content)
        
        # Find relationships between items
        await self._identify_relationships(knowledge_items)
        
        # Store in knowledge base
        for item in knowledge_items:
            await self._store_knowledge_item(item)
        
        logger.info(f"Extracted {len(knowledge_items)} knowledge items from {source_file}")
        return knowledge_items
    
    async def _extract_facts(self, content: str, source_file: str) -> List[KnowledgeItem]:
        """Extract factual information"""
        
        facts = []
        
        # Use regex patterns to identify facts
        import re
        for pattern in self.extraction_patterns['fact']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                fact_text = match.group(0)
                
                # Skip very short or very long matches
                if 10 < len(fact_text) < 200:
                    fact = KnowledgeItem(
                        title=f"Fact from {Path(source_file).stem}",
                        content=fact_text,
                        source_file=source_file,
                        item_type="fact",
                        importance=0.6,
                        tags={'fact', 'extracted'}
                    )
                    facts.append(fact)
        
        return facts[:10]  # Limit to top 10 facts per file
    
    async def _extract_concepts(self, content: str, source_file: str) -> List[KnowledgeItem]:
        """Extract conceptual information"""
        
        concepts = []
        
        # Look for definition patterns
        import re
        for pattern in self.extraction_patterns['concept']:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                concept_text = match.group(0)
                
                if 10 < len(concept_text) < 300:
                    concept = KnowledgeItem(
                        title=f"Concept from {Path(source_file).stem}",
                        content=concept_text,
                        source_file=source_file,
                        item_type="concept",
                        importance=0.7,
                        tags={'concept', 'definition'}
                    )
                    concepts.append(concept)
        
        return concepts[:5]  # Limit to top 5 concepts per file
    
    async def _extract_procedures(self, content: str, source_file: str) -> List[KnowledgeItem]:
        """Extract procedural information"""
        
        procedures = []
        
        # Look for step-by-step instructions
        import re
        
        # Find numbered lists or step patterns
        step_patterns = [
            r'(?:step|stage) \d+[:\.]? (.+)',
            r'\d+\. (.+)',
            r'first (.+), then (.+)',
            r'to (.+), (.+) and (.+)'
        ]
        
        for pattern in step_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                procedure_text = match.group(0)
                
                if 15 < len(procedure_text) < 500:
                    procedure = KnowledgeItem(
                        title=f"Procedure from {Path(source_file).stem}",
                        content=procedure_text,
                        source_file=source_file,
                        item_type="procedure",
                        importance=0.8,
                        tags={'procedure', 'steps'}
                    )
                    procedures.append(procedure)
        
        return procedures[:3]  # Limit to top 3 procedures per file
    
    async def _extract_insights(
        self, 
        content: str, 
        source_file: str,
        file_analysis: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Extract high-level insights and patterns"""
        
        insights = []
        
        # Use file analysis to create insights
        if file_analysis.get('key_concepts'):
            concept_insight = KnowledgeItem(
                title=f"Key concepts in {Path(source_file).stem}",
                content=f"This file focuses on: {', '.join(file_analysis['key_concepts'])}",
                source_file=source_file,
                item_type="insight",
                importance=0.9,
                tags={'insight', 'concepts', 'analysis'}
            )
            insights.append(concept_insight)
        
        if file_analysis.get('content_summary'):
            summary_insight = KnowledgeItem(
                title=f"Summary of {Path(source_file).stem}",
                content=file_analysis['content_summary'],
                source_file=source_file,
                item_type="insight",
                importance=0.7,
                tags={'insight', 'summary'}
            )
            insights.append(summary_insight)
        
        return insights
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        # Generate embedding using sentence transformer
        embedding = self.embedding_model.encode(text)
        
        # Cache embedding
        self.embeddings_cache[text] = embedding
        
        return embedding
    
    async def _identify_relationships(self, knowledge_items: List[KnowledgeItem]):
        """Identify relationships between knowledge items"""
        
        for i, item1 in enumerate(knowledge_items):
            for j, item2 in enumerate(knowledge_items[i+1:], i+1):
                
                # Calculate semantic similarity
                if item1.embedding is not None and item2.embedding is not None:
                    similarity = np.dot(item1.embedding, item2.embedding) / (
                        np.linalg.norm(item1.embedding) * np.linalg.norm(item2.embedding)
                    )
                    
                    # Create relationship if similarity is high
                    if similarity > 0.7:
                        item1.relationships.add(item2.item_id)
                        item2.relationships.add(item1.item_id)
    
    async def _store_knowledge_item(self, item: KnowledgeItem):
        """Store knowledge item in databases"""
        
        # Store in metadata database
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_items 
            (item_id, title, content, source_file, item_type, importance, confidence, 
             tags, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(item.item_id), item.title, item.content, item.source_file,
            item.item_type, item.importance, item.confidence,
            json.dumps(list(item.tags)), item.created_at.isoformat(),
            item.last_accessed.isoformat(), item.access_count
        ))
        
        conn.commit()
        conn.close()
        
        # Store in vector database
        if item.embedding is not None:
            self.vector_db.add(
                ids=[str(item.item_id)],
                embeddings=[item.embedding.tolist()],
                documents=[item.content],
                metadatas=[{
                    'title': item.title,
                    'source_file': item.source_file or '',
                    'item_type': item.item_type,
                    'importance': item.importance,
                    'created_at': item.created_at.isoformat()
                }]
            )
        
        # Add to knowledge graph
        self.knowledge_graph.nodes[item.item_id] = item
        
        # Add relationships to graph
        for related_id in item.relationships:
            self.knowledge_graph.edges[item.item_id].add(related_id)
            self.knowledge_graph.edges[related_id].add(item.item_id)
    
    async def search_knowledge(
        self, 
        query: str, 
        limit: int = 10,
        item_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base using semantic similarity
        """
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            
            # Search vector database
            results = self.vector_db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            if results['ids']:
                for i, item_id in enumerate(results['ids'][0]):
                    result = {
                        'item_id': item_id,
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'source_file': results['metadatas'][0][i].get('source_file', ''),
                        'item_type': results['metadatas'][0][i].get('item_type', ''),
                        'importance': results['metadatas'][0][i].get('importance', 0.5)
                    }
                    
                    # Filter by item types if specified
                    if item_types is None or result['item_type'] in item_types:
                        formatted_results.append(result)
            
            # Sort by relevance (similarity * importance)
            formatted_results.sort(
                key=lambda x: x['similarity'] * x['importance'],
                reverse=True
            )
            
            # Update access counts
            await self._update_access_counts([r['item_id'] for r in formatted_results])
            
            return formatted_results[:limit]
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def _update_access_counts(self, item_ids: List[str]):
        """Update access counts for knowledge items"""
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        for item_id in item_ids:
            cursor.execute('''
                UPDATE knowledge_items 
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE item_id = ?
            ''', (datetime.now(timezone.utc).isoformat(), item_id))
        
        conn.commit()
        conn.close()
    
    async def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get comprehensive knowledge base summary"""
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        # Get counts by type
        cursor.execute('''
            SELECT item_type, COUNT(*) 
            FROM knowledge_items 
            GROUP BY item_type
        ''')
        type_counts = dict(cursor.fetchall())
        
        # Get most accessed items
        cursor.execute('''
            SELECT title, access_count, item_type
            FROM knowledge_items 
            ORDER BY access_count DESC 
            LIMIT 5
        ''')
        popular_items = cursor.fetchall()
        
        # Get recent items
        cursor.execute('''
            SELECT title, created_at, item_type
            FROM knowledge_items 
            ORDER BY created_at DESC 
            LIMIT 5
        ''')
        recent_items = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_items': sum(type_counts.values()),
            'items_by_type': type_counts,
            'popular_items': [
                {'title': item[0], 'access_count': item[1], 'type': item[2]}
                for item in popular_items
            ],
            'recent_items': [
                {'title': item[0], 'created_at': item[1], 'type': item[2]}
                for item in recent_items
            ],
            'knowledge_graph_size': len(self.knowledge_graph.nodes),
            'relationships_count': sum(len(edges) for edges in self.knowledge_graph.edges.values()),
            'last_updated': self.knowledge_graph.last_updated.isoformat()
        }
    
    async def synthesize_knowledge(self, topic: str) -> Dict[str, Any]:
        """
        Synthesize knowledge about a specific topic
        
        Combines information from multiple sources to create comprehensive overview
        """
        
        # Search for relevant knowledge
        relevant_items = await self.search_knowledge(topic, limit=20)
        
        if not relevant_items:
            return {'topic': topic, 'synthesis': 'No relevant knowledge found.'}
        
        # Group by type
        facts = [item for item in relevant_items if item['item_type'] == 'fact']
        concepts = [item for item in relevant_items if item['item_type'] == 'concept']
        procedures = [item for item in relevant_items if item['item_type'] == 'procedure']
        insights = [item for item in relevant_items if item['item_type'] == 'insight']
        
        # Create synthesis
        synthesis = {
            'topic': topic,
            'total_sources': len(set(item['source_file'] for item in relevant_items)),
            'confidence': sum(item['similarity'] for item in relevant_items) / len(relevant_items),
            'sections': {}
        }
        
        if concepts:
            synthesis['sections']['concepts'] = {
                'items': [item['content'] for item in concepts[:3]],
                'sources': list(set(item['source_file'] for item in concepts))
            }
        
        if facts:
            synthesis['sections']['facts'] = {
                'items': [item['content'] for item in facts[:5]],
                'sources': list(set(item['source_file'] for item in facts))
            }
        
        if procedures:
            synthesis['sections']['procedures'] = {
                'items': [item['content'] for item in procedures[:3]],
                'sources': list(set(item['source_file'] for item in procedures))
            }
        
        if insights:
            synthesis['sections']['insights'] = {
                'items': [item['content'] for item in insights[:3]],
                'sources': list(set(item['source_file'] for item in insights))
            }
        
        return synthesis
    
    async def _load_existing_knowledge(self):
        """Load existing knowledge from databases"""
        
        if not self.metadata_db_path.exists():
            return
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT COUNT(*) FROM knowledge_items')
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info(f"Loading {count} existing knowledge items")
                
                # Load items (simplified - in production, load in batches)
                cursor.execute('''
                    SELECT item_id, title, content, source_file, item_type, 
                           importance, confidence, tags, created_at, last_accessed, access_count
                    FROM knowledge_items
                    LIMIT 1000
                ''')
                
                for row in cursor.fetchall():
                    item = KnowledgeItem(
                        item_id=UUID(row[0]),
                        title=row[1],
                        content=row[2],
                        source_file=row[3],
                        item_type=row[4],
                        importance=row[5],
                        confidence=row[6],
                        tags=set(json.loads(row[7])),
                        created_at=datetime.fromisoformat(row[8]),
                        last_accessed=datetime.fromisoformat(row[9]),
                        access_count=row[10]
                    )
                    
                    self.knowledge_graph.nodes[item.item_id] = item
            
        except sqlite3.Error as e:
            logger.error(f"Error loading existing knowledge: {e}")
        
        finally:
            conn.close()
    
    async def cleanup(self):
        """Cleanup knowledge base resources"""
        
        # Close database connections
        if hasattr(self, 'vector_db'):
            # ChromaDB cleanup if needed
            pass
        
        # Clear caches
        self.embeddings_cache.clear()
        self.knowledge_graph = KnowledgeGraph()