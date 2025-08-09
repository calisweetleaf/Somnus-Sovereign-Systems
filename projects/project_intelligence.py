"""
MORPHEUS CHAT - Project Intelligence System
Autonomous knowledge management and file organization

Revolutionary Features:
- AI automatically analyzes and organizes all uploaded files
- Builds dynamic knowledge base without user intervention
- Learns project patterns and suggests improvements
- Synthesizes insights across all project content
- Zero user micromanagement required
"""

import asyncio
import json
import logging
import mimetypes
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    file_path: Path
    file_type: str
    mime_type: str
    size_bytes: int
    content_summary: str
    key_concepts: List[str]
    importance_score: float
    relationships: List[str]
    suggested_category: str
    processing_time: float
    last_analyzed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProjectInsight:
    """AI-generated project insight"""
    insight_type: str  # pattern, suggestion, optimization, concern
    title: str
    description: str
    confidence: float
    actionable_items: List[str]
    related_files: List[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProjectIntelligenceEngine:
    """
    Autonomous AI system for project knowledge management
    
    Runs inside each project VM and automatically:
    - Analyzes uploaded files
    - Organizes content by relevance
    - Builds searchable knowledge base
    - Generates project insights
    - Learns user patterns
    """
    
    def __init__(self, project_id: str, vm_instance):
        self.project_id = project_id
        self.vm = vm_instance
        
        # Project paths in VM
        self.project_root = Path("/project")
        self.files_dir = self.project_root / "files"
        self.knowledge_dir = self.project_root / "knowledge"
        self.intelligence_dir = self.project_root / "intelligence"
        
        # Intelligence state
        self.analyzed_files: Dict[str, FileAnalysis] = {}
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)
        self.project_concepts: Counter = Counter()
        self.project_insights: List[ProjectInsight] = []
        self.file_categories: Dict[str, List[str]] = defaultdict(list)
        
        # AI models for analysis
        self.embedding_model = None
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Processing state
        self.is_processing = False
        self.last_intelligence_run = datetime.now(timezone.utc)
        
        logger.info(f"Intelligence engine initialized for project {project_id}")
    
    async def initialize(self):
        """Initialize intelligence systems"""
        
        try:
            # Load embedding model for semantic analysis
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create intelligence directories
            await self._setup_intelligence_workspace()
            
            # Load existing analysis state
            await self._load_intelligence_state()
            
            logger.info(f"Intelligence engine ready for project {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize intelligence engine: {e}")
            raise
    
    async def _setup_intelligence_workspace(self):
        """Setup intelligence workspace in VM"""
        
        workspace_script = '''
# Create intelligence workspace
mkdir -p /project/intelligence/{analysis,knowledge_base,insights,patterns,automation}
mkdir -p /project/knowledge/{categories,concepts,relationships,summaries}

# Install additional analysis tools
pip install --quiet networkx textstat keybert yake
pip install --quiet spacy && python -m spacy download en_core_web_sm

echo "Intelligence workspace ready"
        '''
        
        await self.vm.execute_command(workspace_script)
    
    async def monitor_and_process_files(self):
        """Main intelligence loop - monitors files and processes automatically"""
        
        while True:
            try:
                if not self.is_processing:
                    # Check for new or modified files
                    files_to_process = await self._scan_for_new_files()
                    
                    if files_to_process:
                        await self._process_files_batch(files_to_process)
                    
                    # Generate insights periodically
                    time_since_insights = (datetime.now(timezone.utc) - self.last_intelligence_run).seconds
                    if time_since_insights > 300:  # Every 5 minutes
                        await self._generate_project_insights()
                        self.last_intelligence_run = datetime.now(timezone.utc)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Intelligence monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _scan_for_new_files(self) -> List[Path]:
        """Scan for new or modified files"""
        
        files_to_process = []
        
        try:
            # Get all files in project
            result = await self.vm.execute_command(
                "find /project/files -type f -printf '%p %T@ %s\\n'"
            )
            
            for line in result.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.rsplit(' ', 2)
                if len(parts) != 3:
                    continue
                
                file_path = Path(parts[0])
                timestamp = float(parts[1])
                size = int(parts[2])
                
                # Check if file needs processing
                file_hash = await self._get_file_hash(file_path)
                
                if (file_hash not in self.analyzed_files or
                    self.analyzed_files[file_hash].size_bytes != size):
                    files_to_process.append(file_path)
        
        except Exception as e:
            logger.error(f"Error scanning files: {e}")
        
        return files_to_process
    
    async def _get_file_hash(self, file_path: Path) -> str:
        """Get file hash for change detection"""
        try:
            result = await self.vm.execute_command(f"sha256sum '{file_path}'")
            return result.split()[0]
        except:
            return str(file_path)
    
    async def _process_files_batch(self, files: List[Path]):
        """Process a batch of files"""
        
        self.is_processing = True
        try:
            for file_path in files:
                await self._analyze_single_file(file_path)
            
            # Update knowledge base after processing
            await self._update_knowledge_base()
            
            # Organize files by categories
            await self._auto_organize_files()
            
            # Save state
            await self._save_intelligence_state()
            
        finally:
            self.is_processing = False
    
    async def _analyze_single_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single file comprehensively"""
        
        start_time = datetime.now()
        
        try:
            # Get file info
            file_stats = await self._get_file_stats(file_path)
            mime_type = mimetypes.guess_type(str(file_path))[0] or 'unknown'
            
            # Extract content based on file type
            content = await self._extract_file_content(file_path, mime_type)
            
            # Analyze content
            analysis = await self._analyze_content(content, file_path)
            
            # Create file analysis
            file_analysis = FileAnalysis(
                file_path=file_path,
                file_type=self._categorize_file_type(file_path, mime_type),
                mime_type=mime_type,
                size_bytes=file_stats.get('size', 0),
                content_summary=analysis['summary'],
                key_concepts=analysis['concepts'],
                importance_score=analysis['importance'],
                relationships=analysis['relationships'],
                suggested_category=analysis['category'],
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            # Store analysis
            file_hash = await self._get_file_hash(file_path)
            self.analyzed_files[file_hash] = file_analysis
            
            # Update project knowledge
            await self._integrate_file_knowledge(file_analysis)
            
            logger.info(f"Analyzed file: {file_path.name}")
            return file_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            raise
    
    async def _get_file_stats(self, file_path: Path) -> Dict[str, Any]:
        """Get file statistics"""
        try:
            result = await self.vm.execute_command(f"stat -c '%s %Y' '{file_path}'")
            size, modified = result.strip().split()
            return {
                'size': int(size),
                'modified': int(modified)
            }
        except:
            return {'size': 0, 'modified': 0}
    
    async def _extract_file_content(self, file_path: Path, mime_type: str) -> str:
        """Extract text content from file"""
        
        try:
            if mime_type.startswith('text/'):
                # Text files
                result = await self.vm.execute_command(f"head -c 50000 '{file_path}'")
                return result
            
            elif file_path.suffix.lower() in ['.py', '.js', '.html', '.css', '.md', '.yml', '.yaml', '.json']:
                # Code and markup files
                result = await self.vm.execute_command(f"head -c 50000 '{file_path}'")
                return result
            
            elif file_path.suffix.lower() == '.pdf':
                # PDF files (if pdftotext is available)
                result = await self.vm.execute_command(f"pdftotext '{file_path}' - | head -c 50000")
                return result
            
            else:
                # Binary files - just get filename and size info
                return f"Binary file: {file_path.name} ({mime_type})"
        
        except Exception as e:
            logger.warning(f"Could not extract content from {file_path}: {e}")
            return f"File: {file_path.name}"
    
    async def _analyze_content(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Analyze file content using AI"""
        
        # Extract key concepts using simple keyword extraction
        concepts = await self._extract_concepts(content)
        
        # Generate content summary
        summary = await self._generate_summary(content, file_path)
        
        # Calculate importance score
        importance = await self._calculate_importance(content, concepts, file_path)
        
        # Find relationships to other files
        relationships = await self._find_relationships(concepts, file_path)
        
        # Suggest category
        category = await self._suggest_category(content, concepts, file_path)
        
        return {
            'summary': summary,
            'concepts': concepts,
            'importance': importance,
            'relationships': relationships,
            'category': category
        }
    
    async def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        word_freq = Counter(words)
        
        # Filter out common words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        concepts = []
        for word, freq in word_freq.most_common(20):
            if word not in stop_words and len(word) > 3:
                concepts.append(word)
        
        return concepts[:10]  # Top 10 concepts
    
    async def _generate_summary(self, content: str, file_path: Path) -> str:
        """Generate content summary"""
        
        # Simple extractive summary - first few sentences
        sentences = re.split(r'[.!?]+', content)
        summary_sentences = []
        
        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                summary_sentences.append(sentence)
        
        if summary_sentences:
            return '. '.join(summary_sentences)[:200] + '...'
        else:
            return f"File: {file_path.name} ({len(content)} characters)"
    
    async def _calculate_importance(self, content: str, concepts: List[str], file_path: Path) -> float:
        """Calculate file importance score"""
        
        score = 0.0
        
        # Size factor
        score += min(len(content) / 10000, 1.0) * 0.3
        
        # Concept richness
        score += min(len(concepts) / 10, 1.0) * 0.4
        
        # File type importance
        important_extensions = {'.py': 0.8, '.md': 0.7, '.json': 0.6, '.txt': 0.5}
        score += important_extensions.get(file_path.suffix.lower(), 0.3) * 0.3
        
        return min(score, 1.0)
    
    async def _find_relationships(self, concepts: List[str], file_path: Path) -> List[str]:
        """Find relationships to other analyzed files"""
        
        relationships = []
        
        for other_hash, other_analysis in self.analyzed_files.items():
            if other_analysis.file_path == file_path:
                continue
            
            # Find concept overlap
            overlap = set(concepts) & set(other_analysis.key_concepts)
            if len(overlap) >= 2:
                relationships.append(other_analysis.file_path.name)
        
        return relationships[:5]  # Top 5 relationships
    
    async def _suggest_category(self, content: str, concepts: List[str], file_path: Path) -> str:
        """Suggest file category based on analysis"""
        
        # File extension categories
        code_extensions = {'.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h'}
        doc_extensions = {'.md', '.txt', '.pdf', '.docx', '.doc'}
        data_extensions = {'.json', '.csv', '.xml', '.yaml', '.yml'}
        
        if file_path.suffix.lower() in code_extensions:
            return 'code'
        elif file_path.suffix.lower() in doc_extensions:
            return 'documentation'
        elif file_path.suffix.lower() in data_extensions:
            return 'data'
        
        # Content-based categorization
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['function', 'class', 'import', 'def', 'var', 'const']):
            return 'code'
        elif any(word in content_lower for word in ['research', 'analysis', 'study', 'findings']):
            return 'research'
        elif any(word in content_lower for word in ['meeting', 'agenda', 'minutes', 'action']):
            return 'meetings'
        elif any(word in content_lower for word in ['plan', 'strategy', 'goal', 'objective']):
            return 'planning'
        else:
            return 'general'
    
    def _categorize_file_type(self, file_path: Path, mime_type: str) -> str:
        """Categorize file type"""
        
        if mime_type.startswith('text/'):
            return 'text'
        elif mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
        elif file_path.suffix.lower() in ['.pdf', '.doc', '.docx']:
            return 'document'
        elif file_path.suffix.lower() in ['.py', '.js', '.html', '.css']:
            return 'code'
        elif file_path.suffix.lower() in ['.json', '.csv', '.xml', '.yml']:
            return 'data'
        else:
            return 'unknown'
    
    async def _integrate_file_knowledge(self, analysis: FileAnalysis):
        """Integrate file analysis into project knowledge"""
        
        # Update project concepts
        for concept in analysis.key_concepts:
            self.project_concepts[concept] += 1
        
        # Update knowledge graph
        for related_file in analysis.relationships:
            self.knowledge_graph[analysis.file_path.name].add(related_file)
            self.knowledge_graph[related_file].add(analysis.file_path.name)
        
        # Update file categories
        self.file_categories[analysis.suggested_category].append(analysis.file_path.name)
    
    async def _update_knowledge_base(self):
        """Update project knowledge base"""
        
        knowledge_summary = {
            'total_files': len(self.analyzed_files),
            'top_concepts': dict(self.project_concepts.most_common(20)),
            'file_categories': dict(self.file_categories),
            'relationship_count': sum(len(relations) for relations in self.knowledge_graph.values()),
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        await self.vm.write_file(
            "/project/knowledge/knowledge_summary.json",
            json.dumps(knowledge_summary, indent=2)
        )
    
    async def _auto_organize_files(self):
        """Automatically organize files by categories"""
        
        # Create category directories
        for category in self.file_categories:
            await self.vm.execute_command(f"mkdir -p /project/files/organized/{category}")
        
        # Create symbolic links to organize files
        for category, files in self.file_categories.items():
            for file_name in files:
                source_path = f"/project/files/{file_name}"
                link_path = f"/project/files/organized/{category}/{file_name}"
                
                await self.vm.execute_command(
                    f"ln -sf '{source_path}' '{link_path}'"
                )
    
    async def _generate_project_insights(self):
        """Generate AI insights about the project"""
        
        insights = []
        
        # Pattern insights
        if len(self.project_concepts) > 10:
            top_concepts = self.project_concepts.most_common(5)
            insights.append(ProjectInsight(
                insight_type="pattern",
                title="Project Focus Areas",
                description=f"Main focus areas: {', '.join([c[0] for c in top_concepts])}",
                confidence=0.8,
                actionable_items=["Consider creating focused documentation for these areas"],
                related_files=list(self.file_categories.get('documentation', []))[:3]
            ))
        
        # Organization suggestions
        if len(self.analyzed_files) > 20:
            insights.append(ProjectInsight(
                insight_type="suggestion",
                title="File Organization",
                description="Large number of files detected. Consider using organized folder structure.",
                confidence=0.9,
                actionable_items=[
                    "Use /project/files/organized/ structure",
                    "Create category-specific README files"
                ],
                related_files=[]
            ))
        
        # Store insights
        self.project_insights.extend(insights)
        
        # Keep only recent insights
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)
        self.project_insights = [
            insight for insight in self.project_insights
            if insight.generated_at > cutoff_date
        ]
    
    async def get_project_summary(self) -> Dict[str, Any]:
        """Get comprehensive project summary"""
        
        return {
            'project_id': self.project_id,
            'total_files': len(self.analyzed_files),
            'categories': dict(self.file_categories),
            'top_concepts': dict(self.project_concepts.most_common(10)),
            'insights_count': len(self.project_insights),
            'last_analysis': self.last_intelligence_run.isoformat(),
            'knowledge_graph_size': len(self.knowledge_graph)
        }
    
    async def _save_intelligence_state(self):
        """Save intelligence state to VM"""
        
        state = {
            'analyzed_files': {k: v.__dict__ for k, v in self.analyzed_files.items()},
            'project_concepts': dict(self.project_concepts),
            'file_categories': dict(self.file_categories),
            'knowledge_graph': {k: list(v) for k, v in self.knowledge_graph.items()},
            'project_insights': [insight.__dict__ for insight in self.project_insights],
            'last_intelligence_run': self.last_intelligence_run.isoformat()
        }
        
        await self.vm.write_file(
            "/project/intelligence/state.json",
            json.dumps(state, indent=2, default=str)
        )
    
    async def _load_intelligence_state(self):
        """Load intelligence state from VM"""
        
        try:
            state_content = await self.vm.read_file("/project/intelligence/state.json")
            state = json.loads(state_content)
            
            # Restore state
            self.project_concepts = Counter(state.get('project_concepts', {}))
            self.file_categories = defaultdict(list, state.get('file_categories', {}))
            
            # Convert knowledge graph back to sets
            knowledge_graph_data = state.get('knowledge_graph', {})
            for key, values in knowledge_graph_data.items():
                self.knowledge_graph[key] = set(values)
            
            logger.info(f"Loaded intelligence state for project {self.project_id}")
            
        except Exception as e:
            logger.info(f"No existing intelligence state found: {e}")