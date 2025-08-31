"""
SOMNUS SOVEREIGN SYSTEMS - Production Research Report Exporter
Complete backend system for research report generation, memory integration, and transformation

Features:
- Complete integration with memory system for persistent context
- Research artifact creation in existing artifact system
- Transform functionality for AI-powered report conversions
- Full memory persistence of all research sessions and reports
- Production-ready with comprehensive error handling
"""

import asyncio
import json
import logging
import os
import hashlib
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import zipfile
import io
import traceback

# Core system imports
from schemas.session import SessionID, UserID
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from core.memory_integration import SessionMemoryContext, EnhancedSessionManager

# Research system imports
try:
    from .research_session import ResearchSession
    from .research_intelligence import EntityGraph, ContradictionAnalysis
    from .ai_browser_research_agent import BrowserResearchAgent
    RESEARCH_IMPORTS_AVAILABLE = True
except ImportError:
    RESEARCH_IMPORTS_AVAILABLE = False

# Artifact system imports
try:
    from artifacts_base_layer import ArtifactManager, ArtifactType, SecurityLevel
    ARTIFACT_IMPORTS_AVAILABLE = True
except ImportError:
    ARTIFACT_IMPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# CORE DATA MODELS
# ============================================================================

class ExportFormat(str, Enum):
    """Supported export formats"""
    RESEARCH_ARTIFACT = "research_artifact"
    INTERACTIVE_HTML = "interactive_html"
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"


class TransformType(str, Enum):
    """AI transformation types"""
    WEBSITE = "website"
    INFOGRAPHIC = "infographic" 
    PRESENTATION = "presentation"
    SUMMARY = "summary"
    DASHBOARD = "dashboard"


@dataclass
class ResearchReport:
    """Complete research report data structure"""
    id: str
    session_id: str
    user_id: str
    title: str
    content: str
    summary: str
    sources: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    confidence_score: float
    coherence_score: float
    depth_score: int
    entity_count: int
    source_count: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    checksum: str
    file_size: int


@dataclass
class TransformRequest:
    """Request for AI-powered report transformation"""
    report_id: str
    user_id: str
    transform_type: TransformType
    custom_instructions: Optional[str] = None
    target_audience: str = "general"
    style_preferences: Dict[str, Any] = None
    model_name: Optional[str] = None


@dataclass
class ExportResult:
    """Result of export operation"""
    success: bool
    export_id: str
    report_id: str
    artifact_id: Optional[str] = None
    memory_id: Optional[str] = None
    file_path: Optional[str] = None
    file_size: int = 0
    export_time: float = 0.0
    format: ExportFormat = ExportFormat.RESEARCH_ARTIFACT
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


# ============================================================================
# MEMORY INTEGRATION LAYER
# ============================================================================

class ResearchMemoryIntegrator:
    """Handles all memory operations for research reports"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
    async def store_research_report(
        self,
        report: ResearchReport,
        session_context: SessionMemoryContext
    ) -> UUID:
        """Store complete research report in memory system"""
        
        # Create comprehensive report content for memory storage
        memory_content = self._create_memory_content(report)
        
        # Store as high-importance research document
        memory_id = await self.memory_manager.store_memory(
            user_id=report.user_id,
            content=memory_content,
            memory_type=MemoryType.DOCUMENT,
            importance=MemoryImportance.HIGH,
            scope=MemoryScope.PRIVATE,
            source_session=report.session_id,
            tags=self._generate_tags(report),
            metadata={
                'report_id': report.id,
                'research_session_id': report.session_id,
                'report_type': 'research_report',
                'confidence_score': report.confidence_score,
                'coherence_score': report.coherence_score,
                'source_count': report.source_count,
                'entity_count': report.entity_count,
                'depth_score': report.depth_score,
                'created_at': report.created_at.isoformat(),
                'checksum': report.checksum,
                'has_contradictions': len(report.contradictions) > 0,
                'primary_topics': self._extract_primary_topics(report),
                'research_quality': self._calculate_research_quality(report)
            }
        )
        
        # Store individual entities as separate memories for better retrieval
        await self._store_entities_separately(report, session_context)
        
        # Store contradictions as learning opportunities
        await self._store_contradictions(report, session_context)
        
        # Store high-quality sources for future reference
        await self._store_quality_sources(report, session_context)
        
        logger.info(f"Research report {report.id} stored in memory with ID {memory_id}")
        return memory_id
    
    def _create_memory_content(self, report: ResearchReport) -> str:
        """Create structured content for memory storage"""
        
        content_parts = [
            f"# Research Report: {report.title}",
            f"\n## Executive Summary\n{report.summary}",
            f"\n## Full Analysis\n{report.content}",
            f"\n## Research Metrics",
            f"- Confidence Score: {report.confidence_score:.2%}",
            f"- Coherence Score: {report.coherence_score:.2%}",
            f"- Research Depth: {report.depth_score}",
            f"- Sources Analyzed: {report.source_count}",
            f"- Entities Identified: {report.entity_count}"
        ]
        
        if report.contradictions:
            content_parts.append("\n## Identified Contradictions")
            for i, contradiction in enumerate(report.contradictions, 1):
                content_parts.append(f"{i}. {contradiction.get('description', 'Unknown contradiction')}")
        
        if report.sources:
            content_parts.append("\n## Key Sources")
            for source in report.sources[:10]:  # Top 10 sources
                reliability = source.get('reliability', 0)
                content_parts.append(f"- {source.get('title', 'Unknown')} (Reliability: {reliability:.1%})")
                content_parts.append(f"  URL: {source.get('url', 'Unknown')}")
        
        return "\n".join(content_parts)
    
    def _generate_tags(self, report: ResearchReport) -> List[str]:
        """Generate intelligent tags for the research report"""
        
        tags = [
            'research_report',
            'deep_research',
            f'confidence_{int(report.confidence_score * 100)}'
        ]
        
        # Quality-based tags
        if report.confidence_score > 0.9:
            tags.append('high_confidence')
        elif report.confidence_score > 0.7:
            tags.append('medium_confidence')
        else:
            tags.append('low_confidence')
            
        # Depth tags
        if report.depth_score >= 8:
            tags.append('comprehensive')
        elif report.depth_score >= 5:
            tags.append('detailed')
        else:
            tags.append('basic')
            
        # Source quality tags
        if report.source_count >= 20:
            tags.append('well_sourced')
        elif report.source_count >= 10:
            tags.append('adequately_sourced')
            
        # Contradiction tags
        if report.contradictions:
            tags.append('has_contradictions')
            tags.append('complex_topic')
        else:
            tags.append('coherent_findings')
            
        return tags
    
    def _extract_primary_topics(self, report: ResearchReport) -> List[str]:
        """Extract primary topics from entities and content"""
        
        topics = []
        
        # Extract from entities
        for entity in report.entities[:5]:  # Top 5 entities
            if 'name' in entity:
                topics.append(entity['name'])
                
        # Extract from title
        title_words = [word.lower() for word in report.title.split() if len(word) > 3]
        topics.extend(title_words)
        
        return list(set(topics))[:10]  # Dedupe and limit
    
    def _calculate_research_quality(self, report: ResearchReport) -> str:
        """Calculate overall research quality rating"""
        
        score = 0
        
        # Confidence component (40%)
        score += report.confidence_score * 0.4
        
        # Coherence component (30%)
        score += report.coherence_score * 0.3
        
        # Depth component (20%)
        depth_normalized = min(report.depth_score / 10, 1.0)
        score += depth_normalized * 0.2
        
        # Source quality component (10%)
        source_normalized = min(report.source_count / 20, 1.0)
        score += source_normalized * 0.1
        
        if score >= 0.85:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "adequate"
        else:
            return "needs_improvement"
    
    async def _store_entities_separately(
        self,
        report: ResearchReport,
        session_context: SessionMemoryContext
    ) -> None:
        """Store important entities as separate memories"""
        
        for entity in report.entities[:20]:  # Store top 20 entities
            if entity.get('importance', 0) > 0.7:  # Only high-importance entities
                
                entity_content = f"Entity: {entity.get('name', 'Unknown')}\n"
                if 'description' in entity:
                    entity_content += f"Description: {entity['description']}\n"
                if 'context' in entity:
                    entity_content += f"Context: {entity['context']}\n"
                if 'sources' in entity:
                    entity_content += f"Sources: {', '.join(entity['sources'])}\n"
                
                await self.memory_manager.store_memory(
                    user_id=report.user_id,
                    content=entity_content,
                    memory_type=MemoryType.FACT,
                    importance=MemoryImportance.MEDIUM,
                    scope=MemoryScope.PRIVATE,
                    source_session=report.session_id,
                    tags=['entity', 'research_fact', entity.get('type', 'general')],
                    metadata={
                        'entity_name': entity.get('name'),
                        'entity_type': entity.get('type'),
                        'source_report_id': report.id,
                        'importance_score': entity.get('importance', 0),
                        'reliability': entity.get('reliability', 0)
                    }
                )
    
    async def _store_contradictions(
        self,
        report: ResearchReport,
        session_context: SessionMemoryContext
    ) -> None:
        """Store contradictions as learning opportunities"""
        
        for contradiction in report.contradictions:
            contradiction_content = f"Contradiction Detected: {contradiction.get('title', 'Unknown')}\n"
            contradiction_content += f"Description: {contradiction.get('description', '')}\n"
            contradiction_content += f"Sources involved: {', '.join(contradiction.get('sources', []))}\n"
            contradiction_content += f"Severity: {contradiction.get('severity', 'unknown')}\n"
            
            await self.memory_manager.store_memory(
                user_id=report.user_id,
                content=contradiction_content,
                memory_type=MemoryType.INSIGHT,
                importance=MemoryImportance.HIGH,
                scope=MemoryScope.PRIVATE,
                source_session=report.session_id,
                tags=['contradiction', 'research_insight', 'complexity'],
                metadata={
                    'contradiction_type': contradiction.get('type'),
                    'severity': contradiction.get('severity'),
                    'source_report_id': report.id,
                    'requires_investigation': True
                }
            )
    
    async def _store_quality_sources(
        self,
        report: ResearchReport,
        session_context: SessionMemoryContext
    ) -> None:
        """Store high-quality sources for future research"""
        
        quality_sources = [s for s in report.sources if s.get('reliability', 0) > 0.8]
        
        for source in quality_sources[:10]:  # Top 10 quality sources
            source_content = f"High-Quality Source: {source.get('title', 'Unknown')}\n"
            source_content += f"URL: {source.get('url', '')}\n"
            source_content += f"Author: {source.get('author', 'Unknown')}\n"
            source_content += f"Summary: {source.get('summary', '')}\n"
            source_content += f"Reliability Score: {source.get('reliability', 0):.2%}\n"
            
            await self.memory_manager.store_memory(
                user_id=report.user_id,
                content=source_content,
                memory_type=MemoryType.REFERENCE,
                importance=MemoryImportance.MEDIUM,
                scope=MemoryScope.PRIVATE,
                source_session=report.session_id,
                tags=['quality_source', 'research_reference', source.get('domain', 'web')],
                metadata={
                    'source_url': source.get('url'),
                    'reliability_score': source.get('reliability'),
                    'source_type': source.get('type'),
                    'domain': source.get('domain'),
                    'source_report_id': report.id
                }
            )


# ============================================================================
# ARTIFACT SYSTEM INTEGRATION
# ============================================================================

class ResearchArtifactCreator:
    """Creates research artifacts in the existing artifact system"""
    
    def __init__(self, artifact_manager: Any = None):
        self.artifact_manager = artifact_manager
        
    async def create_research_artifact(
        self,
        report: ResearchReport,
        export_format: ExportFormat = ExportFormat.RESEARCH_ARTIFACT
    ) -> str:
        """Create a research artifact in the artifact system"""
        
        if not ARTIFACT_IMPORTS_AVAILABLE or not self.artifact_manager:
            # Fallback to simple file creation
            return await self._create_standalone_artifact(report, export_format)
        
        artifact_content = self._generate_artifact_content(report, export_format)
        artifact_type = self._map_export_format_to_artifact_type(export_format)
        
        # Create artifact using existing artifact system
        artifact = await self.artifact_manager.create_artifact(
            name=f"Research Report: {report.title}",
            artifact_type=artifact_type,
            created_by=report.user_id,
            session_id=report.session_id,
            description=f"Deep research analysis on {report.title}",
            initial_content=artifact_content,
            security_level=SecurityLevel.SANDBOXED
        )
        
        # Add metadata file
        metadata_content = json.dumps({
            'report_id': report.id,
            'confidence_score': report.confidence_score,
            'source_count': report.source_count,
            'entity_count': report.entity_count,
            'created_at': report.created_at.isoformat(),
            'transform_options': [t.value for t in TransformType]
        }, indent=2)
        
        artifact.add_file('metadata.json', metadata_content, ArtifactType.JSON)
        
        # Add sources data
        sources_content = json.dumps(report.sources, indent=2)
        artifact.add_file('sources.json', sources_content, ArtifactType.JSON)
        
        logger.info(f"Created research artifact {artifact.metadata.artifact_id} for report {report.id}")
        return str(artifact.metadata.artifact_id)
    
    def _generate_artifact_content(
        self,
        report: ResearchReport,
        export_format: ExportFormat
    ) -> str:
        """Generate appropriate content based on export format"""
        
        if export_format == ExportFormat.INTERACTIVE_HTML:
            return self._generate_html_content(report)
        elif export_format == ExportFormat.MARKDOWN:
            return self._generate_markdown_content(report)
        elif export_format == ExportFormat.JSON:
            return json.dumps(asdict(report), indent=2, default=str)
        else:
            return self._generate_html_content(report)  # Default to HTML
    
    def _generate_html_content(self, report: ResearchReport) -> str:
        """Generate interactive HTML content for the research report"""
        
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Somnus Research Report</title>
    <style>
        /* Simplified embedded styles for artifact display */
        body {{ 
            font-family: 'Inter', sans-serif; 
            background: #0f0f1a; 
            color: #f8fafc; 
            margin: 0; 
            padding: 20px; 
        }}
        .report-container {{ 
            max-width: 1000px; 
            margin: 0 auto; 
        }}
        .report-header {{ 
            background: rgba(31, 41, 55, 0.7); 
            padding: 24px; 
            border-radius: 12px; 
            margin-bottom: 24px; 
        }}
        .report-title {{ 
            font-size: 28px; 
            font-weight: 700; 
            margin-bottom: 12px; 
        }}
        .report-meta {{ 
            display: flex; 
            gap: 16px; 
            font-size: 14px; 
            color: #94a3b8; 
        }}
        .confidence-badge {{ 
            background: #10b981; 
            color: white; 
            padding: 4px 12px; 
            border-radius: 16px; 
            font-size: 12px; 
            font-weight: 600; 
        }}
        .report-content {{ 
            background: rgba(31, 41, 55, 0.7); 
            padding: 24px; 
            border-radius: 12px; 
            line-height: 1.6; 
        }}
        .sources-section {{ 
            margin-top: 24px; 
            background: rgba(31, 41, 55, 0.7); 
            padding: 20px; 
            border-radius: 12px; 
        }}
        .source-item {{ 
            padding: 12px; 
            background: #16213e; 
            border-radius: 8px; 
            margin-bottom: 8px; 
        }}
        .source-title {{ 
            font-weight: 600; 
            color: #e2e8f0; 
        }}
        .source-url {{ 
            font-size: 12px; 
            color: #94a3b8; 
            word-break: break-all; 
        }}
        .transform-panel {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(31, 41, 55, 0.9);
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }}
        .transform-button {{
            background: #3b82f6;
            border: none;
            border-radius: 8px;
            color: white;
            padding: 8px 16px;
            font-weight: 600;
            cursor: pointer;
            margin: 4px;
            display: block;
            width: 100%;
        }}
        .transform-button:hover {{
            background: #8b5cf6;
        }}
    </style>
</head>
<body>
    <div class="transform-panel">
        <h4 style="margin: 0 0 12px 0; color: #e2e8f0;">Transform Report</h4>
        <button class="transform-button" onclick="transformReport('website')">🌐 Website</button>
        <button class="transform-button" onclick="transformReport('infographic')">📊 Infographic</button>
        <button class="transform-button" onclick="transformReport('presentation')">🎯 Presentation</button>
        <button class="transform-button" onclick="transformReport('summary')">📄 Summary</button>
        <button class="transform-button" onclick="transformReport('dashboard')">📈 Dashboard</button>
    </div>

    <div class="report-container">
        <div class="report-header">
            <h1 class="report-title">{title}</h1>
            <div class="report-meta">
                <span>📅 {created_at}</span>
                <span>📚 {source_count} sources</span>
                <span>🔗 {entity_count} entities</span>
                <span class="confidence-badge">Confidence: {confidence}%</span>
            </div>
        </div>

        <div class="report-content">
            <h2>Executive Summary</h2>
            <p>{summary}</p>
            
            <h2>Detailed Analysis</h2>
            <div>{content}</div>
            
            {contradictions_section}
        </div>

        <div class="sources-section">
            <h3>Key Sources ({source_count})</h3>
            {sources_html}
        </div>
    </div>

    <script>
        function transformReport(type) {{
            console.log('Transforming to:', type);
            
            // This would call the backend transform API
            fetch('/api/research/transform', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    report_id: '{report_id}',
                    transform_type: type,
                    user_id: '{user_id}'
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    alert(`Report transformed to ${{type}}! Opening new artifact...`);
                    // Would open new artifact in artifact system
                }} else {{
                    alert('Transform failed: ' + data.error);
                }}
            }})
            .catch(error => {{
                console.error('Transform error:', error);
                alert('Transform request failed');
            }});
        }}
    </script>
</body>
</html>'''
        
        # Generate sources HTML
        sources_html = ""
        for source in report.sources[:10]:
            reliability_color = "#10b981" if source.get('reliability', 0) > 0.8 else "#f59e0b" if source.get('reliability', 0) > 0.6 else "#ef4444"
            sources_html += f'''
            <div class="source-item">
                <div class="source-title">{source.get('title', 'Unknown Source')}</div>
                <div class="source-url">{source.get('url', '')}</div>
                <div style="margin-top: 4px; color: {reliability_color}; font-size: 12px;">
                    Reliability: {source.get('reliability', 0):.1%}
                </div>
            </div>
            '''
        
        # Generate contradictions section
        contradictions_section = ""
        if report.contradictions:
            contradictions_section = "<h2>Identified Contradictions</h2>"
            for contradiction in report.contradictions:
                contradictions_section += f'''
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 16px; margin: 12px 0;">
                    <h4 style="color: #ef4444; margin: 0 0 8px 0;">{contradiction.get('title', 'Unknown Contradiction')}</h4>
                    <p style="margin: 0;">{contradiction.get('description', '')}</p>
                </div>
                '''
        
        return html_template.format(
            title=report.title,
            created_at=report.created_at.strftime('%B %d, %Y'),
            source_count=report.source_count,
            entity_count=report.entity_count,
            confidence=int(report.confidence_score * 100),
            summary=report.summary,
            content=report.content.replace('\n', '<br>'),
            contradictions_section=contradictions_section,
            sources_html=sources_html,
            report_id=report.id,
            user_id=report.user_id
        )
    
    def _generate_markdown_content(self, report: ResearchReport) -> str:
        """Generate markdown content for the research report"""
        
        markdown_content = f"""# {report.title}

**Generated:** {report.created_at.strftime('%B %d, %Y at %I:%M %p')}  
**Sources:** {report.source_count} | **Entities:** {report.entity_count} | **Confidence:** {report.confidence_score:.1%}

## Executive Summary

{report.summary}

## Detailed Analysis

{report.content}
"""
        
        if report.contradictions:
            markdown_content += "\n## Identified Contradictions\n\n"
            for i, contradiction in enumerate(report.contradictions, 1):
                markdown_content += f"{i}. **{contradiction.get('title', 'Unknown')}**\n"
                markdown_content += f"   {contradiction.get('description', '')}\n\n"
        
        markdown_content += "\n## Key Sources\n\n"
        for i, source in enumerate(report.sources[:10], 1):
            reliability = source.get('reliability', 0)
            markdown_content += f"{i}. [{source.get('title', 'Unknown Source')}]({source.get('url', '#')})\n"
            markdown_content += f"   - Reliability: {reliability:.1%}\n"
            if 'summary' in source:
                markdown_content += f"   - Summary: {source['summary']}\n"
            markdown_content += "\n"
        
        return markdown_content
    
    def _map_export_format_to_artifact_type(self, export_format: ExportFormat) -> 'ArtifactType':
        """Map export format to artifact type"""
        
        if not ARTIFACT_IMPORTS_AVAILABLE:
            return "text/html"  # Fallback
            
        mapping = {
            ExportFormat.INTERACTIVE_HTML: ArtifactType.HTML,
            ExportFormat.MARKDOWN: ArtifactType.MARKDOWN,
            ExportFormat.JSON: ArtifactType.JSON,
            ExportFormat.RESEARCH_ARTIFACT: ArtifactType.HTML
        }
        return mapping.get(export_format, ArtifactType.HTML)
    
    async def _create_standalone_artifact(
        self,
        report: ResearchReport,
        export_format: ExportFormat
    ) -> str:
        """Create standalone artifact file when artifact system not available"""
        
        artifact_id = str(uuid4())
        artifact_dir = Path(f"artifacts/research/{artifact_id}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Main content file
        if export_format == ExportFormat.INTERACTIVE_HTML:
            content = self._generate_html_content(report)
            main_file = artifact_dir / "index.html"
        elif export_format == ExportFormat.MARKDOWN:
            content = self._generate_markdown_content(report)
            main_file = artifact_dir / "report.md"
        else:
            content = json.dumps(asdict(report), indent=2, default=str)
            main_file = artifact_dir / "report.json"
        
        async with aiofiles.open(main_file, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        # Metadata file
        metadata = {
            'artifact_id': artifact_id,
            'report_id': report.id,
            'created_at': report.created_at.isoformat(),
            'format': export_format.value,
            'transform_options': [t.value for t in TransformType]
        }
        
        async with aiofiles.open(artifact_dir / "metadata.json", 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
        
        logger.info(f"Created standalone research artifact {artifact_id}")
        return artifact_id


# ============================================================================
# AI TRANSFORMATION ENGINE
# ============================================================================

class ReportTransformationEngine:
    """Handles AI-powered transformation of research reports"""
    
    def __init__(self, model_loader: Any = None):
        self.model_loader = model_loader
        self.transform_templates = self._load_transform_templates()
    
    def _load_transform_templates(self) -> Dict[TransformType, str]:
        """Load transformation prompt templates"""
        
        return {
            TransformType.WEBSITE: """Transform the following research report into an interactive website format.

Create an engaging, well-structured website that includes:
- Hero section with key findings
- Navigation for different sections
- Interactive elements where appropriate
- Professional styling with responsive design
- Clear call-to-action sections

Research Report:
{report_content}

Generate complete HTML with embedded CSS and JavaScript for a professional, interactive website.""",

            TransformType.INFOGRAPHIC: """Transform the following research report into a visual infographic format.

Create an HTML-based infographic that includes:
- Compelling visual hierarchy
- Key statistics prominently displayed
- Icon usage and visual elements
- Color-coded sections
- Charts and graphs where data permits
- Minimal text, maximum visual impact

Research Report:
{report_content}

Generate complete HTML with embedded CSS to create a visually striking infographic.""",

            TransformType.PRESENTATION: """Transform the following research report into a slide presentation format.

Create an HTML-based presentation that includes:
- Title slide with key overview
- 5-7 content slides covering main points
- Conclusion slide with actionable insights
- Clean, professional slide design
- Speaker notes embedded
- Navigation controls

Research Report:
{report_content}

Generate complete HTML presentation with CSS animations and slide transitions.""",

            TransformType.SUMMARY: """Transform the following research report into a concise executive summary.

Create a focused summary that includes:
- 2-3 paragraph executive overview
- Key findings in bullet points
- Critical insights and implications
- Actionable recommendations
- Risk factors or considerations
- Next steps

Research Report:
{report_content}

Generate a professional executive summary suitable for decision-makers.""",

            TransformType.DASHBOARD: """Transform the following research report into an analytics dashboard format.

Create an HTML-based dashboard that includes:
- Key performance indicators (KPIs)
- Visual charts and graphs
- Progress indicators
- Trend analysis sections
- Filter and interaction capabilities
- Real-time-style data presentation

Research Report:
{report_content}

Generate complete HTML dashboard with embedded CSS and JavaScript for data visualization."""
        }
    
    async def transform_report(
        self,
        report: ResearchReport,
        transform_request: TransformRequest
    ) -> ExportResult:
        """Transform research report using AI"""
        
        start_time = time.time()
        
        try:
            # Get transformation prompt
            base_prompt = self._get_transform_prompt(
                transform_request.transform_type,
                report,
                transform_request
            )
            
            # Apply custom instructions if provided
            if transform_request.custom_instructions:
                base_prompt += f"\n\nAdditional Instructions: {transform_request.custom_instructions}"
            
            # Generate transformation using model
            transformed_content = await self._generate_with_model(
                base_prompt,
                transform_request.model_name
            )
            
            # Create new artifact with transformed content
            transformed_report = self._create_transformed_report(
                report,
                transformed_content,
                transform_request
            )
            
            # Save as new artifact
            artifact_creator = ResearchArtifactCreator(self.artifact_manager if hasattr(self, 'artifact_manager') else None)
            artifact_id = await artifact_creator.create_research_artifact(
                transformed_report,
                ExportFormat.INTERACTIVE_HTML
            )
            
            export_time = time.time() - start_time
            
            result = ExportResult(
                success=True,
                export_id=str(uuid4()),
                report_id=report.id,
                artifact_id=artifact_id,
                export_time=export_time,
                format=ExportFormat.INTERACTIVE_HTML,
                metadata={
                    'transform_type': transform_request.transform_type.value,
                    'original_report_id': report.id,
                    'target_audience': transform_request.target_audience,
                    'model_used': transform_request.model_name or 'default'
                }
            )
            
            logger.info(f"Successfully transformed report {report.id} to {transform_request.transform_type.value}")
            return result
            
        except Exception as e:
            error_msg = f"Transform failed: {str(e)}"
            logger.error(f"Report transformation error: {error_msg}")
            logger.error(traceback.format_exc())
            
            return ExportResult(
                success=False,
                export_id=str(uuid4()),
                report_id=report.id,
                export_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def _get_transform_prompt(
        self,
        transform_type: TransformType,
        report: ResearchReport,
        request: TransformRequest
    ) -> str:
        """Get appropriate transformation prompt"""
        
        template = self.transform_templates.get(transform_type, self.transform_templates[TransformType.WEBSITE])
        
        # Prepare report content for prompt
        report_content = f"""
Title: {report.title}
Summary: {report.summary}
Full Content: {report.content}
Sources: {report.source_count}
Entities: {report.entity_count}
Confidence: {report.confidence_score:.1%}
"""
        
        if report.contradictions:
            report_content += f"\nContradictions: {len(report.contradictions)} identified"
        
        # Add audience-specific instructions
        audience_instructions = {
            "executive": "Focus on high-level insights and business implications. Use professional language suitable for C-level executives.",
            "technical": "Include technical details and methodological information. Use precise terminology appropriate for technical professionals.",
            "general": "Use accessible language suitable for a general audience. Explain technical concepts clearly.",
            "academic": "Include scholarly perspective with proper citations and rigorous analysis."
        }
        
        audience_context = audience_instructions.get(request.target_audience, audience_instructions["general"])
        
        return template.format(report_content=report_content) + f"\n\nTarget Audience: {audience_context}"
    
    async def _generate_with_model(
        self,
        prompt: str,
        model_name: Optional[str] = None
    ) -> str:
        """Generate content using AI model"""
        
        if not self.model_loader:
            # Fallback for testing - return formatted placeholder
            return f"""<!DOCTYPE html>
<html><head><title>Transformed Report</title></head>
<body><h1>AI Transformation</h1>
<p>This would be the AI-generated transformation of the research report.</p>
<p>Prompt used: {prompt[:200]}...</p>
</body></html>"""
        
        try:
            # Use the model loader to generate content
            model = await self.model_loader.get_model(model_name)
            
            generation_config = {
                'max_tokens': 8192,
                'temperature': 0.7,
                'top_p': 0.9
            }
            
            response = await model.generate(prompt, **generation_config)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            raise
    
    def _create_transformed_report(
        self,
        original_report: ResearchReport,
        transformed_content: str,
        transform_request: TransformRequest
    ) -> ResearchReport:
        """Create new report object with transformed content"""
        
        return ResearchReport(
            id=str(uuid4()),
            session_id=original_report.session_id,
            user_id=original_report.user_id,
            title=f"{original_report.title} - {transform_request.transform_type.value.title()}",
            content=transformed_content,
            summary=f"Transformed version of original research as {transform_request.transform_type.value}",
            sources=original_report.sources,
            entities=original_report.entities,
            contradictions=original_report.contradictions,
            confidence_score=original_report.confidence_score,
            coherence_score=original_report.coherence_score,
            depth_score=original_report.depth_score,
            entity_count=original_report.entity_count,
            source_count=original_report.source_count,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={
                **original_report.metadata,
                'transform_type': transform_request.transform_type.value,
                'original_report_id': original_report.id,
                'target_audience': transform_request.target_audience
            },
            checksum=hashlib.sha256(transformed_content.encode()).hexdigest()[:16],
            file_size=len(transformed_content.encode())
        )


# ============================================================================
# MAIN REPORT EXPORTER CLASS
# ============================================================================

class ResearchReportExporter:
    """Main class for research report export and transformation"""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        artifact_manager: Any = None,
        model_loader: Any = None
    ):
        self.memory_manager = memory_manager
        self.artifact_manager = artifact_manager
        self.model_loader = model_loader
        
        # Initialize subsystems
        self.memory_integrator = ResearchMemoryIntegrator(memory_manager)
        self.artifact_creator = ResearchArtifactCreator(artifact_manager)
        self.transform_engine = ReportTransformationEngine(model_loader)
        self.transform_engine.artifact_manager = artifact_manager
        
        logger.info("ResearchReportExporter initialized with full system integration")
    
    async def export_research_session(
        self,
        research_session: Any,
        session_context: SessionMemoryContext,
        export_format: ExportFormat = ExportFormat.RESEARCH_ARTIFACT
    ) -> ExportResult:
        """Export complete research session to artifact and memory"""
        
        start_time = time.time()
        
        try:
            # Convert research session to report
            report = await self._convert_session_to_report(research_session)
            
            # Store in memory system for persistent context
            memory_id = await self.memory_integrator.store_research_report(
                report,
                session_context
            )
            
            # Create artifact
            artifact_id = await self.artifact_creator.create_research_artifact(
                report,
                export_format
            )
            
            export_time = time.time() - start_time
            
            result = ExportResult(
                success=True,
                export_id=str(uuid4()),
                report_id=report.id,
                artifact_id=artifact_id,
                memory_id=str(memory_id),
                export_time=export_time,
                format=export_format,
                file_size=report.file_size,
                metadata={
                    'session_id': research_session.session_id if hasattr(research_session, 'session_id') else 'unknown',
                    'memory_stored': True,
                    'artifact_created': True,
                    'confidence_score': report.confidence_score,
                    'source_count': report.source_count
                }
            )
            
            logger.info(f"Successfully exported research session to artifact {artifact_id} and memory {memory_id}")
            return result
            
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            logger.error(f"Research export error: {error_msg}")
            logger.error(traceback.format_exc())
            
            return ExportResult(
                success=False,
                export_id=str(uuid4()),
                report_id="unknown",
                export_time=time.time() - start_time,
                error_message=error_msg
            )
    
    async def transform_existing_report(
        self,
        transform_request: TransformRequest
    ) -> ExportResult:
        """Transform existing research report to new format"""
        
        try:
            # Retrieve report from memory or artifact system
            report = await self._retrieve_report(transform_request.report_id)
            
            if not report:
                return ExportResult(
                    success=False,
                    export_id=str(uuid4()),
                    report_id=transform_request.report_id,
                    error_message="Report not found"
                )
            
            # Perform transformation
            return await self.transform_engine.transform_report(report, transform_request)
            
        except Exception as e:
            error_msg = f"Transform failed: {str(e)}"
            logger.error(f"Report transform error: {error_msg}")
            
            return ExportResult(
                success=False,
                export_id=str(uuid4()),
                report_id=transform_request.report_id,
                error_message=error_msg
            )
    
    async def _convert_session_to_report(self, research_session: Any) -> ResearchReport:
        """Convert research session object to standardized report"""
        
        if RESEARCH_IMPORTS_AVAILABLE and hasattr(research_session, '__dict__'):
            # Use actual research session data
            session_data = research_session.__dict__
        else:
            # Fallback for testing
            session_data = {
                'session_id': getattr(research_session, 'session_id', str(uuid4())),
                'query': getattr(research_session, 'query', 'Research Query'),
                'results': getattr(research_session, 'results', []),
                'entities': getattr(research_session, 'entities', []),
                'contradictions': getattr(research_session, 'contradictions', []),
                'sources': getattr(research_session, 'sources', []),
                'created_at': getattr(research_session, 'created_at', datetime.now(timezone.utc))
            }
        
        # Generate comprehensive report content
        content = self._generate_report_content(session_data)
        summary = self._generate_summary(session_data)
        
        # Calculate metrics
        confidence_score = self._calculate_confidence(session_data)
        coherence_score = self._calculate_coherence(session_data)
        depth_score = self._calculate_depth(session_data)
        
        report = ResearchReport(
            id=str(uuid4()),
            session_id=session_data['session_id'],
            user_id=getattr(research_session, 'user_id', 'unknown'),
            title=f"Research Analysis: {session_data.get('query', 'Unknown Topic')}",
            content=content,
            summary=summary,
            sources=session_data.get('sources', []),
            entities=session_data.get('entities', []),
            contradictions=session_data.get('contradictions', []),
            confidence_score=confidence_score,
            coherence_score=coherence_score,
            depth_score=depth_score,
            entity_count=len(session_data.get('entities', [])),
            source_count=len(session_data.get('sources', [])),
            created_at=session_data.get('created_at', datetime.now(timezone.utc)),
            updated_at=datetime.now(timezone.utc),
            metadata={
                'export_version': '1.0',
                'original_query': session_data.get('query'),
                'processing_time': getattr(research_session, 'processing_time', 0),
                'research_depth': depth_score
            },
            checksum=hashlib.sha256(content.encode()).hexdigest()[:16],
            file_size=len(content.encode())
        )
        
        return report
    
    def _generate_report_content(self, session_data: Dict[str, Any]) -> str:
        """Generate comprehensive report content from session data"""
        
        content_parts = [
            "# Research Analysis Report",
            f"\n## Research Objective\n{session_data.get('query', 'Unknown research objective')}",
            "\n## Key Findings"
        ]
        
        # Add findings from results
        results = session_data.get('results', [])
        if results:
            for i, result in enumerate(results[:10], 1):
                if isinstance(result, dict):
                    content_parts.append(f"\n### Finding {i}: {result.get('title', 'Untitled Finding')}")
                    content_parts.append(result.get('content', 'No content available'))
                else:
                    content_parts.append(f"\n### Finding {i}\n{str(result)}")
        else:
            content_parts.append("\nNo specific findings recorded in this research session.")
        
        # Add entity analysis
        entities = session_data.get('entities', [])
        if entities:
            content_parts.append("\n## Entity Analysis")
            content_parts.append(f"This research identified {len(entities)} key entities:")
            for entity in entities[:20]:  # Top 20 entities
                if isinstance(entity, dict):
                    name = entity.get('name', 'Unknown Entity')
                    description = entity.get('description', 'No description available')
                    content_parts.append(f"\n- **{name}**: {description}")
                else:
                    content_parts.append(f"\n- {str(entity)}")
        
        # Add source analysis
        sources = session_data.get('sources', [])
        if sources:
            content_parts.append(f"\n## Source Analysis\nThis analysis incorporates information from {len(sources)} sources:")
            quality_sources = [s for s in sources if isinstance(s, dict) and s.get('reliability', 0) > 0.8]
            content_parts.append(f"\n- High-quality sources: {len(quality_sources)}")
            content_parts.append(f"- Total sources analyzed: {len(sources)}")
            if quality_sources:
                content_parts.append("\n### Top Quality Sources:")
                for source in quality_sources[:5]:
                    title = source.get('title', 'Unknown Source')
                    reliability = source.get('reliability', 0)
                    content_parts.append(f"- {title} (Reliability: {reliability:.1%})")
        
        return "\n".join(content_parts)
    
    def _generate_summary(self, session_data: Dict[str, Any]) -> str:
        """Generate executive summary"""
        
        query = session_data.get('query', 'research topic')
        entity_count = len(session_data.get('entities', []))
        source_count = len(session_data.get('sources', []))
        contradiction_count = len(session_data.get('contradictions', []))
        
        summary = f"This research analysis examined {query}, incorporating insights from {source_count} sources and identifying {entity_count} key entities."
        
        if contradiction_count > 0:
            summary += f" The analysis detected {contradiction_count} contradictions that warrant further investigation."
        
        if source_count > 10:
            summary += " The comprehensive source base provides high confidence in the findings."
        elif source_count > 5:
            summary += " The analysis is based on a moderate source base with adequate coverage."
        else:
            summary += " Additional sources may be needed to strengthen the analysis."
        
        return summary
    
    def _calculate_confidence(self, session_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on session data"""
        
        source_count = len(session_data.get('sources', []))
        entity_count = len(session_data.get('entities', []))
        contradiction_count = len(session_data.get('contradictions', []))
        
        # Base confidence from source count
        source_confidence = min(source_count / 20, 1.0) * 0.4
        
        # Entity analysis confidence
        entity_confidence = min(entity_count / 50, 1.0) * 0.3
        
        # Contradiction penalty
        contradiction_penalty = min(contradiction_count / 10, 0.2) * 0.2
        
        # Base methodology confidence
        base_confidence = 0.3
        
        total_confidence = base_confidence + source_confidence + entity_confidence - contradiction_penalty
        return max(0.1, min(1.0, total_confidence))
    
    def _calculate_coherence(self, session_data: Dict[str, Any]) -> float:
        """Calculate coherence score"""
        
        contradiction_count = len(session_data.get('contradictions', []))
        source_count = max(1, len(session_data.get('sources', [])))
        
        # High contradictions reduce coherence
        contradiction_ratio = contradiction_count / source_count
        coherence = max(0.5, 1.0 - (contradiction_ratio * 2))
        
        return coherence
    
    def _calculate_depth(self, session_data: Dict[str, Any]) -> int:
        """Calculate research depth score"""
        
        source_count = len(session_data.get('sources', []))
        entity_count = len(session_data.get('entities', []))
        results_count = len(session_data.get('results', []))
        
        depth = min(10, (source_count // 3) + (entity_count // 10) + (results_count // 2))
        return max(1, depth)
    
    async def _retrieve_report(self, report_id: str) -> Optional[ResearchReport]:
        """Retrieve report from memory or artifact system"""
        
        try:
            # Try to retrieve from memory first
            memories = await self.memory_manager.retrieve_memories(
                user_id="system",  # This would need proper user context
                query=f"report_id:{report_id}",
                limit=1
            )
            
            if memories:
                memory = memories[0]
                # Reconstruct report from memory (simplified)
                return ResearchReport(
                    id=report_id,
                    session_id=memory.get('metadata', {}).get('research_session_id', 'unknown'),
                    user_id=memory.get('user_id', 'unknown'),
                    title=memory.get('content', '').split('\n')[0].replace('# Research Report: ', ''),
                    content=memory.get('content', ''),
                    summary="Retrieved from memory",
                    sources=[],
                    entities=[],
                    contradictions=[],
                    confidence_score=memory.get('metadata', {}).get('confidence_score', 0.5),
                    coherence_score=memory.get('metadata', {}).get('coherence_score', 0.5),
                    depth_score=memory.get('metadata', {}).get('depth_score', 1),
                    entity_count=memory.get('metadata', {}).get('entity_count', 0),
                    source_count=memory.get('metadata', {}).get('source_count', 0),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    metadata=memory.get('metadata', {}),
                    checksum="",
                    file_size=0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve report {report_id}: {e}")
            return None


# ============================================================================
# FACTORY FUNCTION FOR EASY INTEGRATION
# ============================================================================

async def create_research_report_exporter(
    memory_manager: MemoryManager,
    artifact_manager: Any = None,
    model_loader: Any = None
) -> ResearchReportExporter:
    """Factory function to create configured report exporter"""
    
    exporter = ResearchReportExporter(
        memory_manager=memory_manager,
        artifact_manager=artifact_manager,
        model_loader=model_loader
    )
    
    logger.info("Research report exporter created with full system integration")
    return exporter


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def example_usage():
    """Example of how to use the report exporter"""
    
    # This would be called with real system components
    from core.memory_core import MemoryManager, MemoryConfiguration
    
    # Initialize memory manager
    memory_config = MemoryConfiguration()
    memory_manager = MemoryManager(memory_config)
    await memory_manager.initialize()
    
    # Create exporter
    exporter = await create_research_report_exporter(
        memory_manager=memory_manager
    )
    
    # Example research session (mock data)
    class MockResearchSession:
        def __init__(self):
            self.session_id = str(uuid4())
            self.user_id = "test_user"
            self.query = "AI cognitive architectures"
            self.results = [
                {"title": "Attention mechanisms", "content": "Analysis of attention in transformers..."},
                {"title": "Memory systems", "content": "Cognitive memory architectures..."}
            ]
            self.entities = [
                {"name": "Transformer", "type": "architecture", "description": "Neural network architecture"}
            ]
            self.sources = [
                {"title": "Attention Is All You Need", "url": "https://arxiv.org/abs/1706.03762", "reliability": 0.95}
            ]
            self.contradictions = []
            self.created_at = datetime.now(timezone.utc)
    
    # Mock session context
    class MockSessionContext:
        def __init__(self):
            self.user_id = "test_user"
            self.session_id = str(uuid4())
    
    # Export research session
    mock_session = MockResearchSession()
    mock_context = MockSessionContext()
    
    result = await exporter.export_research_session(
        mock_session,
        mock_context,
        ExportFormat.RESEARCH_ARTIFACT
    )
    
    print(f"Export result: {result}")
    
    # Example transformation
    transform_request = TransformRequest(
        report_id=result.report_id,
        user_id="test_user",
        transform_type=TransformType.WEBSITE,
        target_audience="general"
    )
    
    transform_result = await exporter.transform_existing_report(transform_request)
    print(f"Transform result: {transform_result}")


if __name__ == "__main__":
    asyncio.run(example_usage())