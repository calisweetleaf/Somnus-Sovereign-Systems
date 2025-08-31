"""
Somnus Sovereign Systems - Virtual Machine AI Browser Research System - 1 of 3 methods of web searching for the AI/LLM. One in the virtual machine, one system wide togglable, and one dedicated deep research subsystem platform.

A comprehensive browser-based AI research system with visual interaction,
intelligent content extraction, and automated research workflows.
"""
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import asyncio
import json
import time
import re
import hashlib
from uuid import uuid4
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlparse

# Import placeholder for missing dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
except ImportError:
    # Create simple placeholder for missing scikit-learn
    class TfidfVectorizer:
        def __init__(self, **kwargs):
            pass
        def fit_transform(self, texts):
            return [[0] * len(texts)]
    import numpy as np

from backend.virtual_machine.ai_action_orchestrator import AIActionOrchestrator


class AIBrowserResearch:
    """
    AI browser-based research system with visual interaction capabilities
    and intelligent workflow management
    """
    
    def __init__(self, vm_instance, orchestrator: AIActionOrchestrator):
        self.vm = vm_instance
        self.orchestrator = orchestrator
        self.browsers = {
            'firefox': '/usr/bin/firefox',
            'chrome': '/usr/bin/chromium-browser', 
            'research_browser': '/home/ai/tools/research-browser'
        }
        
        # Load research workflows from templates
        self.research_workflows = self._load_research_templates()
    
    def _load_research_templates(self) -> Dict[str, Any]:
        """Load research workflow templates from /templates/ directory"""
        templates_dir = Path('/templates')
        workflows = {}
        
        if templates_dir.exists():
            for template_file in templates_dir.glob('*.json'):
                try:
                    with open(template_file, 'r') as f:
                        workflow_name = template_file.stem
                        workflows[workflow_name] = json.load(f)
                except Exception as e:
                    print(f"Failed to load template {template_file}: {e}")
        
        return workflows
    
    async def conduct_visual_research(self, query: str, workflow_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Conduct comprehensive research using browser automation with visual interaction
        
        Args:
            query: Research query string
            workflow_type: Type of research workflow to execute
            
        Returns:
            Comprehensive research results with sources, analysis, and artifacts
        """
        browser_session = await self._start_research_browser()
        research_plan = await self._generate_research_plan(query, workflow_type)
        
        results = {
            'query': query,
            'workflow_type': workflow_type,
            'sources_found': [],
            'screenshots': [],
            'downloaded_documents': [],
            'research_notes': [],
            'fact_check_results': [],
            'cross_reference_analysis': [],
            'credibility_assessments': []
        }
        
        for step in research_plan['steps']:
            step_results = await self._execute_research_step(browser_session, step, results)
            self._merge_step_results(results, step_results)
        
        # Synthesize findings and generate insights
        results['synthesis'] = await self._synthesize_research_findings(results)
        results['insights'] = await self._generate_research_insights(results)
        
        await self._finalize_research_session(browser_session, results)
        
        return results
    
    async def _start_research_browser(self):
        """Initialize browser session with research-optimized configuration"""
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=False,
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions-except=/home/ai/extensions/',
                '--load-extension=/home/ai/extensions/'
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        )
        
        page = await context.new_page()
        return page
    
    async def _generate_research_plan(self, query: str, workflow_type: str) -> Dict[str, Any]:
        """Generate adaptive research plan based on query analysis and workflow type"""
        workflow_template = self.research_workflows.get(workflow_type, {})
        
        # Analyze query to determine optimal research strategy
        query_analysis = await self._analyze_research_query(query)
        
        research_plan = {
            'query': query,
            'strategy': query_analysis['strategy'],
            'estimated_sources': query_analysis['expected_source_count'],
            'steps': []
        }
        
        # Generate search steps
        for search_engine in query_analysis['recommended_engines']:
            research_plan['steps'].append({
                'type': 'search_engine',
                'search_engine': search_engine,
                'search_terms': query_analysis['optimized_terms'][search_engine],
                'max_results': workflow_template.get('max_results_per_engine', 20)
            })
        
        # Add deep analysis steps
        research_plan['steps'].append({
            'type': 'deep_read',
            'selection_criteria': workflow_template.get('selection_criteria', 'relevance_score'),
            'max_articles': workflow_template.get('max_deep_reads', 10)
        })
        
        # Add cross-referencing steps
        research_plan['steps'].append({
            'type': 'cross_reference',
            'verification_threshold': workflow_template.get('verification_threshold', 0.8),
            'source_diversity_requirement': workflow_template.get('source_diversity', 3)
        })
        
        return research_plan
    
    async def _execute_research_step(self, browser, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual research step and return results"""
        step_type = step['type']
        
        if step_type == 'search_engine':
            return await self._browser_search(
                browser, 
                step['search_terms'], 
                step['search_engine'],
                step.get('max_results', 20)
            )
        elif step_type == 'deep_read':
            return await self._execute_deep_read_phase(browser, step, results)
        elif step_type == 'cross_reference':
            return await self._execute_cross_reference_phase(browser, step, results)
        elif step_type == 'document_download':
            docs = await self._download_research_documents(browser, step['document_urls'])
            return {'downloaded_documents': docs}
        else:
            return {'error': f'Unknown step type: {step_type}'}
    
    async def _browser_search(self, browser, terms: str, engine: str, max_results: int = 20) -> Dict[str, Any]:
        """Execute browser search with intelligent result extraction"""
        search_engines = {
            'google': 'https://google.com/search?q=',
            'bing': 'https://bing.com/search?q=',
            'duckduckgo': 'https://duckduckgo.com/?q=',
            'arxiv': 'https://arxiv.org/search/?query=',
            'scholar': 'https://scholar.google.com/scholar?q=',
            'pubmed': 'https://pubmed.ncbi.nlm.nih.gov/?term=',
            'semantic_scholar': 'https://www.semanticscholar.org/search?q='
        }
        
        if engine not in search_engines:
            return {'error': f'Unsupported search engine: {engine}'}
        
        url = search_engines[engine] + terms.replace(' ', '+')
        
        try:
            await browser.goto(url, wait_until='domcontentloaded')
            
            # Capture search results page
            screenshot_path = f'/home/ai/research/screenshots/search_{engine}_{int(time.time())}.png'
            await browser.screenshot(path=screenshot_path)
            
            # Extract results using engine-specific selectors
            results = await self._extract_search_results(browser, engine, max_results)
            
            return {
                'engine': engine,
                'query': terms,
                'results': results,
                'screenshot': screenshot_path,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {'error': f'Search failed for {engine}: {str(e)}'}
    
    async def _extract_search_results(self, browser, engine: str, max_results: int) -> List[Dict[str, Any]]:
        """Extract search results with engine-specific optimization"""
        
        engine_selectors = {
            'google': {
                'result_container': 'div[data-ved]',
                'title': 'h3',
                'link': 'a',
                'snippet': '.IsZvec, .aCOpRe'
            },
            'bing': {
                'result_container': '.b_algo',
                'title': 'h2 a',
                'link': 'h2 a',
                'snippet': '.b_caption p'
            },
            'scholar': {
                'result_container': '.gs_ri',
                'title': '.gs_rt a',
                'link': '.gs_rt a',
                'snippet': '.gs_rs'
            }
        }
        
        selectors = engine_selectors.get(engine, engine_selectors['google'])
        
        results = await browser.evaluate(f"""
            () => {{
                const results = [];
                const containers = document.querySelectorAll('{selectors["result_container"]}');
                
                for (let i = 0; i < Math.min(containers.length, {max_results}); i++) {{
                    const container = containers[i];
                    const titleEl = container.querySelector('{selectors["title"]}');
                    const linkEl = container.querySelector('{selectors["link"]}');
                    const snippetEl = container.querySelector('{selectors["snippet"]}');
                    
                    if (titleEl && linkEl) {{
                        results.push({{
                            title: titleEl.textContent?.trim() || '',
                            url: linkEl.href || '',
                            snippet: snippetEl?.textContent?.trim() || '',
                            position: i + 1
                        }});
                    }}
                }}
                
                return results;
            }}
        """)
        
        return results
    
    async def _execute_deep_read_phase(self, browser, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deep reading phase with intelligent article selection"""
        import logging
        logger = logging.getLogger(__name__)
        
        deep_read_results = {
            'articles_analyzed': [],
            'content_extracted': [],
            'key_insights': [],
            'citations_found': []
        }

        # Get sources from previous search steps
        sources_found = []
        
        # Collect sources from all previous search results
        if 'sources_found' in results:
            sources_found.extend(results['sources_found'])
        
        # Also check if there are specific URLs in the step
        if 'urls' in step:
            for url in step['urls']:
                sources_found.append({'url': url, 'title': 'Direct URL', 'snippet': ''})
        
        if not sources_found:
            logger.warning("No sources found for deep reading phase")
            return deep_read_results

        # 1. Score and rank sources by relevance
        scored_sources = []
        query = results.get('query', '')
        
        for source in sources_found:
            try:
                score = self._calculate_relevance_score(
                    source, 
                    step.get('selection_criteria', 'relevance_score'), 
                    query
                )
                scored_sources.append((score, source))
            except Exception as e:
                logger.warning(f"Failed to score source {source.get('url', 'unknown')}: {e}")
                scored_sources.append((0.0, source))

        # Sort by score in descending order
        scored_sources.sort(key=lambda x: x[0], reverse=True)

        # 2. Select the top articles for deep reading
        max_articles = step.get('max_articles', 10)
        selected_articles = [source for score, source in scored_sources[:max_articles]]
        
        logger.info(f"Selected {len(selected_articles)} articles for deep reading")

        # 3. Process each selected article
        for i, article_source in enumerate(selected_articles):
            url = article_source.get('url')
            if not url:
                continue
            
            try:
                logger.info(f"Deep reading article {i+1}/{len(selected_articles)}: {url}")
                
                # Add small delay to be respectful to servers
                if i > 0:
                    await asyncio.sleep(2)
                
                article_analysis = await self._deep_read_article(browser, url)
                
                if article_analysis and not article_analysis.get('error'):
                    deep_read_results['articles_analyzed'].append(article_analysis)
                    
                    # Extract content and insights
                    article_data = article_analysis.get('article_data', {})
                    analysis = article_analysis.get('analysis', {})
                    
                    if article_data.get('content'):
                        deep_read_results['content_extracted'].append({
                            'url': url,
                            'title': article_data.get('title', ''),
                            'content': article_data.get('content', ''),
                            'word_count': article_data.get('word_count', 0)
                        })
                    
                    # Collect insights
                    if analysis.get('key_topics'):
                        deep_read_results['key_insights'].extend([
                            {'url': url, 'topic': topic} 
                            for topic in analysis['key_topics']
                        ])
                    
                    # Collect citations
                    citation_analysis = analysis.get('citation_analysis', {})
                    if citation_analysis.get('citations'):
                        deep_read_results['citations_found'].extend([
                            {'url': url, 'citation': citation}
                            for citation in citation_analysis['citations']
                        ])
                        
                else:
                    error_msg = article_analysis.get('error', 'Unknown error') if article_analysis else 'No response'
                    logger.error(f"Failed to analyze article {url}: {error_msg}")
                    deep_read_results['articles_analyzed'].append({
                        'url': url, 
                        'error': error_msg,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                logger.error(f"Exception while deep reading article {url}: {e}")
                deep_read_results['articles_analyzed'].append({
                    'url': url, 
                    'error': str(e),
                    'timestamp': time.time()
                })

        # 4. Generate summary statistics
        successful_analyses = [a for a in deep_read_results['articles_analyzed'] if 'error' not in a]
        
        deep_read_results['summary'] = {
            'total_articles_attempted': len(selected_articles),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(selected_articles) - len(successful_analyses),
            'total_content_words': sum([
                content.get('word_count', 0) 
                for content in deep_read_results['content_extracted']
            ]),
            'unique_insights': len(set([
                insight['topic'] for insight in deep_read_results['key_insights']
            ])),
            'total_citations': len(deep_read_results['citations_found'])
        }
        
        logger.info(f"Deep reading phase completed: {deep_read_results['summary']}")
        
        return deep_read_results
    
    async def _execute_cross_reference_phase(self, browser, step: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cross-referencing phase for fact verification and source consistency analysis"""
        import logging
        logger = logging.getLogger(__name__)
        
        cross_ref_results = {
            'claims_verified': [],
            'source_consistency': {},
            'contradiction_analysis': [],
            'credibility_scores': {},
            'consensus_findings': [],
            'disputed_claims': []
        }
        
        # Extract claims from analyzed articles
        articles_analyzed = results.get('articles_analyzed', [])
        if not articles_analyzed:
            logger.warning("No articles available for cross-referencing")
            return cross_ref_results
        
        logger.info(f"Cross-referencing {len(articles_analyzed)} articles")
        
        # 1. Extract key claims from all articles
        all_claims = []
        source_claims = {}  # url -> list of claims
        
        for article in articles_analyzed:
            if 'error' in article:
                continue
                
            url = article.get('url', 'unknown')
            article_data = article.get('article_data', {})
            analysis = article.get('analysis', {})
            
            # Extract claims from content (simplified approach)
            content = article_data.get('content', '')
            claims = self._extract_claims_from_content(content, url)
            
            all_claims.extend(claims)
            source_claims[url] = claims
            
            # Calculate source credibility
            credibility = self._calculate_source_credibility(article_data, analysis)
            cross_ref_results['credibility_scores'][url] = credibility
        
        logger.info(f"Extracted {len(all_claims)} total claims from articles")
        
        # 2. Group similar claims for verification
        claim_groups = self._group_similar_claims(all_claims)
        
        verification_threshold = step.get('verification_threshold', 0.8)
        source_diversity_requirement = step.get('source_diversity_requirement', 3)
        
        # 3. Verify each claim group
        for group_id, claim_group in claim_groups.items():
            try:
                verification_result = await self._verify_claim_group(
                    claim_group, 
                    source_claims,
                    cross_ref_results['credibility_scores'],
                    verification_threshold,
                    source_diversity_requirement
                )
                
                cross_ref_results['claims_verified'].append(verification_result)
                
                # Categorize results
                if verification_result['verification_status'] == 'verified':
                    cross_ref_results['consensus_findings'].append(verification_result)
                elif verification_result['verification_status'] == 'disputed':
                    cross_ref_results['disputed_claims'].append(verification_result)
                    
            except Exception as e:
                logger.error(f"Error verifying claim group {group_id}: {e}")
        
        # 4. Analyze source consistency
        cross_ref_results['source_consistency'] = self._analyze_source_consistency(
            source_claims, 
            cross_ref_results['credibility_scores']
        )
        
        # 5. Detect contradictions between sources
        contradictions = self._detect_contradictions(all_claims, source_claims)
        cross_ref_results['contradiction_analysis'] = contradictions
        
        # 6. Generate verification summary
        verified_claims = [c for c in cross_ref_results['claims_verified'] if c['verification_status'] == 'verified']
        disputed_claims = [c for c in cross_ref_results['claims_verified'] if c['verification_status'] == 'disputed']
        
        cross_ref_results['summary'] = {
            'total_claims_analyzed': len(all_claims),
            'claim_groups_formed': len(claim_groups),
            'verified_claims': len(verified_claims),
            'disputed_claims': len(disputed_claims),
            'contradictions_found': len(contradictions),
            'high_credibility_sources': len([
                url for url, score in cross_ref_results['credibility_scores'].items() 
                if score > 0.7
            ]),
            'consensus_level': len(verified_claims) / max(1, len(cross_ref_results['claims_verified']))
        }
        
        logger.info(f"Cross-reference phase completed: {cross_ref_results['summary']}")
        
        return cross_ref_results

    def _extract_claims_from_content(self, content: str, source_url: str) -> List[Dict[str, Any]]:
        """Extract factual claims from article content"""
        import re
        
        claims = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Look for sentences that contain factual claim indicators
        claim_indicators = [
            r'\b(is|are|was|were)\b.*\b(percent|%|\d+)\b',  # Statistical claims
            r'\b(according to|research shows|study finds|data indicates)\b',  # Research citations
            r'\b(increased|decreased|rose|fell|grew|declined)\b.*\b(\d+|percent|%)\b',  # Trend claims
            r'\b(will|expect|predict|forecast)\b',  # Predictive claims
            r'\b(caused|leads to|results in|due to)\b',  # Causal claims
        ]
        
        for i, sentence in enumerate(sentences[:100]):  # Limit to first 100 sentences
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            for pattern in claim_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append({
                        'text': sentence,
                        'source_url': source_url,
                        'position': i,
                        'claim_type': 'factual',
                        'confidence': 0.7  # Default confidence
                    })
                    break
        
        return claims[:20]  # Limit to 20 claims per article

    def _group_similar_claims(self, claims: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group similar claims together for cross-verification"""
        import hashlib
        
        claim_groups = {}
        
        for claim in claims:
            # Simple grouping by key terms (can be enhanced with semantic similarity)
            claim_text = claim['text'].lower()
            
            # Extract key terms for grouping
            key_terms = set()
            words = claim_text.split()
            
            # Look for numbers, proper nouns, and important keywords
            for word in words:
                if (word.isdigit() or 
                    word.replace('%', '').replace(',', '').isdigit() or
                    word[0].isupper() or
                    word in ['increased', 'decreased', 'study', 'research', 'report']):
                    key_terms.add(word.lower())
            
            # Create group key based on key terms
            group_key = '_'.join(sorted(key_terms)[:3])  # Use top 3 key terms
            if not group_key:
                group_key = hashlib.md5(claim_text.encode()).hexdigest()[:8]
            
            if group_key not in claim_groups:
                claim_groups[group_key] = []
            claim_groups[group_key].append(claim)
        
        return claim_groups

    async def _verify_claim_group(
        self, 
        claim_group: List[Dict[str, Any]], 
        source_claims: Dict[str, List[Dict[str, Any]]],
        credibility_scores: Dict[str, float],
        verification_threshold: float,
        source_diversity_requirement: int
    ) -> Dict[str, Any]:
        """Verify a group of similar claims across multiple sources"""
        
        # Count sources supporting this claim
        supporting_sources = set()
        total_credibility = 0.0
        claim_texts = []
        
        for claim in claim_group:
            source_url = claim['source_url']
            supporting_sources.add(source_url)
            total_credibility += credibility_scores.get(source_url, 0.5)
            claim_texts.append(claim['text'])
        
        # Calculate verification metrics
        source_count = len(supporting_sources)
        avg_credibility = total_credibility / len(claim_group) if claim_group else 0
        diversity_met = source_count >= source_diversity_requirement
        
        # Determine verification status
        if diversity_met and avg_credibility >= verification_threshold:
            verification_status = 'verified'
        elif source_count > 1 and avg_credibility > 0.5:
            verification_status = 'likely'
        elif source_count == 1:
            verification_status = 'unverified'
        else:
            verification_status = 'disputed'
        
        return {
            'claim_group_id': hashlib.md5(str(claim_texts).encode()).hexdigest()[:8],
            'representative_claim': claim_group[0]['text'] if claim_group else '',
            'verification_status': verification_status,
            'supporting_sources': list(supporting_sources),
            'source_count': source_count,
            'average_credibility': avg_credibility,
            'diversity_requirement_met': diversity_met,
            'credibility_threshold_met': avg_credibility >= verification_threshold,
            'all_claims': claim_texts
        }

    def _calculate_source_credibility(self, article_data: Dict[str, Any], analysis: Dict[str, Any]) -> float:
        """Calculate credibility score for a source based on multiple factors"""
        credibility = 0.5  # Base credibility
        
        # Domain-based credibility
        url = article_data.get('url', '')
        if any(domain in url for domain in ['.edu', '.gov', 'arxiv.org', 'pubmed']):
            credibility += 0.3
        elif any(domain in url for domain in ['wikipedia.org', 'reuters.com', 'bbc.com', 'nature.com']):
            credibility += 0.2
        elif '.blog' in url or 'medium.com' in url:
            credibility -= 0.1
        
        # Content quality indicators
        source_quality = analysis.get('source_quality_indicators', {})
        if source_quality.get('has_author'):
            credibility += 0.1
        if source_quality.get('has_publication_date'):
            credibility += 0.05
        if source_quality.get('adequate_length'):
            credibility += 0.1
        
        # Citation quality
        citation_analysis = analysis.get('citation_analysis', {})
        if citation_analysis.get('has_references'):
            credibility += 0.1
        
        # Bias score (lower bias = higher credibility)
        bias_score = analysis.get('bias_indicators', {}).get('bias_score', 0.5)
        credibility -= bias_score * 0.2
        
        # Expertise indicators
        expertise_score = analysis.get('expertise_indicators', {}).get('overall_expertise_score', 0)
        credibility += expertise_score * 0.2
        
        return max(0.0, min(1.0, credibility))

    def _analyze_source_consistency(
        self, 
        source_claims: Dict[str, List[Dict[str, Any]]], 
        credibility_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze consistency between different sources"""
        
        # Initialize analysis structure
        consistency_analysis = {
            'highly_consistent_sources': [],
            'moderately_consistent_sources': [],
            'inconsistent_sources': [],
            'overall_consistency_score': 0.0
        }
        
        # Extract domains from URLs
        domains = []
        for url in source_claims.keys():
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if domain:
                domains.append(domain)
        
        unique_domains = set(domains)
        domain_counts = Counter(domains)
        
        # Categorize sources by consistency
        for url, claims in source_claims.items():
            if not claims:
                continue
                
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Calculate how well this source aligns with others
            alignment_score = 0.0
            comparisons = 0
            
            for other_url, other_claims in source_claims.items():
                if other_url == url:
                    continue
                    
                other_parsed = urlparse(other_url)
                other_domain = other_parsed.netloc
                
                # Compare claims between sources
                similarity = self._calculate_claim_similarity(claims, other_claims)
                if similarity > 0.6:  # Threshold for consistency
                    alignment_score += similarity
                    comparisons += 1
            
            if comparisons > 0:
                source_credibility = credibility_scores.get(url, 0.5)
                final_alignment = alignment_score / comparisons
                
                if final_alignment > 0.8 and source_credibility > 0.7:
                    consistency_analysis['highly_consistent_sources'].append(url)
                elif final_alignment > 0.6 and source_credibility > 0.5:
                    consistency_analysis['moderately_consistent_sources'].append(url)
                else:
                    consistency_analysis['inconsistent_sources'].append(url)
        
        # Calculate overall consistency score
        if consistency_analysis['highly_consistent_sources'] or consistency_analysis['moderately_consistent_sources']:
            consistency_analysis['overall_consistency_score'] = (
                len(consistency_analysis['highly_consistent_sources']) * 0.8 +
                len(consistency_analysis['moderately_consistent_sources']) * 0.6
            ) / max(1, len(source_claims))
        
        return consistency_analysis

    def _calculate_claim_similarity(self, claims1: List[Dict[str, Any]], claims2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between two sets of claims using TF-IDF vectorization"""
        if not claims1 or not claims2:
            return 0.0
        
        # Combine all claim texts
        all_texts = []
        for claim in claims1 + claims2:
            all_texts.append(claim['text'])
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between the two claim sets
        vec1 = tfidf_matrix[len(claims1):].sum(axis=0)
        vec2 = tfidf_matrix[:len(claims1)].sum(axis=0)
        
        # Handle case where vectors are empty
        if vec1.nnz == 0 or vec2.nnz == 0:
            return 0.0
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2.T)[0, 0]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _detect_contradictions(
        self, 
        all_claims: List[Dict[str, Any]], 
        source_claims: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect contradictions between claims from different sources"""
        contradictions = []
        
        # Look for opposing claims using predefined patterns
        contradiction_patterns = [
            (r'\bincreased\b', r'\bdecreased\b'),
            (r'\brose\b', r'\bfell\b'),
            (r'\bgrew\b', r'\bshrank\b'),
            (r'\bimproved\b', r'\bworsened\b'),
            (r'\bsuccessful\b', r'\bunsuccessful\b'),
            (r'\bsafe\b', r'\bunsafe\b'),
        ]
        
        claim_map = {}
        for claim in all_claims:
            source_url = claim['source_url']
            text = claim['text'].lower()
            claim_map.setdefault(source_url, []).append(text)
        
        for pattern1, pattern2 in contradiction_patterns:
            for source1, claims1 in claim_map.items():
                for source2, claims2 in claim_map.items():
                    if source1 != source2:
                        # Check if both patterns appear in respective sources
                        source1_matches = any(re.search(pattern1, claim) for claim in claims1)
                        source2_matches = any(re.search(pattern2, claim) for claim in claims2)
                        
                        if source1_matches and source2_matches:
                            contradictions.append({
                                'source1': source1,
                                'source2': source2,
                                'pattern1': pattern1,
                                'pattern2': pattern2,
                                'confidence': min(len(claims1), len(claims2)) / max(len(claims1), len(claims2))
                            })
        
        return contradictions[:10]  # Return up to 10 contradictions

    async def _deep_read_article(self, browser, url: str) -> Dict[str, Any]:
        """Perform deep reading and analysis of an article"""
        try:
            await browser.goto(url, wait_until='domcontentloaded')
            
            # Capture article screenshot
            screenshot_path = f'/home/ai/research/articles/{self._sanitize_filename(url)}.png'
            await browser.screenshot(path=screenshot_path)
            
            # Extract comprehensive article data
            article_data = await browser.evaluate(f"""
                () => {{
                    const article = {{}};
                    
                    // Extract main content
                    const contentElement = document.querySelector('article, .article-content, .post-content');
                    article.content = contentElement ? contentElement.innerText : '';
                    
                    // Extract metadata
                    article.title = document.title || '';
                    article.url = window.location.href;
                    
                    // Author information
                    const authorElements = document.querySelectorAll('[rel="author"], .author, .byline');
                    article.author = authorElements.length > 0 ? authorElements[0].innerText.trim() : '';
                    
                    // Publication date
                    const dateElements = document.querySelectorAll('time, .date, .published');
                    article.publish_date = dateElements.length > 0 ? dateElements[0].innerText.trim() : '';
                    
                    // Word count estimation
                    article.word_count = article.content.split(/\s+/).filter(Boolean).length;
                    
                    // Links and references
                    const linkElements = document.querySelectorAll('a[href]');
                    article.links = Array.from(linkElements).map(link => ({{url: link.href, text: link.innerText.trim()}}));
                    
                    return article;
                }}
            """)
            
            # Perform content analysis
            analysis = await self._analyze_article_content(article_data)
            
            return {
                'url': url,
                'article_data': article_data,
                'analysis': analysis,
                'screenshot': screenshot_path,
                'extraction_timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'extraction_timestamp': time.time()
            }

    async def _analyze_article_content(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive content analysis and insight generation"""
        content = article_data.get('content', '')
        
        if not content:
            return {
                'error': 'No content available for analysis',
                'readability_score': 0,
                'key_topics': [],
                'sentiment_analysis': {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0},
                'fact_density': 0,
                'source_quality_indicators': {},
                'citation_analysis': {'citation_count': 0, 'citation_quality': 'unknown'},
                'bias_indicators': {'bias_score': 0, 'indicators': []}
            }
        
        analysis = {
            'readability_score': self._calculate_readability(content),
            'key_topics': self._extract_key_topics(content),
            'sentiment_analysis': self._analyze_sentiment(content),
            'fact_density': self._calculate_fact_density(content),
            'source_quality_indicators': self._assess_source_quality(article_data),
            'citation_analysis': self._analyze_citations(article_data),
            'bias_indicators': self._detect_bias_indicators(content),
            'content_structure': self._analyze_content_structure(article_data),
            'information_density': self._calculate_information_density(content),
            'expertise_indicators': self._detect_expertise_indicators(content, article_data)
        }
        
        # Add overall quality score
        analysis['overall_quality_score'] = self._calculate_overall_quality(analysis)
        
        return analysis

    async def _synthesize_research_findings(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize research findings into coherent insights and conclusions"""
        import logging
        from collections import defaultdict, Counter
        
        logger = logging.getLogger(__name__)
        
        synthesis = {
            'primary_findings': [],
            'consensus_points': [],
            'contradictions': [],
            'confidence_levels': {},
            'source_diversity_analysis': {},
            'temporal_analysis': {},
            'credibility_weighted_conclusions': [],
            'knowledge_gaps': [],
            'evidence_strength': {}
        }
        
        # Extract data from results
        articles_analyzed = results.get('articles_analyzed', [])
        sources_found = results.get('sources_found', [])
        fact_check_results = results.get('fact_check_results', [])
        cross_reference_analysis = results.get('cross_reference_analysis', [])
        
        if not articles_analyzed:
            logger.warning("No articles analyzed for synthesis")
            return synthesis
        
        logger.info(f"Synthesizing findings from {len(articles_analyzed)} articles")
        
        # 1. Extract key findings from all articles
        all_findings = []
        source_credibility = {}
        
        for article in articles_analyzed:
            if 'error' in article:
                continue
                
            url = article.get('url', 'unknown')
            analysis = article.get('analysis', {})
            article_data = article.get('article_data', {})
            
            # Calculate source credibility
            credibility = self._calculate_source_credibility_for_synthesis(analysis, article_data)
            source_credibility[url] = credibility
            
            # Extract findings
            findings = self._extract_findings_from_article(article, credibility)
            all_findings.extend(findings)
        
        # 2. Group similar findings
        finding_groups = self._group_similar_findings(all_findings)
        
        # 3. Identify consensus points
        consensus_threshold = 0.6  # 60% of sources must agree
        for group_key, group_findings in finding_groups.items():
            if len(group_findings) >= 2:  # At least 2 sources
                consensus_analysis = self._analyze_consensus(group_findings, source_credibility)
                
                if consensus_analysis['consensus_strength'] >= consensus_threshold:
                    synthesis['consensus_points'].append({
                        'finding': consensus_analysis['representative_finding'],
                        'supporting_sources': len(group_findings),
                        'consensus_strength': consensus_analysis['consensus_strength'],
                        'weighted_credibility': consensus_analysis['weighted_credibility']
                    })
    
        # 4. Identify contradictions
        contradictions = self._identify_contradictions(finding_groups, source_credibility)
        synthesis['contradictions'] = contradictions
        
        # 5. Generate primary findings (high-confidence, well-supported)
        primary_findings = self._generate_primary_findings(
            finding_groups, 
            source_credibility, 
            synthesis['consensus_points']
        )
        synthesis['primary_findings'] = primary_findings
        
        # 6. Calculate confidence levels
        synthesis['confidence_levels'] = self._calculate_confidence_levels(
            synthesis['consensus_points'],
            synthesis['contradictions'],
            source_credibility
        )
        
        # 7. Analyze source diversity
        synthesis['source_diversity_analysis'] = self._analyze_source_diversity(
            articles_analyzed, 
            sources_found
        )
        
        # 8. Temporal analysis (if dates available)
        synthesis['temporal_analysis'] = self._perform_temporal_analysis(articles_analyzed)
        
        # 9. Generate credibility-weighted conclusions
        synthesis['credibility_weighted_conclusions'] = self._generate_weighted_conclusions(
            synthesis['primary_findings'],
            source_credibility
        )
        
        # 10. Identify knowledge gaps
        synthesis['knowledge_gaps'] = self._identify_knowledge_gaps(
            results.get('query', ''),
            synthesis['primary_findings'],
            synthesis['contradictions']
        )
        
        # 11. Assess evidence strength
        synthesis['evidence_strength'] = self._assess_evidence_strength(synthesis)
        
        # 12. Generate synthesis summary
        synthesis['summary'] = self._generate_synthesis_summary(synthesis)
        
        logger.info(f"Research synthesis completed: {synthesis['summary']}")
        
        return synthesis

    def _calculate_source_credibility_for_synthesis(self, analysis: Dict[str, Any], article_data: Dict[str, Any]) -> float:
        """Calculate source credibility for synthesis purposes"""
        credibility = 0.5  # Base credibility
        
        # Domain credibility
        url = article_data.get('url', '')
        if any(domain in url for domain in ['.edu', '.gov', 'arxiv.org', 'pubmed']):
            credibility += 0.3
        elif any(domain in url for domain in ['wikipedia.org', 'reuters.com', 'bbc.com', 'nature.com']):
            credibility += 0.2
        elif '.blog' in url or 'medium.com' in url:
            credibility -= 0.1
        
        # Content quality indicators
        source_quality = analysis.get('source_quality_indicators', {})
        if source_quality.get('has_author'):
            credibility += 0.1
        if source_quality.get('has_publication_date'):
            credibility += 0.05
        if source_quality.get('adequate_length'):
            credibility += 0.1
        
        # Citation quality
        citation_analysis = analysis.get('citation_analysis', {})
        if citation_analysis.get('has_references'):
            credibility += 0.1
        
        # Bias score (lower bias = higher credibility)
        bias_score = analysis.get('bias_indicators', {}).get('bias_score', 0.5)
        credibility -= bias_score * 0.2
        
        # Expertise indicators
        expertise_score = analysis.get('expertise_indicators', {}).get('overall_expertise_score', 0)
        credibility += expertise_score * 0.2
        
        return max(0.0, min(1.0, credibility))

    def _extract_findings_from_article(self, article: Dict[str, Any], credibility: float) -> List[Dict[str, Any]]:
        """Extract key findings from an individual article"""
        findings = []
        
        article_data = article.get('article_data', {})
        analysis = article.get('analysis', {})
        url = article.get('url', 'unknown')
        
        # Extract from key topics
        key_topics = analysis.get('key_topics', [])
        for topic in key_topics[:5]:  # Limit to top 5 topics
            findings.append({
                'type': 'topic',
                'content': topic,
                'source_url': url,
                'credibility': credibility,
                'evidence_type': 'topical'
            })
        
        # Extract from content analysis
        content = article_data.get('content', '')
        if content:
            # Look for conclusions or summary statements
            conclusion_findings = self._extract_conclusion_statements(content, url, credibility)
            findings.extend(conclusion_findings)
            
            # Extract quantitative claims
            quantitative_findings = self._extract_quantitative_claims(content, url, credibility)
            findings.extend(quantitative_findings)
        
        return findings

    def _extract_conclusion_statements(self, content: str, source_url: str, credibility: float) -> List[Dict[str, Any]]:
        """Extract conclusion-like statements from content"""
        import re
        
        findings = []
        
        # Patterns that often indicate conclusions or findings
        conclusion_patterns = [
            r'(the study shows|research shows|findings indicate|results suggest|we conclude|in conclusion)\s+([^.!?]+[.!?])',
            r'(this suggests|this indicates|this shows|evidence suggests)\s+([^.!?]+[.!?])',
            r'(therefore|thus|consequently|as a result)\s+([^.!?]+[.!?])',
            r'(the data shows|analysis reveals|our findings)\s+([^.!?]+[.!?])'
        ]
        
        for pattern in conclusion_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                finding_text = match.group(2).strip()
                if len(finding_text) > 20:  # Meaningful length
                    findings.append({
                        'type': 'conclusion',
                        'content': finding_text,
                        'source_url': source_url,
                        'credibility': credibility,
                        'evidence_type': 'conclusion',
                        'context': match.group(1)
                    })
        
        return findings[:5]  # Limit to 5 conclusion statements

    def _extract_quantitative_claims(self, content: str, source_url: str, credibility: float) -> List[Dict[str, Any]]:
        """Extract quantitative claims and statistics"""
        import re
        
        findings = []
        
        # Patterns for quantitative claims
        quant_patterns = [
            r'(\d+(?:\.\d+)?%)\s+([^.!?]{10,}[.!?])',  # Percentage claims
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s+(million|billion|thousand)\s+([^.!?]{10,}[.!?])',  # Large numbers
            r'(\d+)\s+(times|fold)\s+(more|less|higher|lower)\s+([^.!?]{10,}[.!?])'  # Comparisons
        ]
        
        for pattern in quant_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract the quantitative claim
                if len(match.groups()) >= 2:
                    quantity = match.group(1)
                    context = match.group(-1) if len(match.groups()) > 2 else match.group(2);
                    
                    findings.append({
                        'type': 'quantitative',
                        'content': context.strip(),
                        'quantity': quantity,
                        'source_url': source_url,
                        'credibility': credibility,
                        'evidence_type': 'quantitative'
                    })
        
        return findings[:3]  # Limit to 3 quantitative claims

    def _group_similar_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group similar findings together"""
        import hashlib
        from collections import defaultdict
        
        groups = defaultdict(list)
        
        for finding in findings:
            # Create grouping key based on content similarity
            content = finding.get('content', '').lower()
            
            # Extract key terms for grouping
            words = content.split()
            key_terms = []
            
            # Focus on important words (nouns, adjectives, numbers)
            for word in words:
                if (len(word) > 3 and 
                    not word in ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'were'] and
                    (word.isalpha() or word.replace('.', '').replace('%', '').isdigit())):
                    key_terms.append(word)
            
            # Create group key from top terms
            if key_terms:
                group_key = '_'.join(sorted(key_terms[:3]))
            else:
                group_key = hashlib.md5(content.encode()).hexdigest()[:8]
            
            groups[group_key].append(finding)
        
        return dict(groups)

    def _analyze_consensus(self, findings: List[Dict[str, Any]], source_credibility: Dict[str, float]) -> Dict[str, Any]:
        """Analyze consensus strength among similar findings"""
        
        if not findings:
            return {'consensus_strength': 0.0, 'weighted_credibility': 0.0, 'representative_finding': ''}
        
        # Calculate consensus metrics
        unique_sources = set(f.get('source_url', '') for f in findings)
        total_credibility = sum(source_credibility.get(f.get('source_url', ''), 0.5) for f in findings)
        weighted_credibility = total_credibility / len(findings)
        
        # Consensus strength based on source count and credibility
        source_diversity_score = min(1.0, len(unique_sources) / 3)  # Normalize to max of 3 sources
        consensus_strength = (source_diversity_score * 0.6) + (weighted_credibility * 0.4)
        
        # Select representative finding (highest credibility)
        representative = max(findings, key=lambda f: source_credibility.get(f.get('source_url', ''), 0.5))
        
        return {
            'consensus_strength': consensus_strength,
            'weighted_credibility': weighted_credibility,
            'representative_finding': representative.get('content', ''),
            'supporting_sources': list(unique_sources),
            'source_count': len(unique_sources)
        }

    def _identify_contradictions(self, finding_groups: Dict[str, List[Dict[str, Any]]], source_credibility: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify contradictions between different findings"""
        contradictions = []
        
        # Look for opposing claims within the same topic areas
        contradiction_patterns = [
            (['increase', 'rise', 'grow', 'improve'], ['decrease', 'fall', 'decline', 'worsen']),
            (['effective', 'successful', 'beneficial'], ['ineffective', 'unsuccessful', 'harmful']),
            (['safe', 'secure', 'stable'], ['dangerous', 'risky', 'unstable']),
            (['more', 'higher', 'greater'], ['less', 'lower', 'smaller'])
        ]
        
        group_items = list(finding_groups.items())
        
        for i, (group1_key, group1_findings) in enumerate(group_items):
            for j, (group2_key, group2_findings) in enumerate(group_items[i+1:], i+1):
                
                # Check if groups might be contradictory
                for positive_terms, negative_terms in contradiction_patterns:
                    group1_text = ' '.join([f.get('content', '') for f in group1_findings]).lower()
                    group2_text = ' '.join([f.get('content', '') for f in group2_findings]).lower()
                    
                    # Check for contradiction patterns
                    group1_positive = any(term in group1_text for term in positive_terms)
                    group1_negative = any(term in group1_text for term in negative_terms)
                    group2_positive = any(term in group2_text for term in positive_terms)
                    group2_negative = any(term in group2_text for term in negative_terms)
                    
                    if ((group1_positive and group2_negative) or (group1_negative and group2_positive)):
                        # Calculate contradiction strength
                        group1_credibility = sum(source_credibility.get(f.get('source_url', ''), 0.5) for f in group1_findings) / len(group1_findings)
                        group2_credibility = sum(source_credibility.get(f.get('source_url', ''), 0.5) for f in group2_findings) / len(group2_findings)
                        
                        contradictions.append({
                            'group1_finding': group1_findings[0].get('content', ''),
                            'group2_finding': group2_findings[0].get('content', ''),
                            'group1_sources': list(set(f.get('source_url', '') for f in group1_findings)),
                            'group2_sources': list(set(f.get('source_url', '') for f in group2_findings)),
                            'group1_credibility': group1_credibility,
                            'group2_credibility': group2_credibility,
                            'contradiction_type': f"{positive_terms[0]}_vs_{negative_terms[0]}",
                            'strength': abs(group1_credibility - group2_credibility)
                        })
                        break
    
        return contradictions[:5]  # Limit to 5 contradictions

    # MISSING METHOD IMPLEMENTATIONS (STUBS) - These methods are referenced but not implemented
    async def _analyze_research_query(self, query: str) -> Dict[str, Any]:
        """Analyze research query to determine optimal search strategy"""
        return {
            'strategy': 'comprehensive',
            'expected_source_count': 10,
            'recommended_engines': ['google', 'scholar', 'duckduckgo'],
            'optimized_terms': {
                'google': query,
                'scholar': query,
                'duckduckgo': query
            }
        }
    
    def _merge_step_results(self, results: Dict[str, Any], step_results: Dict[str, Any]) -> None:
        """Merge step results into the main results dictionary"""
        for key, value in step_results.items():
            if key in results and isinstance(results[key], list) and isinstance(value, list):
                results[key].extend(value)
            elif key not in results:
                results[key] = value

    async def _generate_research_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research insights from collected data"""
        return {
            'key_insights': [],
            'patterns_identified': [],
            'recommendations': [],
            'confidence_scores': {}
        }

    async def _finalize_research_session(self, browser_session, results: Dict[str, Any]) -> None:
        """Finalize the research session and clean up resources"""
        try:
            if browser_session:
                await browser_session.close()
        except Exception as e:
            logger.error(f"Error finalizing research session: {e}")

    def _sanitize_filename(self, url: str) -> str:
        """Sanitize URL for use as filename"""
        import re
        # Remove protocol and replace invalid characters
        sanitized = re.sub(r'[^\w\-_\.]', '_', url.replace('https://', '').replace('http://', ''))
        return sanitized[:50]  # Limit length

    def _calculate_relevance_score(self, source: Dict[str, Any], criteria: str, query: str) -> float:
        """Calculate relevance score for a source"""
        # Simple relevance scoring based on title and snippet similarity to query
        title = source.get('title', '').lower()
        snippet = source.get('snippet', '').lower()
        query_lower = query.lower()
        
        score = 0.0
        query_words = query_lower.split()
        
        for word in query_words:
            if word in title:
                score += 0.3
            if word in snippet:
                score += 0.2
        
        return min(1.0, score)

    async def _download_research_documents(self, browser, urls: List[str]) -> List[Dict[str, Any]]:
        """Download research documents from URLs"""
        return []  # Stub implementation

    # Content analysis method stubs
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score"""
        words = len(content.split())
        sentences = len(re.split(r'[.!?]+', content))
        if sentences == 0:
            return 0.0
        return max(0.0, min(1.0, (words / sentences) / 20.0))  # Simplified readability

    def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        words = content.lower().split()
        word_freq = Counter(words)
        # Filter out common words and get top 10 most frequent
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = {word: freq for word, freq in word_freq.items() if word not in common_words and len(word) > 3}
        return [word for word, freq in sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]]

    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of content"""
        # Simple sentiment analysis based on keyword presence
        positive_words = {'good', 'great', 'excellent', 'positive', 'beneficial', 'effective'}
        negative_words = {'bad', 'terrible', 'negative', 'harmful', 'ineffective', 'poor'}
        
        content_lower = content.lower()
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0}
        
        return {
            'positive': pos_count / total,
            'negative': neg_count / total,
            'neutral': 1 - (pos_count + neg_count) / len(content.split()),
            'compound': (pos_count - neg_count) / total
        }

    def _calculate_fact_density(self, content: str) -> float:
        """Calculate fact density in content"""
        fact_indicators = ['according to', 'research shows', 'study found', 'data indicates', 'percent', '%']
        content_lower = content.lower()
        fact_count = sum(1 for indicator in fact_indicators if indicator in content_lower)
        return min(1.0, fact_count / max(1, len(content.split()) / 100))

    def _assess_source_quality(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess source quality indicators"""
        return {
            'has_author': bool(article_data.get('author')),
            'has_publication_date': bool(article_data.get('publish_date')),
            'adequate_length': len(article_data.get('content', '').split()) > 100
        }

    def _analyze_citations(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze citations in article"""
        links = article_data.get('links', [])
        citation_count = len([link for link in links if any(domain in link.get('url', '') for domain in ['.edu', '.gov', 'pubmed', 'arxiv'])])
        return {
            'citation_count': citation_count,
            'has_references': citation_count > 0,
            'citations': links[:citation_count]
        }

    def _detect_bias_indicators(self, content: str) -> Dict[str, Any]:
        """Detect bias indicators in content"""
        bias_words = ['obviously', 'clearly', 'everyone knows', 'it is certain', 'without doubt']
        content_lower = content.lower()
        bias_count = sum(1 for word in bias_words if word in content_lower)
        
        return {
            'bias_score': min(1.0, bias_count / max(1, len(content.split()) / 100)),
            'indicators': [word for word in bias_words if word in content_lower]
        }

    def _analyze_content_structure(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content structure"""
        content = article_data.get('content', '')
        return {
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
            'word_count': len(content.split()),
            'average_sentence_length': len(content.split()) / max(1, len(re.split(r'[.!?]+', content)))
        }

    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density"""
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        return unique_words / max(1, total_words)

    def _detect_expertise_indicators(self, content: str, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect expertise indicators"""
        expertise_words = ['research', 'study', 'analysis', 'data', 'methodology', 'findings', 'conclusion']
        content_lower = content.lower()
        expertise_count = sum(1 for word in expertise_words if word in content_lower)
        
        return {
            'overall_expertise_score': min(1.0, expertise_count / max(1, len(content.split()) / 50)),
            'expertise_indicators': [word for word in expertise_words if word in content_lower]
        }

    def _calculate_overall_quality(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = [
            analysis.get('readability_score', 0) * 0.2,
            len(analysis.get('key_topics', [])) / 10 * 0.2,
            analysis.get('fact_density', 0) * 0.3,
            (1 - analysis.get('bias_indicators', {}).get('bias_score', 0)) * 0.3
        ]
        return sum(scores)

    # Additional synthesis helper methods (stubs)
    def _generate_primary_findings(self, finding_groups, source_credibility, consensus_points):
        """Generate primary findings from analysis"""
        return []

    def _calculate_confidence_levels(self, consensus_points, contradictions, source_credibility):
        """Calculate confidence levels for findings"""
        return {'overall_confidence': 0.7}

    def _analyze_source_diversity(self, articles_analyzed, sources_found):
        """Analyze diversity of sources"""
        return {'diversity_score': 0.8}

    def _perform_temporal_analysis(self, articles_analyzed):
        """Perform temporal analysis of articles"""
        return {'temporal_patterns': []}

    def _generate_weighted_conclusions(self, primary_findings, source_credibility):
        """Generate credibility-weighted conclusions"""
        return []

    def _identify_knowledge_gaps(self, query, primary_findings, contradictions):
        """Identify knowledge gaps in research"""
        return []

    def _assess_evidence_strength(self, synthesis):
        """Assess strength of evidence"""
        return {'strength_score': 0.7}

    def _generate_synthesis_summary(self, synthesis):
        """Generate synthesis summary"""
        return {
            'total_findings': len(synthesis.get('primary_findings', [])),
            'consensus_points': len(synthesis.get('consensus_points', [])),
            'contradictions': len(synthesis.get('contradictions', []))
        }