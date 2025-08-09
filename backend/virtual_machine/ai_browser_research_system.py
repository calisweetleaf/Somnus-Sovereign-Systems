"""
Somnus Sovereign Systems - Virtual Machine AI Browser Research System - 1 of 3 methods of web searching for the AI/LLM. One in the virtual machine, one system wide togglable, and one dedicated deep research subsystem platform.

A comprehensive browser-based AI research system with visual interaction,
intelligent content extraction, and automated research workflows.
"""
import logging
import asyncio
import json
import time
import re
import hashlib
from uuid import uuid4
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter, defaultdict

from backend.virtual_machine.ai_action_orchestrator import AIActionOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)


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
    credibility_met = avg_credibility >= verification_threshold
    
    # Determine verification status
    if diversity_met and credibility_met:
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
        'credibility_threshold_met': credibility_met,
        'all_claims': claim_texts
    }

def _calculate_source_credibility(self, article_data: Dict[str, Any], analysis: Dict[str, Any]) -> float:
    """Calculate credibility score for a source"""
    credibility = 0.5  # Base credibility
    
    # Domain-based credibility
    url = article_data.get('url', '')
    if any(domain in url for domain in ['.edu', '.gov', 'arxiv.org', 'pubmed']):
        credibility += 0.3
    elif any(domain in url for domain in ['wikipedia.org', 'reuters.com', 'bbc.com']):
        credibility += 0.2
    elif any(domain in url for domain in ['.blog', 'medium.com']):
        credibility -= 0.1
    
    # Content quality indicators
    if article_data.get('author'):
        credibility += 0.1
    if article_data.get('publish_date'):
        credibility += 0.1
    if article_data.get('word_count', 0) > 500:
        credibility += 0.1
    
    # Analysis-based factors
    if analysis.get('source_quality_indicators', {}).get('has_citations'):
        credibility += 0.1
    
    bias_score = analysis.get('bias_indicators', {}).get('bias_score', 0.5)
    credibility -= bias_score * 0.2  # Reduce credibility for high bias
    
    return max(0.0, min(1.0, credibility))

def _analyze_source_consistency(
    self, 
    source_claims: Dict[str, List[Dict[str, Any]]], 
    credibility_scores: Dict[str, float]
) -> Dict[str, Any]:
    """Analyze consistency between different sources"""
    
    consistency_analysis = {
        'highly_consistent_sources': [],
        'moderately_consistent_sources': [],
        'inconsistent_sources': [],
        'overall_consistency_score': 0.0
    }
    
    # Simple consistency analysis based on claim overlap
    source_urls = list(source_claims.keys())
    consistency_scores = {}
    
    for url in source_urls:
        if len(source_claims[url]) == 0:
            continue
            
        # Calculate how well this source aligns with others
        alignment_score = 0.0
        comparisons = 0
        
        for other_url in source_urls:
            if url == other_url or len(source_claims[other_url]) == 0:
                continue
                
            # Simple text similarity for claim alignment
            similarity = self._calculate_claim_similarity(
                source_claims[url], 
                source_claims[other_url]
            )
            alignment_score += similarity
            comparisons += 1
        
        if comparisons > 0:
            consistency_scores[url] = alignment_score / comparisons
        else:
            consistency_scores[url] = 0.5
    
    # Categorize sources by consistency
    for url, score in consistency_scores.items():
        credibility = credibility_scores.get(url, 0.5)
        
        if score > 0.7 and credibility > 0.6:
            consistency_analysis['highly_consistent_sources'].append({
                'url': url, 'consistency_score': score, 'credibility': credibility
            })
        elif score > 0.4:
            consistency_analysis['moderately_consistent_sources'].append({
                'url': url, 'consistency_score': score, 'credibility': credibility
            })
        else:
            consistency_analysis['inconsistent_sources'].append({
                'url': url, 'consistency_score': score, 'credibility': credibility
            })
    
    # Calculate overall consistency
    if consistency_scores:
        consistency_analysis['overall_consistency_score'] = sum(consistency_scores.values()) / len(consistency_scores)
    
    return consistency_analysis

def _calculate_claim_similarity(self, claims1: List[Dict[str, Any]], claims2: List[Dict[str, Any]]) -> float:
    """Calculate similarity between two sets of claims"""
    if not claims1 or not claims2:
        return 0.0
    
    # Simple keyword-based similarity
    keywords1 = set()
    keywords2 = set()
    
    for claim in claims1:
        words = claim['text'].lower().split()
        keywords1.update([w for w in words if len(w) > 3])
    
    for claim in claims2:
        words = claim['text'].lower().split()
        keywords2.update([w for w in words if len(w) > 3])
    
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = keywords1.intersection(keywords2)
    union = keywords1.union(keywords2)
    
    return len(intersection) / len(union) if union else 0.0

def _detect_contradictions(
    self, 
    all_claims: List[Dict[str, Any]], 
    source_claims: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Detect contradictions between claims from different sources"""
    contradictions = []
    
    # Look for opposing claims (simplified approach)
    contradiction_patterns = [
        (r'\bincreased\b', r'\bdecreased\b'),
        (r'\brose\b', r'\bfell\b'),
        (r'\bgrew\b', r'\bshrank\b'),
        (r'\bimproved\b', r'\bworsened\b'),
        (r'\beffective\b', r'\bineffective\b'),
        (r'\bsafe\b', r'\bunsafe\b'),
    ]
    
    for i, claim1 in enumerate(all_claims):
        for j, claim2 in enumerate(all_claims[i+1:], i+1):
            if claim1['source_url'] == claim2['source_url']:
                continue  # Skip claims from same source
            
            text1 = claim1['text'].lower()
            text2 = claim2['text'].lower()
            
            # Check for contradiction patterns
            for pattern1, pattern2 in contradiction_patterns:
                if (re.search(pattern1, text1) and re.search(pattern2, text2)) or \
                   (re.search(pattern2, text1) and re.search(pattern1, text2)):
                    
                    contradictions.append({
                        'claim1': claim1,
                        'claim2': claim2,
                        'contradiction_type': f"{pattern1}_vs_{pattern2}",
                        'confidence': 0.6
                    })
                    break
    
    return contradictions[:10]  # Limit to 10 contradictions
    
    async def _deep_read_article(self, browser, url: str) -> Dict[str, Any]:
        """Perform deep reading and analysis of individual article"""
        try:
            await browser.goto(url, wait_until='domcontentloaded')
            
            # Capture article screenshot
            screenshot_path = f'/home/ai/research/articles/{self._sanitize_filename(url)}.png'
            await browser.screenshot(path=screenshot_path)
            
            # Extract comprehensive article data
            article_data = await browser.evaluate("""
                () => {
                    // Intelligent content extraction with multiple fallback strategies
                    const contentSelectors = [
                        'article', '.article-content', '.post-content', '.entry-content',
                        'main', '[role="main"]', '.content', '#content', '.body'
                    ];
                    
                    let bestContent = '';
                    let bestScore = 0;
                    
                    for (const selector of contentSelectors) {{
                        const element = document.querySelector(selector);
                        if (element) {{
                            const text = element.textContent || '';
                            const score = text.length * (text.split(' ').length > 100 ? 1 : 0.5);
                            if (score > bestScore) {{
                                bestContent = text;
                                bestScore = score;
                            }}
                        }}
                    }}
                    
                    // Extract metadata and structural information
                    return {
                        title: document.title || '',
                        content: bestContent.trim(),
                        word_count: bestContent.split(' ').length,
                        meta_description: document.querySelector('meta[name="description"]')?.content || '',
                        author: document.querySelector('[rel="author"], .author, .byline')?.textContent?.trim() || '',
                        publish_date: document.querySelector('time, .date, .published')?.textContent?.trim() || '',
                        tags: Array.from(document.querySelectorAll('.tag, .tags a, .category')).map(el => el.textContent?.trim()),
                        images: Array.from(document.querySelectorAll('img')).map(img => ({
                            src: img.src,
                            alt: img.alt,
                            caption: img.closest('figure')?.querySelector('figcaption')?.textContent
                        })),
                        links: Array.from(document.querySelectorAll('a[href^="http"]')).map(a => ({
                            url: a.href,
                            text: a.textContent?.trim(),
                            is_external: !a.href.includes(window.location.hostname)
                        })),
                        headings: Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({
                            level: parseInt(h.tagName.substring(1)),
                            text: h.textContent?.trim()
                        }))
                    };
                }
            """)
            
            # Perform content analysis and insight extraction
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
            'citation_analysis': {'citation_count': 0, 'citations': []},
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

def _calculate_readability(self, content: str) -> float:
    """Calculate content readability score using Flesch Reading Ease formula"""
    import re
    
    # Clean content for analysis
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
    
    if not sentences:
        return 0.0
    
    words = content.split()
    if not words:
        return 0.0
    
    # Count syllables (simplified approach)
    def count_syllables(word):
        word = word.lower().strip('.,!?";')
        if not word:
            return 0
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    total_syllables = sum(count_syllables(word) for word in words)
    avg_sentence_length = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    
    # Flesch Reading Ease formula
    readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Normalize to 0-100 scale
    return max(0, min(100, readability))

def _extract_key_topics(self, content: str) -> List[str]:
    """Extract key topics from content using TF-IDF and keyword extraction"""
    import re
    from collections import Counter
    
    # Clean and tokenize content
    content_lower = content.lower()
    
    # Remove common words (stop words)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'cannot', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
        'their', 'he', 'she', 'his', 'her', 'him', 'we', 'us', 'our', 'you', 'your',
        'i', 'me', 'my', 'said', 'say', 'says', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
    }
    
    # Extract words (letters only, 3+ characters)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content_lower)
    
    # Filter out stop words and get word frequencies
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    
    # Get top topics by frequency
    top_words = [word for word, count in word_counts.most_common(15) if count > 1]
    
    # Extract noun phrases (simplified approach)
    # Look for capitalized sequences (potential proper nouns/topics)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
    
    # Clean and deduplicate proper nouns
    proper_nouns = list(set([
        noun.strip() for noun in proper_nouns 
        if len(noun) > 2 and noun.lower() not in stop_words
    ]))[:10]
    
    # Combine regular keywords and proper nouns
    key_topics = top_words + proper_nouns
    
    # Remove duplicates while preserving order
    seen = set()
    final_topics = []
    for topic in key_topics:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            final_topics.append(topic)
    
    return final_topics[:12]  # Return top 12 topics

def _analyze_sentiment(self, content: str) -> Dict[str, float]:
    """Analyze content sentiment using keyword-based approach"""
    
    # Positive sentiment words
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
        'positive', 'beneficial', 'effective', 'successful', 'improve', 'better', 'best',
        'advantage', 'benefit', 'progress', 'growth', 'increase', 'gain', 'win', 'success',
        'happy', 'pleased', 'satisfied', 'delighted', 'excited', 'optimistic', 'confident',
        'love', 'like', 'enjoy', 'appreciate', 'praise', 'celebrate', 'triumph', 'victory'
    }
    
    # Negative sentiment words
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'worst', 'fail',
        'negative', 'harmful', 'dangerous', 'ineffective', 'unsuccessful', 'worse', 'decline',
        'decrease', 'loss', 'problem', 'issue', 'concern', 'worry', 'fear', 'anxiety',
        'sad', 'angry', 'frustrated', 'disappointed', 'upset', 'concerned', 'worried',
        'hate', 'dislike', 'criticize', 'condemn', 'oppose', 'reject', 'defeat', 'failure'
    }
    
    # Tokenize content
    words = content.lower().split()
    
    positive_count = sum(1 for word in words if word.strip('.,!?":;') in positive_words)
    negative_count = sum(1 for word in words if word.strip('.,!?":;') in negative_words)
    total_sentiment_words = positive_count + negative_count
    
    if total_sentiment_words == 0:
        return {
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'compound': 0.0
        }
    
    positive_ratio = positive_count / total_sentiment_words
    negative_ratio = negative_count / total_sentiment_words
    neutral_ratio = 1.0 - (positive_ratio + negative_ratio)
    
    # Calculate compound score (range: -1 to 1)
    compound = (positive_count - negative_count) / max(1, len(words)) * 10
    compound = max(-1.0, min(1.0, compound))
    
    return {
        'positive': positive_ratio,
        'negative': negative_ratio,
        'neutral': max(0.0, neutral_ratio),
        'compound': compound
    }

def _calculate_fact_density(self, content: str) -> float:
    """Calculate density of factual statements in content"""
    import re
    
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return 0.0
    
    # Indicators of factual statements
    fact_patterns = [
        r'\b\d+([,\d]*\.?\d*)?%?\b',  # Numbers and percentages
        r'\b(according to|research shows|study found|data indicates|statistics show)\b',
        r'\b(is|are|was|were)\s+\w+\s+(percent|million|billion|thousand)\b',
        r'\b(increased|decreased|rose|fell|grew|declined)\s+by\s+\d+',
        r'\b(published|reported|announced|released)\s+(in|on|by)\b',
        r'\b\d{4}\b',  # Years
        r'\b(university|institute|organization|department|agency)\b.*\b(found|reported|stated)\b',
        r'\b(compared to|versus|relative to)\b'
    ]
    
    factual_sentences = 0
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check if sentence contains factual indicators
        for pattern in fact_patterns:
            if re.search(pattern, sentence_lower):
                factual_sentences += 1
                break
    
    return factual_sentences / len(sentences)

def _assess_source_quality(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess various quality indicators of the source"""
    
    quality_indicators = {
        'has_author': bool(article_data.get('author', '').strip()),
        'has_publication_date': bool(article_data.get('publish_date', '').strip()),
        'adequate_length': article_data.get('word_count', 0) >= 300,
        'has_headings': len(article_data.get('headings', [])) > 0,
        'has_images': len(article_data.get('images', [])) > 0,
        'has_external_links': len([
            link for link in article_data.get('links', []) 
            if link.get('is_external', False)
        ]) > 0,
        'has_tags': len(article_data.get('tags', [])) > 0,
        'proper_structure': len(article_data.get('headings', [])) >= 2
    }
    
    # Calculate overall quality score
    quality_score = sum(quality_indicators.values()) / len(quality_indicators)
    quality_indicators['overall_quality_score'] = quality_score
    
    # Additional metadata
    quality_indicators['word_count'] = article_data.get('word_count', 0)
    quality_indicators['heading_count'] = len(article_data.get('headings', []))
    quality_indicators['image_count'] = len(article_data.get('images', []))
    quality_indicators['external_link_count'] = len([
        link for link in article_data.get('links', []) 
        if link.get('is_external', False)
    ])
    
    return quality_indicators

def _analyze_citations(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze citations and references in the article"""
    content = article_data.get('content', '')
    links = article_data.get('links', [])
    
    # Look for citation patterns in text
    citation_patterns = [
        r'\([^)]*\d{4}[^)]*\)',  # (Author, 2024)
        r'\[\d+\]',  # [1], [2], etc.
        r'\b\w+\s+et\s+al\.\s*\(\d{4}\)',  # Smith et al. (2024)
        r'\b\w+\s*\(\d{4}\)',  # Smith (2024)
        r'doi:\s*[\w\./\-]+',  # DOI references
        r'arxiv:\s*[\w\./\-]+',  # ArXiv references
    ]
    
    citations_found = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        citations_found.extend(matches)
    
    # Analyze link quality for additional citations
    academic_domains = [
        'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
        'researchgate.net', 'jstor.org', 'springer.com', 'nature.com',
        'science.org', 'ieee.org', 'acm.org'
    ]
    
    academic_links = [
        link for link in links 
        if any(domain in link.get('url', '') for domain in academic_domains)
    ]
    
    return {
        'citation_count': len(citations_found),
        'citations': citations_found[:10],  # Limit to 10 examples
        'academic_link_count': len(academic_links),
        'academic_links': [link.get('url', '') for link in academic_links[:5]],
        'has_references': len(citations_found) > 0 or len(academic_links) > 0
    }

def _detect_bias_indicators(self, content: str) -> Dict[str, Any]:
    """Detect potential bias indicators in content"""
    
    # Strong opinion words that may indicate bias
    biased_language = {
        'extremist', 'radical', 'outrageous', 'absurd', 'ridiculous', 'pathetic',
        'brilliant', 'genius', 'perfect', 'flawless', 'devastating', 'shocking',
        'alarming', 'incredible', 'unbelievable', 'obviously', 'clearly',
        'undoubtedly', 'certainly', 'definitely', 'absolutely', 'never', 'always',
        'everyone', 'nobody', 'all', 'none', 'completely', 'totally', 'utterly'
    }
    
    # Loaded/emotional language
    loaded_language = {
        'freedom', 'liberty', 'tyranny', 'oppression', 'justice', 'injustice',
        'hero', 'villain', 'victim', 'perpetrator', 'crisis', 'disaster',
        'triumph', 'victory', 'defeat', 'failure', 'threat', 'danger',
        'propaganda', 'agenda', 'conspiracy', 'cover-up'
    }
    
    content_lower = content.lower()
    words = content_lower.split()
    
    # Count bias indicators
    bias_word_count = sum(1 for word in words if word.strip('.,!?":;') in biased_language)
    loaded_word_count = sum(1 for word in words if word.strip('.,!?":;') in loaded_language)
    
    # Calculate bias score
    total_bias_indicators = bias_word_count + loaded_word_count
    bias_score = min(1.0, total_bias_indicators / max(1, len(words)) * 100)
    
    # Collect specific indicators found
    indicators_found = []
    for word in set(words):
        clean_word = word.strip('.,!?":;')
        if clean_word in biased_language:
            indicators_found.append(f"Biased language: {clean_word}")
        elif clean_word in loaded_language:
            indicators_found.append(f"Loaded language: {clean_word}")
    
    return {
        'bias_score': bias_score,
        'indicators': indicators_found[:10],  # Limit to 10 examples
        'bias_word_count': bias_word_count,
        'loaded_word_count': loaded_word_count,
        'objectivity_score': max(0.0, 1.0 - bias_score)
    }

def _analyze_content_structure(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the structural quality of the content"""
    
    headings = article_data.get('headings', [])
    content = article_data.get('content', '')
    
    # Analyze heading structure
    heading_levels = [h.get('level', 1) for h in headings]
    has_proper_hierarchy = len(set(heading_levels)) > 1 if heading_levels else False
    
    # Analyze paragraph structure
    paragraphs = content.split('\n\n') if content else []
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / max(1, len(paragraphs))
    
    return {
        'heading_count': len(headings),
        'has_proper_heading_hierarchy': has_proper_hierarchy,
        'paragraph_count': len(paragraphs),
        'average_paragraph_length': avg_paragraph_length,
        'has_introduction': bool(paragraphs) and len(paragraphs[0].split()) > 30,
        'has_conclusion': bool(paragraphs) and len(paragraphs[-1].split()) > 30,
        'well_structured': len(headings) >= 2 and len(paragraphs) >= 3
    }

def _calculate_information_density(self, content: str) -> float:
    """Calculate how information-dense the content is"""
    
    if not content:
        return 0.0
    
    words = content.split()
    
    # Information-rich words (simplified approach)
    info_rich_patterns = [
        r'\b\d+([,\d]*\.?\d*)?\b',  # Numbers
        r'\b[A-Z][a-z]+\b',  # Proper nouns
        r'\b\w+ing\b',  # Gerunds (often actions/processes)
        r'\b\w+ed\b',   # Past participles (often results/states)
        r'\b\w{7,}\b'   # Long words (often technical terms)
    ]
    
    info_rich_count = 0
    for word in words:
        for pattern in info_rich_patterns:
            if re.match(pattern, word):
                info_rich_count += 1
                break
    
    return info_rich_count / max(1, len(words))

def _detect_expertise_indicators(self, content: str, article_data: Dict[str, Any]) -> Dict[str, Any]:
    """Detect indicators of author expertise and authority"""
    
    expertise_indicators = {
        'technical_terminology': 0,
        'specific_examples': 0,
        'methodological_language': 0,
        'quantitative_evidence': 0,
        'has_author_credentials': False
    }
    
    content_lower = content.lower()
    
    # Technical terminology (domain-specific)
    technical_terms = [
        'methodology', 'analysis', 'hypothesis', 'conclusion', 'evidence',
        'significant', 'correlation', 'variable', 'parameter', 'algorithm',
        'implementation', 'framework', 'protocol', 'specification'
    ]
    
    expertise_indicators['technical_terminology'] = sum(
        1 for term in technical_terms if term in content_lower
    )
    
    # Methodological language
    method_terms = [
        'we tested', 'we analyzed', 'we found', 'we observed', 'we measured',
        'the study', 'our research', 'the experiment', 'the survey', 'the data'
    ]
    
    expertise_indicators['methodological_language'] = sum(
        1 for term in method_terms if term in content_lower
    )
    
    # Quantitative evidence
    quant_patterns = [
        r'\b\d+%\b', r'\bp\s*<\s*0\.\d+\b', r'\bn\s*=\s*\d+\b',
        r'\bmean\s*=\s*\d+', r'\bstd\s*=\s*\d+'
    ]
    
    expertise_indicators['quantitative_evidence'] = sum(
        len(re.findall(pattern, content_lower)) for pattern in quant_patterns
    )
    
    # Specific examples (simplified detection)
    example_indicators = ['for example', 'for instance', 'such as', 'including']
    expertise_indicators['specific_examples'] = sum(
        1 for indicator in example_indicators if indicator in content_lower
    )
    
    # Author credentials (from metadata)
    author = article_data.get('author', '')
    credential_indicators = ['dr.', 'ph.d', 'professor', 'md', 'phd']
    expertise_indicators['has_author_credentials'] = any(
        cred in author.lower() for cred in credential_indicators
    )
    
    # Calculate overall expertise score
    expertise_score = (
        min(5, expertise_indicators['technical_terminology']) * 0.3 +
        min(5, expertise_indicators['methodological_language']) * 0.3 +
        min(5, expertise_indicators['quantitative_evidence']) * 0.2 +
        min(5, expertise_indicators['specific_examples']) * 0.1 +
        (1 if expertise_indicators['has_author_credentials'] else 0) * 0.1
    ) / 5
    
    expertise_indicators['overall_expertise_score'] = expertise_score
    
    return expertise_indicators

def _calculate_overall_quality(self, analysis: Dict[str, Any]) -> float:
    """Calculate overall content quality score"""
    
    # Weight different quality factors
    weights = {
        'readability_score': 0.15,
        'fact_density': 0.20,
        'source_quality': 0.20,
        'citation_quality': 0.15,
        'bias_score': 0.15,  # Lower bias = higher quality
        'expertise_score': 0.15
    }
    
    # Normalize scores to 0-1 range
    readability = analysis.get('readability_score', 0) / 100
    fact_density = analysis.get('fact_density', 0)
    source_quality = analysis.get('source_quality_indicators', {}).get('overall_quality_score', 0)
    citation_quality = min(1.0, analysis.get('citation_analysis', {}).get('citation_count', 0) / 10)
    bias_score = 1.0 - analysis.get('bias_indicators', {}).get('bias_score', 0)  # Invert bias score
    expertise_score = analysis.get('expertise_indicators', {}).get('overall_expertise_score', 0)
    
    # Calculate weighted average
    overall_quality = (
        readability * weights['readability_score'] +
        fact_density * weights['fact_density'] +
        source_quality * weights['source_quality'] +
        citation_quality * weights['citation_quality'] +
        bias_score * weights['bias_score'] +
        expertise_score * weights['expertise_score']
    )
    
    return round(overall_quality, 3)
    
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
        r'(\d+(?:\.\d+)?)\s+(times|fold)\s+(more|less|higher|lower)\s+([^.!?]{10,}[.!?])'  # Comparisons
    ]
    
    for pattern in quant_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            # Extract the quantitative claim
            if len(match.groups()) >= 2:
                quantity = match.group(1)
                context = match.group(-1) if len(match.groups()) > 2 else match.group(2)
                
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
    
    def _generate_primary_findings(self, finding_groups: Dict[str, List[Dict[str, Any]]], source_credibility: Dict[str, float], consensus_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate primary findings from the research"""
        primary_findings = []
        
        # Include all consensus points as primary findings
        for consensus in consensus_points:
            if consensus['consensus_strength'] > 0.7:  # High consensus threshold
                primary_findings.append({
                    'finding': consensus['finding'],
                    'confidence': consensus['consensus_strength'],
                    'support_level': 'strong_consensus',
                    'source_count': consensus['supporting_sources']
                })
        
        # Add high-credibility individual findings
        for group_key, findings in finding_groups.items():
            if len(findings) == 1:  # Single source findings
                finding = findings[0]
                source_cred = source_credibility.get(finding.get('source_url', ''), 0.5)
                
                if source_cred > 0.8:  # Very high credibility source
                    primary_findings.append({
                        'finding': finding.get('content', ''),
                        'confidence': source_cred,
                        'support_level': 'high_credibility_source',
                        'source_count': 1
                    })
        
        # Sort by confidence and limit
        primary_findings.sort(key=lambda x: x['confidence'], reverse=True)
        return primary_findings[:10]

    def _calculate_confidence_levels(self, consensus_points: List[Dict[str, Any]], contradictions: List[Dict[str, Any]], source_credibility: Dict[str, float]) -> Dict[str, float]:
        """Calculate overall confidence levels for the research"""
        
        if not source_credibility:
            return {'overall': 0.0, 'consensus': 0.0, 'source_quality': 0.0}
    
    # Overall source quality
    avg_source_credibility = sum(source_credibility.values()) / len(source_credibility)
    
    # Consensus strength
    if consensus_points:
        avg_consensus_strength = sum(cp['consensus_strength'] for cp in consensus_points) / len(consensus_points)
    else:
        avg_consensus_strength = 0.0
    
    # Contradiction impact (reduces confidence)
    contradiction_penalty = min(0.3, len(contradictions) * 0.1)
    
    # Calculate overall confidence
    overall_confidence = (avg_source_credibility * 0.4 + 
                         avg_consensus_strength * 0.4 + 
                         (1.0 - contradiction_penalty) * 0.2)
    
    return {
        'overall': round(overall_confidence, 3),
        'consensus': round(avg_consensus_strength, 3),
        'source_quality': round(avg_source_credibility, 3),
        'contradiction_impact': round(contradiction_penalty, 3)
    }

def _analyze_source_diversity(self, articles_analyzed: List[Dict[str, Any]], sources_found: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze diversity of sources used in research"""
    
    # Extract domains
    domains = []
    for article in articles_analyzed:
        if 'error' not in article:
            url = article.get('url', '')
            if url:
                domain = url.split('/')[2] if '://' in url else url.split('/')[0]
                domains.append(domain)
    
    # Analyze diversity
    unique_domains = set(domains)
    domain_counts = Counter(domains)
    
    # Categorize source types
    academic_domains = sum(1 for d in domains if any(suffix in d for suffix in ['.edu', '.gov', 'arxiv', 'pubmed']))
    news_domains = sum(1 for d in domains if any(suffix in d for suffix in ['.com', '.org', '.net']) and 'blog' not in d)
    
    return {
        'total_sources': len(articles_analyzed),
        'unique_domains': len(unique_domains),
        'domain_diversity_ratio': len(unique_domains) / max(1, len(domains)),
        'academic_sources': academic_domains,
        'news_sources': news_domains,
        'most_frequent_domain': domain_counts.most_common(1)[0] if domain_counts else None,
        'domain_distribution': dict(domain_counts.most_common(5))
    }

def _perform_temporal_analysis(self, articles_analyzed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform temporal analysis of sources if dates are available"""
    import re
    from datetime import datetime
    
    dates = []
    
    for article in articles_analyzed:
        if 'error' not in article:
            article_data = article.get('article_data', {})
            date_str = article_data.get('publish_date', '')
            
            if date_str:
                # Try to extract year from date string
                year_match = re.search(r'\b(20\d{2})\b', date_str)
                if year_match:
                    dates.append(int(year_match.group(1)))
    
    if not dates:
        return {'analysis': 'No temporal data available'}
    
    current_year = datetime.now().year
    recent_sources = sum(1 for d in dates if current_year - d <= 2)
    older_sources = len(dates) - recent_sources
    
    return {
        'date_range': f"{min(dates)}-{max(dates)}" if dates else "Unknown",
        'recent_sources': recent_sources,
        'older_sources': older_sources,
        'average_age': round(sum(current_year - d for d in dates) / len(dates), 1),
        'recency_score': recent_sources / len(dates)
    }

def _generate_weighted_conclusions(self, primary_findings: List[Dict[str, Any]], source_credibility: Dict[str, float]) -> List[Dict[str, Any]]:
    """Generate conclusions weighted by source credibility"""
    weighted_conclusions = []
    
    for finding in primary_findings:
        if finding['confidence'] > 0.6:  # Only high-confidence findings
            weighted_conclusions.append({
                'conclusion': finding['finding'],
                'confidence_level': finding['confidence'],
                'evidence_strength': finding['support_level'],
                'reliability': 'high' if finding['confidence'] > 0.8 else 'moderate'
            })
    
    return weighted_conclusions

def _identify_knowledge_gaps(self, query: str, primary_findings: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]) -> List[str]:
    """Identify gaps in knowledge based on research results"""
    gaps = []
    
    # If there are contradictions, that indicates knowledge gaps
    if contradictions:
        gaps.append("Conflicting evidence exists on key points, requiring further investigation")
    
    # If few primary findings, may indicate limited research
    if len(primary_findings) < 3:
        gaps.append("Limited conclusive evidence found, more comprehensive research needed")
    
    # Generic gaps based on common research areas
    query_lower = query.lower()
    
    if 'effect' in query_lower or 'impact' in query_lower:
        gaps.append("Long-term effects and broader impacts may require additional study")
    
    if 'cause' in query_lower:
        gaps.append("Causal mechanisms may need further investigation")
    
    if not gaps:
        gaps.append("Research appears comprehensive, but ongoing monitoring recommended")
    
    return gaps

def _assess_evidence_strength(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall strength of evidence"""
    
    primary_findings_count = len(synthesis.get('primary_findings', []))
    consensus_points_count = len(synthesis.get('consensus_points', []))
    contradictions_count = len(synthesis.get('contradictions', []))
    confidence = synthesis.get('confidence_levels', {}).get('overall', 0.0)
    
    # Calculate evidence strength score
    strength_score = (
        min(1.0, primary_findings_count / 5) * 0.3 +  # Primary findings (max 5)
        min(1.0, consensus_points_count / 3) * 0.3 +   # Consensus points (max 3)
        confidence * 0.3 +                             # Overall confidence
        max(0.0, 1.0 - contradictions_count / 5) * 0.1  # Contradiction penalty
    )
    
    # Categorize strength
    if strength_score > 0.8:
        strength_category = 'very_strong'
    elif strength_score > 0.6:
        strength_category = 'strong'
    elif strength_score > 0.4:
        strength_category = 'moderate'
    elif strength_score > 0.2:
        strength_category = 'weak'
    else:
        strength_category = 'very_weak'
    
    return {
        'strength_score': round(strength_score, 3),
        'strength_category': strength_category,
        'primary_findings_count': primary_findings_count,
        'consensus_points_count': consensus_points_count,
        'contradictions_count': contradictions_count
    }

def _generate_synthesis_summary(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the synthesis results"""
    
    return {
        'total_primary_findings': len(synthesis.get('primary_findings', [])),
        'consensus_points': len(synthesis.get('consensus_points', [])),
        'contradictions_found': len(synthesis.get('contradictions', [])),
        'overall_confidence': synthesis.get('confidence_levels', {}).get('overall', 0.0),
        'evidence_strength': synthesis.get('evidence_strength', {}).get('strength_category', 'unknown'),
        'knowledge_gaps_identified': len(synthesis.get('knowledge_gaps', [])),
        'source_diversity': synthesis.get('source_diversity_analysis', {}).get('domain_diversity_ratio', 0.0),
        'research_quality': self._categorize_research_quality(synthesis)
    }

def _categorize_research_quality(self, synthesis: Dict[str, Any]) -> str:
    """Categorize overall research quality"""
    
    confidence = synthesis.get('confidence_levels', {}).get('overall', 0.0)
    evidence_strength = synthesis.get('evidence_strength', {}).get('strength_score', 0.0)
    diversity = synthesis.get('source_diversity_analysis', {}).get('domain_diversity_ratio', 0.0)
    
    overall_score = (confidence + evidence_strength + diversity) / 3
    
    if overall_score > 0.8:
        return 'excellent'
    elif overall_score > 0.6:
        return 'good'
    elif overall_score > 0.4:
        return 'fair'
    else:
        return 'poor'
    
async def _generate_research_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate actionable insights from research results"""
    import logging
    
    logger = logging.getLogger(__name__)
    
    insights = {
        'key_insights': [],
        'implications': [],
        'recommendations': [],
        'knowledge_gaps': [],
        'further_research_directions': [],
        'confidence_assessment': {},
        'practical_applications': [],
        'risk_factors': [],
        'opportunity_areas': []
    }

    synthesis = results.get('synthesis', {})
    query = results.get('query', '')
    
    if not synthesis:
        logger.warning("No synthesis data available for insight generation")
        return insights

    logger.info("Generating research insights from synthesis")

    # 1. Extract key insights from primary findings
    primary_findings = synthesis.get('primary_findings', [])
    insights['key_insights'] = self._extract_key_insights(primary_findings, query)

    # 2. Generate implications from consensus points and contradictions
    insights['implications'] = self._generate_implications(synthesis)

    # 3. Develop practical recommendations
    insights['recommendations'] = self._generate_recommendations(synthesis, query)

    # 4. Identify knowledge gaps for future research
    insights['knowledge_gaps'] = self._identify_enhanced_knowledge_gaps(synthesis, query)

    # 5. Suggest further research directions
    insights['further_research_directions'] = self._suggest_research_directions(synthesis, query)

    # 6. Assess confidence and reliability
    insights['confidence_assessment'] = self._assess_insights_confidence(synthesis)

    # 7. Identify practical applications
    insights['practical_applications'] = self._identify_practical_applications(synthesis, query)

    # 8. Assess risk factors
    insights['risk_factors'] = self._identify_risk_factors(synthesis)

    # 9. Identify opportunity areas
    insights['opportunity_areas'] = self._identify_opportunities(synthesis, query)

    # 10. Generate insights summary
    insights['summary'] = self._generate_insights_summary(insights)

    logger.info(f"Generated {len(insights['key_insights'])} key insights and {len(insights['recommendations'])} recommendations")

    return insights

def _extract_key_insights(self, primary_findings: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Extract key insights from primary research findings"""
    
    key_insights = []
    
    for finding in primary_findings[:8]:  # Focus on top findings
        insight = {
            'insight': finding.get('finding', ''),
            'confidence': finding.get('confidence', 0.0),
            'support_level': finding.get('support_level', 'unknown'),
            'significance': self._assess_insight_significance(finding, query),
            'category': self._categorize_insight(finding.get('finding', ''))
        }
        
        # Add explanation of why this is significant
        insight['explanation'] = self._generate_insight_explanation(insight, query)
        
        key_insights.append(insight)
    
    # Sort by significance and confidence
    key_insights.sort(key=lambda x: (x['significance'], x['confidence']), reverse=True)
    
    return key_insights

def _generate_implications(self, synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate implications from research findings"""
    
    implications = []
    
    # Implications from consensus points
    consensus_points = synthesis.get('consensus_points', [])
    for consensus in consensus_points:
        if consensus['consensus_strength'] > 0.7:
            implications.append({
                'type': 'consensus_implication',
                'implication': f"Strong consensus indicates: {consensus['finding']}",
                'confidence': consensus['consensus_strength'],
                'basis': f"Supported by {consensus['supporting_sources']} sources",
                'impact_level': 'high' if consensus['consensus_strength'] > 0.8 else 'moderate'
            })
    
    # Implications from contradictions
    contradictions = synthesis.get('contradictions', [])
    for contradiction in contradictions[:3]:  # Limit to top 3
        implications.append({
            'type': 'contradiction_implication',
            'implication': f"Conflicting evidence exists regarding: {contradiction.get('contradiction_type', 'key findings')}",
            'confidence': contradiction.get('strength', 0.5),
            'basis': "Contradictory claims from different sources",
            'impact_level': 'moderate',
            'recommendation': "Further investigation required to resolve disagreement"
        })
    
    # Implications from evidence strength
    evidence_strength = synthesis.get('evidence_strength', {})
    strength_category = evidence_strength.get('strength_category', 'unknown')
    
    if strength_category in ['very_strong', 'strong']:
        implications.append({
            'type': 'evidence_implication',
            'implication': f"Research evidence is {strength_category}, supporting confident decision-making",
            'confidence': evidence_strength.get('strength_score', 0.0),
            'basis': f"Evidence strength score: {evidence_strength.get('strength_score', 0.0):.2f}",
            'impact_level': 'high'
        })
    elif strength_category in ['weak', 'very_weak']:
        implications.append({
            'type': 'evidence_implication',
            'implication': f"Research evidence is {strength_category}, requiring cautious interpretation",
            'confidence': evidence_strength.get('strength_score', 0.0),
            'basis': f"Limited evidence available",
            'impact_level': 'moderate',
            'recommendation': "Seek additional evidence before making major decisions"
        })
    
    return implications

def _generate_recommendations(self, synthesis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Generate actionable recommendations based on research findings"""
    
    recommendations = []
    
    # Recommendations based on primary findings
    primary_findings = synthesis.get('primary_findings', [])
    for finding in primary_findings[:5]:
        if finding.get('confidence', 0) > 0.7:
            recommendation = self._generate_finding_based_recommendation(finding, query)
            if recommendation:
                recommendations.append(recommendation)
    
    # Recommendations for handling contradictions
    contradictions = synthesis.get('contradictions', [])
    if contradictions:
        recommendations.append({
            'type': 'research_methodology',
            'recommendation': "Conduct additional research to resolve contradictory findings",
            'priority': 'high',
            'timeframe': 'short_term',
            'rationale': f"Found {len(contradictions)} significant contradictions in the literature",
            'specific_actions': [
                "Identify methodological differences between conflicting studies",
                "Seek recent meta-analyses or systematic reviews",
                "Consider consulting domain experts for interpretation"
            ]
        })
    
    # Recommendations based on knowledge gaps
    knowledge_gaps = synthesis.get('knowledge_gaps', [])
    if knowledge_gaps:
        recommendations.append({
            'type': 'knowledge_development',
            'recommendation': "Address identified knowledge gaps through targeted research",
            'priority': 'medium',
            'timeframe': 'medium_term',
            'rationale': "Several important knowledge gaps were identified",
            'specific_actions': [
                f"Focus research efforts on: {', '.join(knowledge_gaps[:3])}",
                "Collaborate with researchers in related fields",
                "Consider longitudinal studies for long-term effects"
            ]
        })
    
    # Recommendations based on source diversity
    source_diversity = synthesis.get('source_diversity_analysis', {})
    diversity_ratio = source_diversity.get('domain_diversity_ratio', 0)
    
    if diversity_ratio < 0.5:
        recommendations.append({
            'type': 'source_improvement',
            'recommendation': "Expand source diversity for more comprehensive understanding",
            'priority': 'medium',
            'timeframe': 'immediate',
            'rationale': f"Source diversity ratio: {diversity_ratio:.2f} indicates limited perspectives",
            'specific_actions': [
                "Include more academic and peer-reviewed sources",
                "Seek international perspectives",
                "Consider industry and practical viewpoints"
            ]
        })
    
    # Query-specific recommendations
    query_lower = query.lower()
    if 'implementation' in query_lower or 'practical' in query_lower:
        recommendations.append({
            'type': 'implementation',
            'recommendation': "Develop phased implementation plan based on evidence strength",
            'priority': 'high',
            'timeframe': 'immediate',
            'rationale': "Query focuses on practical implementation",
            'specific_actions': [
                "Start with high-confidence findings for initial implementation",
                "Plan pilot studies for moderate-confidence recommendations",
                "Monitor outcomes and adjust based on results"
            ]
        })
    
    return recommendations

def _identify_enhanced_knowledge_gaps(self, synthesis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Identify comprehensive knowledge gaps with detailed analysis"""
    
    gaps = []
    
    # Gaps from contradictions
    contradictions = synthesis.get('contradictions', [])
    for contradiction in contradictions:
        gaps.append({
            'gap_type': 'methodological_disagreement',
            'description': f"Conflicting findings on {contradiction.get('contradiction_type', 'key issue')}",
            'severity': 'high',
            'impact': 'Prevents confident decision-making',
            'research_needed': "Comparative methodology study or meta-analysis"
        })
    
    # Gaps from limited evidence
    evidence_strength = synthesis.get('evidence_strength', {})
    if evidence_strength.get('strength_category') in ['weak', 'very_weak']:
        gaps.append({
            'gap_type': 'insufficient_evidence',
            'description': "Limited research available on this topic",
            'severity': 'medium',
            'impact': 'Reduces confidence in conclusions',
            'research_needed': "Additional primary research studies"
        })
    
    # Temporal gaps
    temporal_analysis = synthesis.get('temporal_analysis', {})
    if temporal_analysis.get('recency_score', 0) < 0.5:
        gaps.append({
            'gap_type': 'outdated_research',
            'description': "Most available research is not recent",
            'severity': 'medium',
            'impact': 'May not reflect current state of knowledge',
            'research_needed': "Updated studies with current methodologies"
        })
    
    # Domain-specific gaps based on query
    query_lower = query.lower()
    
    if 'long-term' in query_lower or 'future' in query_lower:
        gaps.append({
            'gap_type': 'temporal_scope',
            'description': "Limited long-term outcome data",
            'severity': 'medium',
            'impact': 'Cannot assess sustained effects',
            'research_needed': "Longitudinal studies and follow-up research"
        })
    
    if 'cost' in query_lower or 'economic' in query_lower:
        gaps.append({
            'gap_type': 'economic_analysis',
            'description': "Insufficient economic impact analysis",
            'severity': 'medium',
            'impact': 'Cannot assess cost-effectiveness',
            'research_needed': "Health economics or cost-benefit studies"
        })
    
    return gaps

def _suggest_research_directions(self, synthesis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Suggest specific future research directions"""
    
    directions = []
    
    # Directions based on contradictions
    contradictions = synthesis.get('contradictions', [])
    if contradictions:
        directions.append({
            'direction': 'Contradiction Resolution Studies',
            'description': 'Systematic investigation of methodological differences causing contradictory results',
            'methodology': 'Meta-analysis with subgroup analysis',
            'priority': 'high',
            'timeline': '6-12 months',
            'required_resources': 'Statistical expertise, access to primary study data'
        })
    
    # Directions based on knowledge gaps
    knowledge_gaps = synthesis.get('knowledge_gaps', [])
    for gap in knowledge_gaps[:2]:
        if isinstance(gap, dict):
            directions.append({
                'direction': f'Gap Investigation: {gap.get("gap_type", "Unknown")}',
                'description': gap.get('description', ''),
                'methodology': gap.get('research_needed', 'Primary research study'),
                'priority': gap.get('severity', 'medium'),
                'timeline': '12-24 months',
                'required_resources': 'Research funding, institutional support'
            })
    
    # Methodology improvements
    source_diversity = synthesis.get('source_diversity_analysis', {})
    if source_diversity.get('domain_diversity_ratio', 0) < 0.6:
        directions.append({
            'direction': 'Multi-Perspective Analysis',
            'description': 'Research incorporating diverse geographical, cultural, and methodological perspectives',
            'methodology': 'Cross-cultural validation studies',
            'priority': 'medium',
            'timeline': '18-36 months',
            'required_resources': 'International collaboration, cultural expertise'
        })
    
    # Practical application research
    query_lower = query.lower()
    if 'practical' in query_lower or 'implementation' in query_lower:
        directions.append({
            'direction': 'Implementation Science Research',
            'description': 'Study of real-world implementation challenges and solutions',
            'methodology': 'Mixed-methods implementation studies',
            'priority': 'high',
            'timeline': '24-36 months',
            'required_resources': 'Implementation partners, outcome measurement tools'
        })
    
    return directions

def _assess_insights_confidence(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """Assess confidence in generated insights"""
    
    confidence_levels = synthesis.get('confidence_levels', {})
    evidence_strength = synthesis.get('evidence_strength', {})
    
    overall_confidence = confidence_levels.get('overall', 0.0)
    source_quality = confidence_levels.get('source_quality', 0.0)
    consensus_strength = confidence_levels.get('consensus', 0.0)
    
    # Calculate insight-specific confidence
    insight_confidence = (overall_confidence * 0.4 + 
                         source_quality * 0.3 + 
                         consensus_strength * 0.3)
    
    # Confidence categories
    if insight_confidence > 0.8:
        confidence_category = 'very_high'
        reliability_note = "Insights are well-supported by strong evidence"
    elif insight_confidence > 0.6:
        confidence_category = 'high'
        reliability_note = "Insights are supported by good evidence"
    elif insight_confidence > 0.4:
        confidence_category = 'moderate'
        reliability_note = "Insights have moderate support, use with caution"
    else:
        confidence_category = 'low'
        reliability_note = "Insights have limited support, require validation"
    
    return {
        'overall_confidence': round(insight_confidence, 3),
        'confidence_category': confidence_category,
        'reliability_note': reliability_note,
        'source_quality': round(source_quality, 3),
        'consensus_strength': round(consensus_strength, 3),
        'evidence_quality': evidence_strength.get('strength_category', 'unknown')
    }

def _identify_practical_applications(self, synthesis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Identify practical applications of research findings"""
    
    applications = []
    
    primary_findings = synthesis.get('primary_findings', [])
    
    for finding in primary_findings[:5]:
        if finding.get('confidence', 0) > 0.6:
            application = self._derive_practical_application(finding, query)
            if application:
                applications.append(application)
    
    # General applications based on query type
    query_lower = query.lower()
    
    if 'health' in query_lower or 'medical' in query_lower:
        applications.append({
            'domain': 'healthcare',
            'application': 'Clinical practice guideline development',
            'description': 'Use findings to inform evidence-based clinical guidelines',
            'implementation_level': 'institutional',
            'feasibility': 'high' if synthesis.get('confidence_levels', {}).get('overall', 0) > 0.7 else 'moderate'
        })
    
    if 'policy' in query_lower or 'government' in query_lower:
        applications.append({
            'domain': 'policy',
            'application': 'Policy recommendation development',
            'description': 'Inform policy decisions with research evidence',
            'implementation_level': 'governmental',
            'feasibility': 'moderate'
        })
    
    if 'business' in query_lower or 'management' in query_lower:
        applications.append({
            'domain': 'business',
            'application': 'Strategic planning and decision-making',
            'description': 'Incorporate insights into business strategy',
            'implementation_level': 'organizational',
            'feasibility': 'high'
        })
    
    return applications

def _identify_risk_factors(self, synthesis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify risk factors based on research findings"""
    
    risks = []
    
    # Risks from contradictory evidence
    contradictions = synthesis.get('contradictions', [])
    if contradictions:
        risks.append({
            'risk_type': 'decision_uncertainty',
            'description': 'Contradictory evidence may lead to poor decision-making',
            'severity': 'medium',
            'mitigation': 'Seek additional evidence or expert consultation',
            'probability': 'moderate'
        })
    
    # Risks from low confidence
    confidence = synthesis.get('confidence_levels', {}).get('overall', 0.0)
    if confidence < 0.5:
        risks.append({
            'risk_type': 'low_reliability',
            'description': 'Low confidence in findings increases risk of incorrect conclusions',
            'severity': 'high',
            'mitigation': 'Expand research scope or await additional evidence',
            'probability': 'high'
        })
    
    # Risks from limited source diversity
    diversity_ratio = synthesis.get('source_diversity_analysis', {}).get('domain_diversity_ratio', 0)
    if diversity_ratio < 0.4:
        risks.append({
            'risk_type': 'perspective_bias',
            'description': 'Limited source diversity may create biased conclusions',
            'severity': 'medium',
            'mitigation': 'Actively seek diverse perspectives and sources',
            'probability': 'moderate'
        })
    
    return risks

def _identify_opportunities(self, synthesis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """Identify opportunity areas based on research findings"""
    
    opportunities = []
    
    # Opportunities from strong consensus
    consensus_points = synthesis.get('consensus_points', [])
    high_consensus = [cp for cp in consensus_points if cp.get('consensus_strength', 0) > 0.8]
    
    if high_consensus:
        opportunities.append({
            'opportunity_type': 'confident_implementation',
            'description': 'Strong consensus enables confident implementation of findings',
            'potential_impact': 'high',
            'timeframe': 'immediate',
            'resource_requirements': 'moderate'
        })
    
    # Opportunities from knowledge gaps
    knowledge_gaps = synthesis.get('knowledge_gaps', [])
    if knowledge_gaps:
        opportunities.append({
            'opportunity_type': 'research_advancement',
            'description': 'Identified knowledge gaps present research opportunities',
            'potential_impact': 'medium',
            'timeframe': 'long_term',
            'resource_requirements': 'high'
        })
    
    # Query-specific opportunities
    query_lower = query.lower()
    
    if 'innovation' in query_lower or 'technology' in query_lower:
        opportunities.append({
            'opportunity_type': 'technological_development',
            'description': 'Research findings may inform technological innovations',
            'potential_impact': 'high',
            'timeframe': 'medium_term',
            'resource_requirements': 'high'
        })
    
    return opportunities

def _generate_insights_summary(self, insights: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary of insights analysis"""
    
    return {
        'total_insights': len(insights.get('key_insights', [])),
        'high_confidence_insights': len([
            i for i in insights.get('key_insights', []) 
            if i.get('confidence', 0) > 0.7
        ]),
        'actionable_recommendations': len(insights.get('recommendations', [])),
        'identified_gaps': len(insights.get('knowledge_gaps', [])),
        'research_directions': len(insights.get('further_research_directions', [])),
        'practical_applications': len(insights.get('practical_applications', [])),
        'risk_factors': len(insights.get('risk_factors', [])),
        'opportunities': len(insights.get('opportunity_areas', [])),
        'overall_usefulness': self._assess_overall_usefulness(insights)
    }

def _assess_insight_significance(self, finding: Dict[str, Any], query: str) -> float:
    """Assess significance of an insight relative to the query"""
    
    # Base significance from confidence
    significance = finding.get('confidence', 0.0)
    
    # Boost for direct relevance to query
    finding_text = finding.get('finding', '').lower()
    query_words = set(query.lower().split())
    finding_words = set(finding_text.split())
    
    relevance_overlap = len(query_words.intersection(finding_words)) / max(1, len(query_words))
    significance += relevance_overlap * 0.3
    
    # Boost for support level
    support_level = finding.get('support_level', '')
    if support_level == 'strong_consensus':
        significance += 0.2
    elif support_level == 'high_credibility_source':
        significance += 0.1
    
    return min(1.0, significance)

def _categorize_insight(self, finding_text: str) -> str:
    """Categorize the type of insight"""
    
    text_lower = finding_text.lower()
    
    if any(word in text_lower for word in ['cause', 'leads to', 'results in', 'due to']):
        return 'causal'
    elif any(word in text_lower for word in ['increase', 'decrease', 'change', 'trend']):
        return 'trend'
    elif any(word in text_lower for word in ['effective', 'successful', 'beneficial', 'harmful']):
        return 'evaluative'
    elif any(word in text_lower for word in ['correlation', 'associated', 'related']):
        return 'correlational'
    elif any(word in text_lower for word in ['recommend', 'suggest', 'should', 'ought']):
        return 'prescriptive'
    else:
        return 'descriptive'

def _generate_insight_explanation(self, insight: Dict[str, Any], query: str) -> str:
    """Generate explanation of why an insight is significant"""
    
    confidence = insight.get('confidence', 0.0)
    category = insight.get('category', 'descriptive')
    
    if confidence > 0.8:
        confidence_phrase = "strongly supported by evidence"
    elif confidence > 0.6:
        confidence_phrase = "well-supported by evidence"
    elif confidence > 0.4:
        confidence_phrase = "moderately supported"
    else:
        confidence_phrase = "has limited support"
    
    category_phrases = {
        'causal': 'establishes a cause-and-effect relationship',
        'trend': 'identifies an important trend or pattern',
        'evaluative': 'provides an assessment of effectiveness or value',
        'correlational': 'reveals important associations',
        'prescriptive': 'offers actionable guidance',
        'descriptive': 'provides important factual information'
    }
    
    explanation = f"This insight {category_phrases.get(category, 'provides valuable information')} and is {confidence_phrase}."
    
    return explanation

def _generate_finding_based_recommendation(self, finding: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Generate recommendation based on a specific finding"""
    
    finding_text = finding.get('finding', '')
    confidence = finding.get('confidence', 0.0)
    
    # Determine recommendation type based on finding content
    text_lower = finding_text.lower()
    
    if 'effective' in text_lower or 'beneficial' in text_lower:
        recommendation_type = 'adoption'
        action = "Consider adopting or implementing"
    elif 'harmful' in text_lower or 'risk' in text_lower:
        recommendation_type = 'avoidance'
        action = "Consider avoiding or mitigating"
    elif 'increase' in text_lower or 'improve' in text_lower:
        recommendation_type = 'enhancement'
        action = "Focus on enhancing or increasing"
    else:
        recommendation_type = 'investigation'
        action = "Further investigate"
    
    priority = 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
    
    return {
        'type': recommendation_type,
        'recommendation': f"{action} based on finding: {finding_text[:100]}...",
        'priority': priority,
        'timeframe': 'immediate' if priority == 'high' else 'short_term',
        'confidence_basis': confidence,
        'rationale': f"Recommendation based on {finding.get('support_level', 'available')} evidence"
    }

def _derive_practical_application(self, finding: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Derive practical application from a finding"""
    
    finding_text = finding.get('finding', '').lower()
    
    # Determine application domain
    if any(word in query.lower() for word in ['health', 'medical', 'clinical']):
        domain = 'healthcare'
        application_type = 'clinical_practice'
    elif any(word in query.lower() for word in ['business', 'management', 'organization']):
        domain = 'business'
        application_type = 'strategic_planning'
    elif any(word in query.lower() for word in ['education', 'learning', 'teaching']):
        domain = 'education'
        application_type = 'pedagogical_improvement'
    else:
        domain = 'general'
        application_type = 'decision_support'
    
    return {
        'domain': domain,
        'application': f"Apply finding to {application_type}",
        'description': f"Use evidence from: {finding_text[:150]}...",
        'implementation_level': 'organizational',
        'feasibility': 'high' if finding.get('confidence', 0) > 0.7 else 'moderate'
    }

def _assess_overall_usefulness(self, insights: Dict[str, Any]) -> str:
    """Assess overall usefulness of generated insights"""
    
    insight_count = len(insights.get('key_insights', []))
    high_conf_count = len([i for i in insights.get('key_insights', []) if i.get('confidence', 0) > 0.7])
    recommendation_count = len(insights.get('recommendations', []))
    
    usefulness_score = (
        min(1.0, insight_count / 5) * 0.4 +
        min(1.0, high_conf_count / 3) * 0.4 +
        min(1.0, recommendation_count / 3) * 0.2
    )
    
    if usefulness_score > 0.8:
        return 'very_useful'
    elif usefulness_score > 0.6:
        return 'useful'
    elif usefulness_score > 0.4:
        return 'moderately_useful'
    else:
        return 'limited_usefulness'
    
    async def _download_research_documents(self, browser, document_urls: List[str]) -> List[Dict[str, Any]]:
        """Download and process research documents"""
        downloaded_docs = []
        
        for url in document_urls:
            try:
                # Navigate to document
                await browser.goto(url)
                
                # Attempt download
                async with browser.expect_download() as download_info:
                    await browser.click('a[href$=".pdf"], .download-link, [data-download]')
                
                download = await download_info.value
                download_path = f'/home/ai/research/documents/{download.suggested_filename}'
                await download.save_as(download_path)
                
                downloaded_docs.append({
                    'url': url,
                    'local_path': download_path,
                    'filename': download.suggested_filename,
                    'download_timestamp': time.time()
                })
                
            except Exception as e:
                downloaded_docs.append({
                    'url': url,
                    'error': str(e),
                    'download_timestamp': time.time()
                })
        
        return downloaded_docs
    
    async def install_research_extension(self, extension_name: str) -> bool:
        """Install browser extension for enhanced research capabilities"""
        extensions = {
            'hypothesis': {
                'id': 'bjfhmglciegochdpefhhlphglcehbmek',
                'features': ['web_annotation', 'collaborative_notes']
            },
            'zotero': {
                'id': 'ekhagklcjbdpajgpjgmbionohlpdbjgc', 
                'features': ['citation_management', 'pdf_extraction']
            },
            'research_assistant': {
                'path': '/home/ai/extensions/research_assistant',
                'features': ['auto_fact_check', 'source_credibility', 'bias_detection']
            }
        }
        
        if extension_name not in extensions:
            return False
        
        extension_info = extensions[extension_name]
        
        try:
            if 'id' in extension_info:
                # Install from Chrome Web Store
                install_command = f"""
                mkdir -p /home/ai/.config/chromium/Default/Extensions/{extension_info['id']}
                # Additional installation logic would go here
                """
            else:
                # Load local extension
                install_command = f"""
                ln -sf {extension_info['path']} /home/ai/.config/chromium/Default/Extensions/
                """
            
            success = await self.vm.execute_command(install_command)
            
            if success:
                if not hasattr(self.vm, 'installed_tools'):
                    self.vm.installed_tools = []
                self.vm.installed_tools.append(f'browser_extension_{extension_name}')
                
                # Log the extension installation event on the host
                self.orchestrator.log_event(
                    event_type="extension_installed",
                    content=f"Browser research extension installed: {extension_name}",
                    metadata={
                        "extension_name": extension_name,
                        "features": extension_info.get("features", [])
                    }
                )
                return True
                
        except Exception as e:
            print(f"Failed to install extension {extension_name}: {e}")
        
        return False
    
    async def create_research_automation(self, workflow_name: str, workflow_config: Dict[str, Any]) -> str:
        """Create custom research automation script"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Custom research automation: {workflow_name}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

import asyncio
import json
from ai_browser_research_system import AIBrowserResearch

class {workflow_name.title().replace('_', '')}Automation:
    def __init__(self, vm_instance):
        self.research_system = AIBrowserResearch(vm_instance)
        self.config = {json.dumps(workflow_config, indent=8)}
    
    async def execute(self, query: str) -> dict:
        """Execute automated research workflow"""
        return await self.research_system.conduct_visual_research(
            query, 
            workflow_type='{workflow_name}'
        )

async def main():
    # Automation execution logic
    automation = {workflow_name.title().replace('_', '')}Automation(vm_instance)
    result = await automation.execute(input("Research query: "))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        script_path = f'/home/ai/tools/{workflow_name}_automation.py'
        
        await self.vm.write_file(script_path, script_content)
        await self.vm.execute_command(f'chmod +x {script_path}')
        
        # Register automation capability
        if not hasattr(self.vm, 'installed_tools'):
            self.vm.installed_tools = []
        self.vm.installed_tools.append(f'browser_extension_{workflow_name}')

        # Create an artifact for the automation script on the host
        await self.orchestrator.create_and_setup_artifact(
            name=f"Research Automation: {workflow_name}",
            content=script_content,
            artifact_type="text/python", # Assuming Python script
            description=f"Automated research workflow script: {workflow_name}",
            execution_environment="unlimited", # Automation scripts might need unlimited power
            enabled_capabilities=["unlimited_power", "code_execution"]
        )
        
        return script_path
    
    async def _finalize_research_session(self, browser, results: Dict[str, Any]):
        """Cleanup and finalize research session with comprehensive artifact creation"""
        import logging
        import json
        
        logger = logging.getLogger(__name__)
        
        try:
            # 1. Close browser gracefully
            if browser:
                try:
                    await browser.close()
                    logger.info("Browser session closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
            
            # 2. Generate comprehensive session metadata
            session_metadata = {
                'session_info': {
                    'timestamp': time.time(),
                    'session_id': str(uuid4()),
                    'query': results.get('query', ''),
                    'workflow_type': results.get('workflow_type', 'comprehensive'),
                    'completion_time': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                },
                'research_metrics': {
                    'sources_found': len(results.get('sources_found', [])),
                    'articles_analyzed': len([a for a in results.get('articles_analyzed', []) if 'error' not in a]),
                    'failed_articles': len([a for a in results.get('articles_analyzed', []) if 'error' in a]),
                    'screenshots_taken': len(results.get('screenshots', [])),
                    'documents_downloaded': len(results.get('downloaded_documents', [])),
                    'research_notes_created': len(results.get('research_notes', [])),
                    'fact_checks_performed': len(results.get('fact_check_results', [])),
                    'cross_references_analyzed': len(results.get('cross_reference_analysis', []))
                },
                'quality_metrics': self._calculate_session_quality_metrics(results),
                'synthesis_summary': results.get('synthesis', {}).get('summary', {}),
                'insights_summary': results.get('insights', {}).get('summary', {}),
                'research_scope': self._analyze_research_scope(results),
                'session_outcome': self._assess_session_outcome(results)
            }
            
            # 3. Save session metadata to VM
            session_timestamp = int(time.time())
            metadata_filename = f'research_session_{session_timestamp}_metadata.json'
            metadata_path = f'/home/ai/research/sessions/{metadata_filename}'
            
            try:
                # Ensure directory exists
                await self.vm.execute_command('mkdir -p /home/ai/research/sessions')
                
                # Write metadata file
                metadata_json = json.dumps(session_metadata, indent=2)
                await self.vm.write_file(metadata_path, metadata_json)
                
                logger.info(f"Session metadata saved to: {metadata_path}")
                
            except Exception as e:
                logger.error(f"Failed to save session metadata: {e}")
            
            # 4. Create research report artifact on host
            research_report = self._generate_research_report(results, session_metadata)
        
            try:
                await self.orchestrator.create_and_setup_artifact(
                    name=f"Research Report: {results.get('query', 'Untitled')[:50]}",
                    content=research_report,
                    artifact_type="text/markdown",
                    description=f"Comprehensive AI research report for query: {results.get('query', '')}",
                    execution_environment="documentation",
                    enabled_capabilities=["unlimited_power"]
                )
                
                logger.info("Research report artifact created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create research report artifact: {e}")
        
            # 5. Create session data artifact (JSON)
            try:
                session_data_json = json.dumps(results, indent=2, default=str)
                
                await self.orchestrator.create_and_setup_artifact(
                    name=f"Research Data: {results.get('query', 'Untitled')[:50]}",
                    content=session_data_json,
                    artifact_type="application/json",
                    description="Raw research session data and analysis results",
                    execution_environment="data_processing",
                    enabled_capabilities=["data_analysis"]
                )
                
                logger.info("Research data artifact created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create research data artifact: {e}")
        
            # 6. Update AI's research database
            await self._update_research_database(results, session_metadata)
        
            # 7. Generate and save automation script if patterns detected
            automation_script = await self._generate_automation_script_if_needed(results, session_metadata)
            if automation_script:
                try:
                    script_path = f'/home/ai/tools/auto_research_{session_timestamp}.py'
                    await self.vm.write_file(script_path, automation_script)
                    await self.vm.execute_command(f'chmod +x {script_path}')
                    
                    # Add to installed tools
                    if not hasattr(self.vm, 'installed_tools'):
                        self.vm.installed_tools = []
                    self.vm.installed_tools.append(f'auto_research_{session_timestamp}')
                    
                    logger.info(f"Automation script generated: {script_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate automation script: {e}")
        
            # 8. Log session completion event
            try:
                await self.orchestrator.log_event(
                    event_type="research_session_completed",
                    content=f"AI research session completed: {results.get('query', '')}",
                    metadata={
                        "session_timestamp": session_timestamp,
                        "query": results.get('query', ''),
                        "sources_analyzed": session_metadata['research_metrics']['sources_found'],
                        "quality_score": session_metadata['quality_metrics']['overall_quality_score'],
                        "outcome": session_metadata['session_outcome']
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to log session completion event: {e}")
        
            # 9. Cleanup temporary files
            await self._cleanup_session_temp_files(session_timestamp)
        
            logger.info(f"Research session finalized successfully: {session_metadata['session_info']['session_id']}")
        
        except Exception as e:
            logger.error(f"Error during research session finalization: {e}")
            # Ensure browser is closed even if other cleanup fails
            try:
                if browser:
                    await browser.close()
            except:
                pass

def _calculate_session_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate quality metrics for the research session"""
    
    # Source quality metrics
    articles_analyzed = results.get('articles_analyzed', [])
    successful_articles = [a for a in articles_analyzed if 'error' not in a]
    
    if not articles_analyzed:
        return {'overall_quality_score': 0.0, 'note': 'No articles analyzed'}
    
    success_rate = len(successful_articles) / len(articles_analyzed)
    
    # Content quality assessment
    total_content_quality = 0.0
    content_items = 0
    
    for article in successful_articles:
        analysis = article.get('analysis', {})
        if 'overall_quality_score' in analysis:
            total_content_quality += analysis['overall_quality_score']
            content_items += 1
    
    avg_content_quality = total_content_quality / max(1, content_items)
    
    # Synthesis quality
    synthesis = results.get('synthesis', {})
    confidence_level = synthesis.get('confidence_levels', {}).get('overall', 0.0)
    evidence_strength = synthesis.get('evidence_strength', {}).get('strength_score', 0.0)
    
    # Source diversity
    source_diversity = synthesis.get('source_diversity_analysis', {}).get('domain_diversity_ratio', 0.0)
    
    # Calculate overall quality score
    overall_quality = (
        success_rate * 0.25 +           # Data collection success
        avg_content_quality * 0.25 +    # Content quality
        confidence_level * 0.25 +       # Analysis confidence
        evidence_strength * 0.15 +      # Evidence strength
        source_diversity * 0.10         # Source diversity
    )
    
    return {
        'overall_quality_score': round(overall_quality, 3),
        'success_rate': round(success_rate, 3),
        'avg_content_quality': round(avg_content_quality, 3),
        'confidence_level': round(confidence_level, 3),
        'evidence_strength': round(evidence_strength, 3),
        'source_diversity': round(source_diversity, 3),
        'quality_category': self._categorize_quality_score(overall_quality)
    }

def _analyze_research_scope(self, results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the scope and coverage of the research session"""
    
    query = results.get('query', '')
    articles_analyzed = results.get('articles_analyzed', [])
    
    # Temporal scope
    temporal_analysis = results.get('synthesis', {}).get('temporal_analysis', {})
    
    # Topic coverage
    all_topics = []
    for article in articles_analyzed:
        if 'error' not in article:
            analysis = article.get('analysis', {})
            topics = analysis.get('key_topics', [])
            all_topics.extend(topics)
    
    unique_topics = list(set(all_topics))
    
    # Geographic scope (simplified detection)
    geographic_indicators = ['global', 'international', 'worldwide', 'national', 'local', 'regional']
    has_geographic_scope = any(indicator in query.lower() for indicator in geographic_indicators)
    
    # Methodological scope
    methodology_types = []
    for article in articles_analyzed:
        if 'error' not in article:
            content = article.get('article_data', {}).get('content', '').lower()
            if 'experiment' in content or 'trial' in content:
                methodology_types.append('experimental')
            elif 'survey' in content or 'questionnaire' in content:
                methodology_types.append('survey')
            elif 'interview' in content or 'qualitative' in content:
                methodology_types.append('qualitative')
            elif 'meta-analysis' in content or 'systematic review' in content:
                methodology_types.append('review')
            elif 'observational' in content or 'cohort' in content:
                methodology_types.append('observational')
    
    return {
        'query_complexity': len(query.split()),
        'topic_coverage': len(unique_topics),
        'unique_topics': unique_topics[:10],  # Top 10 topics
        'temporal_scope': temporal_analysis.get('date_range', 'Unknown'),
        'has_geographic_scope': has_geographic_scope,
        'methodology_diversity': list(set(methodology_types)),
        'scope_rating': self._rate_research_scope(len(unique_topics), len(set(methodology_types)), has_geographic_scope)
    }

def _assess_session_outcome(self, results: Dict[str, Any]) -> str:
    """Assess the overall outcome of the research session"""
    
    # Check for successful data collection
    articles_analyzed = results.get('articles_analyzed', [])
    successful_articles = [a for a in articles_analyzed if 'error' not in a]
    
    if not articles_analyzed:
        return 'failed_no_data'
    
    success_rate = len(successful_articles) / len(articles_analyzed)
    
    # Check synthesis quality
    synthesis = results.get('synthesis', {})
    confidence = synthesis.get('confidence_levels', {}).get('overall', 0.0)
    primary_findings = len(synthesis.get('primary_findings', []))
    
    # Check insights quality
    insights = results.get('insights', {})
    actionable_recommendations = len(insights.get('recommendations', []))
    
    # Determine outcome
    if success_rate > 0.8 and confidence > 0.7 and primary_findings >= 3:
        return 'excellent'
    elif success_rate > 0.6 and confidence > 0.5 and primary_findings >= 2:
        return 'good'
    elif success_rate > 0.4 and confidence > 0.3 and primary_findings >= 1:
        return 'acceptable'
    elif success_rate > 0.2:
        return 'limited_success'
    else:
        return 'poor'

def _generate_research_report(self, results: Dict[str, Any], session_metadata: Dict[str, Any]) -> str:
    """Generate a comprehensive research report in markdown format"""
    
    query = results.get('query', 'Research Query')
    synthesis = results.get('synthesis', {})
    insights = results.get('insights', {})
    
    report = f"""# AI Research Report: {query}

## Executive Summary

**Research Query:** {query}  
**Session Date:** {session_metadata['session_info']['completion_time']}  
**Overall Quality Score:** {session_metadata['quality_metrics']['overall_quality_score']:.2f}/1.0  
**Session Outcome:** {session_metadata['session_outcome'].replace('_', ' ').title()}

## Research Metrics

- **Sources Found:** {session_metadata['research_metrics']['sources_found']}
- **Articles Successfully Analyzed:** {session_metadata['research_metrics']['articles_analyzed']}
- **Documents Downloaded:** {session_metadata['research_metrics']['documents_downloaded']}
- **Fact Checks Performed:** {session_metadata['research_metrics']['fact_checks_performed']}

## Key Findings

"""
    
    # Add primary findings
    primary_findings = synthesis.get('primary_findings', [])
    if primary_findings:
        report += "### Primary Research Findings\n\n"
        for i, finding in enumerate(primary_findings[:5], 1):
            confidence = finding.get('confidence', 0.0)
            report += f"{i}. **{finding.get('finding', '')}**\n"
            report += f"   - Confidence Level: {confidence:.2f}\n"
            report += f"   - Support: {finding.get('support_level', 'Unknown')}\n\n"
    
    # Add consensus points
    consensus_points = synthesis.get('consensus_points', [])
    if consensus_points:
        report += "### Consensus Points\n\n"
        for point in consensus_points[:3]:
            report += f"- {point.get('finding', '')}\n"
            report += f"  - Consensus Strength: {point.get('consensus_strength', 0.0):.2f}\n"
            report += f"  - Supporting Sources: {point.get('supporting_sources', 0)}\n\n"
    
    # Add insights
    key_insights = insights.get('key_insights', [])
    if key_insights:
        report += "## Key Insights\n\n"
        for insight in key_insights[:5]:
            report += f"- **{insight.get('insight', '')}**\n"
            report += f"  - Significance: {insight.get('significance', 0.0):.2f}\n"
            report += f"  - Category: {insight.get('category', 'Unknown')}\n"
            if 'explanation' in insight:
                report += f"  - {insight['explanation']}\n"
            report += "\n"
    
    # Add recommendations
    recommendations = insights.get('recommendations', [])
    if recommendations:
        report += "## Recommendations\n\n"
        for rec in recommendations[:5]:
            report += f"### {rec.get('recommendation', '')}\n\n"
            report += f"- **Type:** {rec.get('type', 'Unknown')}\n"
            report += f"- **Priority:** {rec.get('priority', 'Medium')}\n"
            report += f"- **Timeframe:** {rec.get('timeframe', 'Unknown')}\n"
            if 'rationale' in rec:
                report += f"- **Rationale:** {rec['rationale']}\n"
            if 'specific_actions' in rec:
                report += "- **Specific Actions:**\n"
                for action in rec['specific_actions']:
                    report += f"  - {action}\n"
            report += "\n"
    
    # Add contradictions if any
    contradictions = synthesis.get('contradictions', [])
    if contradictions:
        report += "## Contradictory Evidence\n\n"
        for contradiction in contradictions[:3]:
            report += f"- **Issue:** {contradiction.get('contradiction_type', 'Unknown')}\n"
            report += f"  - Finding 1: {contradiction.get('group1_finding', '')}\n"
            report += f"  - Finding 2: {contradiction.get('group2_finding', '')}\n"
            report += f"  - Requires further investigation\n\n"
    
    # Add quality assessment
    report += "## Research Quality Assessment\n\n"
    quality_metrics = session_metadata['quality_metrics']
    report += f"- **Overall Quality:** {quality_metrics['quality_category'].replace('_', ' ').title()}\n"
    report += f"- **Source Success Rate:** {quality_metrics['success_rate']:.1%}\n"
    report += f"- **Content Quality:** {quality_metrics['avg_content_quality']:.2f}/1.0\n"
    report += f"- **Evidence Strength:** {quality_metrics['evidence_strength']:.2f}/1.0\n"
    report += f"- **Source Diversity:** {quality_metrics['source_diversity']:.2f}/1.0\n\n"
    
    # Add knowledge gaps
    knowledge_gaps = insights.get('knowledge_gaps', [])
    if knowledge_gaps:
        report += "## Identified Knowledge Gaps\n\n"
        for gap in knowledge_gaps[:3]:
            if isinstance(gap, dict):
                report += f"- **{gap.get('gap_type', 'Unknown')}:** {gap.get('description', '')}\n"
                report += f"  - Severity: {gap.get('severity', 'Unknown')}\n"
                report += f"  - Research Needed: {gap.get('research_needed', 'Unknown')}\n\n"
            else:
                report += f"- {gap}\n\n"
    
    # Add footer
    report += f"""## Session Information

- **Session ID:** {session_metadata['session_info']['session_id']}
- **Generated by:** Somnus AI Research System
- **Workflow Type:** {results.get('workflow_type', 'Comprehensive')}
- **Research Scope:** {session_metadata.get('research_scope', {}).get('scope_rating', 'Unknown')}

---

*This report was automatically generated by the Somnus AI Browser Research System. All findings should be verified through additional sources when making important decisions.*
"""
    
    return report

    async def _update_research_database(self, results: Dict[str, Any], session_metadata: Dict[str, Any]):
        """Update AI's personal research database with session results"""
        
        logger = logging.getLogger(__name__)
        
        try:
            # Create research database entry
            db_entry = {
                'session_id': session_metadata['session_info']['session_id'],
                'query': results.get('query', ''),
                'timestamp': session_metadata['session_info']['timestamp'],
                'quality_score': session_metadata['quality_metrics']['overall_quality_score'],
                'outcome': session_metadata['session_outcome'],
                'key_topics': session_metadata.get('research_scope', {}).get('unique_topics', []),
                'primary_findings_count': len(results.get('synthesis', {}).get('primary_findings', [])),
                'sources_count': session_metadata['research_metrics']['sources_found'],
                'insights_count': len(results.get('insights', {}).get('key_insights', []))
            }
            
            # Save to AI's research database
            db_path = '/home/ai/research/database/research_sessions.jsonl'
            await self.vm.execute_command('mkdir -p /home/ai/research/database')
            
            # Append to JSONL database
            db_line = json.dumps(db_entry) + '\n'
            append_command = f'echo \'{db_line}\' >> {db_path}'
            await self.vm.execute_command(append_command)
            
            # Update research index for quick searching
            await self._update_research_index(db_entry)
            
        except Exception as e:
            logger.error(f"Failed to update research database: {e}")

    async def _update_research_index(self, db_entry: Dict[str, Any]):
        """Update searchable research index"""
        
        logger = logging.getLogger(__name__)
        
        try:
            # Create/update topic index
            topics_index_path = '/home/ai/research/database/topics_index.json'
            
            # Load existing index or create new
            try:
                index_content = await self.vm.read_file(topics_index_path)
                topics_index = json.loads(index_content) if index_content else {}
            except:
                topics_index = {}
            
            # Add topics from this session
            session_id = db_entry['session_id']
            for topic in db_entry.get('key_topics', []):
                if topic not in topics_index:
                    topics_index[topic] = []
                topics_index[topic].append({
                    'session_id': session_id,
                    'query': db_entry['query'],
                    'quality_score': db_entry['quality_score']
                })
            
            # Save updated index
            index_json = json.dumps(topics_index, indent=2)
            await self.vm.write_file(topics_index_path, index_json)
            
        except Exception as e:
            logger.error(f"Failed to update research index: {e}")

async def _generate_automation_script_if_needed(self, results: Dict[str, Any], session_metadata: Dict[str, Any]) -> Optional[str]:
    """Generate automation script if research patterns are detected"""
    
    # Check if this query type appears frequently
    query = results.get('query', '').lower()
    quality_score = session_metadata['quality_metrics']['overall_quality_score']
    
    # Only generate automation for high-quality research
    if quality_score < 0.7:
        return None
    
    # Detect automation-worthy patterns
    automation_indicators = [
        'daily', 'weekly', 'monthly', 'regular', 'ongoing', 'monitor', 'track', 'update'
    ]
    
    if not any(indicator in query for indicator in automation_indicators):
        return None
    
    # Generate automation script
    script_content = f'''#!/usr/bin/env python3
"""
AI Research Automation Script
Generated from successful research session: {session_metadata['session_info']['session_id']}
Original Query: {results.get('query', '')}
Quality Score: {quality_score:.2f}
"""

import asyncio
import json
import time
from datetime import datetime
from ai_browser_research_system import AIBrowserResearch

class AutomatedResearch:
    def __init__(self, vm_instance, orchestrator):
        self.research_system = AIBrowserResearch(vm_instance, orchestrator)
        self.original_query = "{results.get('query', '')}"
        self.workflow_type = "{results.get('workflow_type', 'comprehensive')}"
        
    async def run_automated_research(self, modified_query=None):
        """Run automated research with optional query modification"""
        
        # Use modified query or generate time-based variation
        if modified_query:
            query = modified_query
        else:
            # Add current date context to original query
            current_date = datetime.now().strftime("%Y-%m")
            query = f"{{self.original_query}} {{current_date}} latest developments"
        
        print(f"Running automated research for: {{query}}")
        
        # Execute research
        results = await self.research_system.conduct_visual_research(
            query=query,
            workflow_type=self.workflow_type
        )
        
        # Compare with baseline results if available
        baseline_path = '/home/ai/research/automation/baseline_{session_metadata["session_info"]["session_id"]}.json'
        await self._compare_with_baseline(results, baseline_path)
        
        return results
    
    async def _compare_with_baseline(self, current_results, baseline_path):
        """Compare current results with baseline from original research"""
        # Implementation for comparing research results over time
        pass
    
    async def schedule_regular_research(self, interval_days=7):
        """Schedule this research to run regularly"""
        # Implementation for scheduling regular automated research
        pass

async def main():
    """Main execution function"""
    from ai_action_orchestrator import AIActionOrchestrator
    
    # Initialize components (would need actual VM instance)
    # vm_instance = get_current_vm_instance()
    # orchestrator = AIActionOrchestrator(vm_instance)
    # automation = AutomatedResearch(vm_instance, orchestrator)
    
    # Run automated research
    # results = await automation.run_automated_research()
    
    print("Automated research script ready for deployment")
    print("Note: Requires VM instance and orchestrator initialization")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    return script_content

    async def _cleanup_session_temp_files(self, session_timestamp: int):
        """Clean up temporary files from the research session"""
        
        logger = logging.getLogger(__name__)
        
        try:
            # Clean old temporary screenshots (keep only last 10 sessions)
            cleanup_commands = [
                f'find /home/ai/research/screenshots -name "search_*" -mtime +7 -delete',
                f'find /home/ai/research/cache -name "*.tmp" -mtime +1 -delete',
                f'find /tmp -name "research_*" -mtime +1 -delete'
            ]
            
            for command in cleanup_commands:
                try:
                    await self.vm.execute_command(command)
                except Exception as e:
                    logger.warning(f"Cleanup command failed: {command} - {e}")
                    
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")

def _categorize_quality_score(self, score: float) -> str:
    """Categorize quality score into human-readable categories"""
    
    if score >= 0.9:
        return 'excellent'
    elif score >= 0.8:
        return 'very_good'
    elif score >= 0.7:
        return 'good'
    elif score >= 0.6:
        return 'acceptable'
    elif score >= 0.4:
        return 'poor'
    else:
        return 'very_poor'

def _rate_research_scope(self, topic_count: int, methodology_count: int, has_geographic: bool) -> str:
    """Rate the scope of research coverage"""
    
    scope_score = 0
    
    # Topic diversity
    if topic_count >= 10:
        scope_score += 3
    elif topic_count >= 5:
        scope_score += 2
    elif topic_count >= 2:
        scope_score += 1
    
    # Methodology diversity
    if methodology_count >= 3:
        scope_score += 2
    elif methodology_count >= 2:
        scope_score += 1
    
    # Geographic scope
    if has_geographic:
        scope_score += 1
    
    # Convert to rating
    if scope_score >= 5:
        return 'comprehensive'
    elif scope_score >= 3:
        return 'good'
    elif scope_score >= 2:
        return 'moderate'
    else:
        return 'limited'
    
    # Utility methods
    def _sanitize_filename(self, url: str) -> str:
        """Sanitize URL for filename usage"""
        import re
        return re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))

    def _get_domain_from_url(self, url: str) -> str:
        """Extracts the domain from a given URL."""
        from urllib.parse import urlparse
        parsed_uri = urlparse(url)
        return '{uri.netloc}'.format(uri=parsed_uri)

    def _merge_step_results(self, main_results: Dict[str, Any], step_results: Dict[str, Any]):
        """Merge step results into main results structure"""
        if 'results' in step_results:
            main_results['sources_found'].extend(step_results['results'])
        if 'articles_analyzed' in step_results:
            main_results['research_notes'].extend(step_results['articles_analyzed'])
        if 'claims_verified' in step_results:
            main_results['fact_check_results'].extend(step_results['claims_verified'])

    def _calculate_relevance_score(self, source: Dict[str, Any], criteria: str, query: str) -> float:
        """Calculate relevance score for a given source."""
        score = 0.0
        query_words = set(query.lower().split())

        # Keyword match in title and snippet
        title = source.get('title', '').lower()
        snippet = source.get('snippet', '').lower()
        
        for word in query_words:
            if word in title:
                score += 0.5
            if word in snippet:
                score += 0.2

        # Source domain (simple heuristic: prioritize known academic/news domains)
        url = source.get('url', '')
        if "arxiv.org" in url or "pubmed.gov" in url or ".edu" in url:
            score += 0.8
        elif "wikipedia.org" in url or "nytimes.com" in url:
            score += 0.4

        # Position in search results (lower position is better)
        position = source.get('position', 100) # Default to high position if not found
        score += max(0, 1.0 - (position / 20.0)) # Max 1.0 for position 1, decays to 0

        return score
    
    async def _analyze_research_query(self, query: str) -> Dict[str, Any]:
        """Analyze research query to determine optimal strategy dynamically."""
        query_lower = query.lower()
        strategy = 'comprehensive'
        expected_source_count = 50
        recommended_engines = ['google', 'bing', 'duckduckgo']
        optimized_terms = {
            'google': query,
            'bing': query,
            'duckduckgo': query
        }

        # Prioritize general search engines for "how to" or "tutorial" queries
        if "how to" in query_lower or "tutorial" in query_lower or "guide" in query_lower:
            strategy = 'practical_guide'
            recommended_engines = ['google', 'bing'] # Focus on general web
            expected_source_count = 30
            optimized_terms['google'] = f"{query} tutorial"
            optimized_terms['bing'] = f"{query} guide"

        # Prioritize academic search engines for academic terms or author names
        academic_keywords = ["research paper", "study", "academic", "journal", "phd", "dissertation", "literature review", "meta-analysis"]
        author_name_pattern = r"[A-Z][a-z]+, [A-Z]\." # Simple pattern for "Lastname, F."

        if any(keyword in query_lower for keyword in academic_keywords) or re.search(author_name_pattern, query):
            strategy = 'academic_deep_dive'
            recommended_engines = ['scholar', 'arxiv', 'pubmed', 'semantic_scholar']
            expected_source_count = 100
            optimized_terms['scholar'] = f'"{query}" academic'
            optimized_terms['arxiv'] = f'"{query}"'
            optimized_terms['pubmed'] = f'"{query}"'
            optimized_terms['semantic_scholar'] = f'"{query}"'
            # Add general search for broader context if needed
            if 'google' not in recommended_engines:
                recommended_engines.append('google')
                optimized_terms['google'] = query

        # Simple synonym/related concept generation (can be enhanced with LLM)
        if "machine learning" in query_lower:
            optimized_terms['google'] += " AI, deep learning"
        elif "climate change" in query_lower:
            optimized_terms['google'] += " global warming, environmental impact"

        return {
            'strategy': strategy,
            'expected_source_count': expected_source_count,
            'recommended_engines': recommended_engines,
            'optimized_terms': optimized_terms
        }
    