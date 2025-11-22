import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from bs4 import BeautifulSoup
import requests
import time
import arxiv
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

@dataclass
class SearchResult:
    """Data class to store search results"""
    title: str
    link: str = ""
    snippet: str = ""
    body: str = ""
    authors: List[str] = None
    published: str = ""
    abstract: str = ""
    pdf_url: str = ""
    arxiv_url: str = ""
    # Clinical trials specific fields
    nct_id: str = ""
    status: str = ""
    study_type: str = ""
    conditions: List[str] = None
    interventions: List[str] = None
    phase: str = ""
    enrollment: int = 0
    locations: List[str] = None

class SearchTool:
    """Tool for performing Google, arXiv, and ClinicalTrials.gov searches"""
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

    def google_search(self, query: str, num_results: int = 2, max_chars: int = 500) -> List[SearchResult]:
        """
        Perform Google search and return enriched results
        """
        if not self.api_key or not self.search_engine_id:
            print("Missing API credentials")
            return []

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": num_results
            }

            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"Google Search API error: {response.status_code}")
                print(f"Response content: {response.text}")
                return []

            results = response.json().get("items", [])
            enriched_results = []

            for item in results:
                body = self._get_page_content(item["link"], max_chars)
                result = SearchResult(
                    title=item["title"],
                    link=item["link"],
                    snippet=item["snippet"],
                    body=body
                )
                enriched_results.append(result)
                time.sleep(1)  # Rate limiting

            return enriched_results

        except Exception as e:
            print(f"Error in Google search: {str(e)}")
            return []

    def arxiv_search(self, query: str, max_results: int = 2) -> List[SearchResult]:
        """
        Perform arXiv search and return results
        """
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            results = []
            for paper in client.results(search):
                paper_id = paper.entry_id.split('/')[-1]
                arxiv_url = f"https://arxiv.org/abs/{paper_id}"
                result = SearchResult(
                    title=paper.title,
                    authors=[author.name for author in paper.authors],
                    published=paper.published.strftime("%Y-%m-%d"),
                    abstract=paper.summary,
                    pdf_url=paper.pdf_url,
                    arxiv_url=arxiv_url
                )
                results.append(result)
                time.sleep(0.5)  # Gentle rate limiting

            return results

        except Exception as e:
            print(f"Error in Arxiv search: {str(e)}")
            return []

    def clinicaltrials_search_beta_api(self, query: str, max_results: int = 2) -> List[SearchResult]:
        """
        Perform ClinicalTrials.gov search using their current Beta API
        https://clinicaltrials.gov/data-api/ui
        """
        try:
            # Base URL for the ClinicalTrials.gov Beta API
            base_url = "https://clinicaltrials.gov/api/v2/studies"
            
            # Parameters for the API request
            params = {
                "query.term": query,
                "pageSize": max_results,
                "format": "json"
            }
            
            # Make the API request
            response = requests.get(base_url, params=params, timeout=15)
            
            if response.status_code != 200:
                print(f"ClinicalTrials.gov API error: {response.status_code}")
                print(f"Response content: {response.text}")
                return []
            
            # Parse the JSON response
            data = response.json()
            studies = data.get("studies", [])
            
            if not studies:
                print(f"No clinical trial studies found for query: {query}")
                return []
            
            results = []
            for study in studies:
                # Extract study details
                protocol_section = study.get("protocolSection", {})
                identification = protocol_section.get("identificationModule", {})
                status_module = protocol_section.get("statusModule", {})
                design_module = protocol_section.get("designModule", {})
                conditions_module = protocol_section.get("conditionsModule", {})
                description_module = protocol_section.get("descriptionModule", {})
                
                # Extract NCT ID
                nct_id = identification.get("nctId", "")
                if not nct_id:
                    continue
                
                # Extract title
                title = identification.get("briefTitle", "")
                official_title = identification.get("officialTitle", "")
                
                # Extract study details
                study_type = design_module.get("studyType", "")
                phase_list = design_module.get("phases", [])
                phase = ", ".join(phase_list) if phase_list else "Not Specified"
                
                # Extract status
                status = status_module.get("overallStatus", "")
                
                # Extract conditions
                conditions = conditions_module.get("conditions", [])
                
                # Extract description/summary
                summary = description_module.get("briefSummary", "")
                detailed_desc = description_module.get("detailedDescription", "")
                
                # Create the study URL - using the stable URL format
                # Changed from /study/ to /ct2/show/ which is more stable
                study_url = f"https://clinicaltrials.gov/ct2/show/{nct_id}"
                
                result = SearchResult(
                    title=title if title else official_title,
                    link=study_url,
                    snippet=summary[:200] + "..." if len(summary) > 200 else summary,
                    abstract=summary if summary else detailed_desc,
                    nct_id=nct_id,
                    status=status,
                    study_type=study_type,
                    phase=phase,
                    conditions=conditions
                )
                results.append(result)
                
                # Get full study details if needed
                if not summary and not detailed_desc:
                    detailed_study = self._get_clinical_trial_details_api(nct_id)
                    if detailed_study:
                        result.abstract = detailed_study.abstract
                        result.conditions = detailed_study.conditions
                        result.interventions = detailed_study.interventions
                
                time.sleep(2)  # Increased rate limiting for API requests
                
            return results
            
        except Exception as e:
            print(f"Error in ClinicalTrials.gov Beta API search: {str(e)}")
            return []

    def _get_clinical_trial_details_api(self, nct_id: str) -> Optional[SearchResult]:
        """
        Fetch detailed information for a specific study using the Beta API
        """
        # Validate NCT ID format
        if not nct_id or not nct_id.startswith("NCT"):
            print(f"Invalid NCT ID format: {nct_id}")
            return None
            
        try:
            # Base URL for the ClinicalTrials.gov Beta API - single study endpoint
            base_url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
            
            # Make the API request
            response = requests.get(base_url, params={"format": "json"}, timeout=15)
            
            if response.status_code != 200:
                print(f"ClinicalTrials.gov API error for study {nct_id}: {response.status_code}")
                return None
            
            # Parse the JSON response
            study = response.json()
            
            # Extract study details
            protocol_section = study.get("protocolSection", {})
            identification = protocol_section.get("identificationModule", {})
            description_module = protocol_section.get("descriptionModule", {})
            conditions_module = protocol_section.get("conditionsModule", {})
            intervention_module = protocol_section.get("armsInterventionsModule", {})
            
            # Extract title
            title = identification.get("briefTitle", "")
            
            # Extract summary
            summary = description_module.get("briefSummary", "")
            detailed_desc = description_module.get("detailedDescription", "")
            
            # Extract conditions
            conditions = conditions_module.get("conditions", [])
            
            # Extract interventions
            interventions_list = []
            for intervention in intervention_module.get("interventions", []):
                intervention_name = intervention.get("name", "")
                if intervention_name:
                    interventions_list.append(intervention_name)
            
            # Create the study URL using stable format
            study_url = f"https://clinicaltrials.gov/ct2/show/{nct_id}"
            
            return SearchResult(
                title=title,
                link=study_url,
                nct_id=nct_id,
                abstract=summary if summary else detailed_desc,
                conditions=conditions,
                interventions=interventions_list
            )
            
        except Exception as e:
            print(f"Error fetching details for study {nct_id}: {str(e)}")
            return None

    def clinicaltrials_search_scrape(self, query: str, max_results: int = 2) -> List[SearchResult]:
        """
        Perform ClinicalTrials.gov search by scraping the website
        Use this as a fallback if the API method doesn't work
        """
        try:
            # Construct the search URL - updated to use v2 search
            base_url = "https://clinicaltrials.gov/search"
            params = {
                "term": query,
                "draw": 1,
                "rank": 1
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            # Make the initial request to get the search results page
            response = requests.get(base_url, params=params, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"ClinicalTrials.gov search error: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, "html.parser")
            study_links = soup.select(".ct-search-result a.ct-search-result__title-link")[:max_results]
            
            if not study_links:
                print("No study links found on the search results page")
                return []
            
            results = []
            for link in study_links:
                href = link.get('href', '')
                if not href:
                    continue
                    
                # Extract NCT ID from href and validate
                try:
                    # Different ways the NCT ID might appear in the URL
                    if '/study/' in href:
                        nct_id = href.split('/')[-1]
                    elif '/ct2/show/' in href:
                        nct_id = href.split('/')[-1]
                    else:
                        # Try to find NCT pattern (NCTXXXXXXXX)
                        import re
                        nct_match = re.search(r'(NCT\d{8})', href)
                        if nct_match:
                            nct_id = nct_match.group(1)
                        else:
                            print(f"Could not extract NCT ID from href: {href}")
                            continue
                except Exception as e:
                    print(f"Error extracting NCT ID from {href}: {str(e)}")
                    continue
                
                # Ensure proper URL format
                study_url = f"https://clinicaltrials.gov/ct2/show/{nct_id}"
                
                # Get detailed information from the study page
                study_info = self._get_clinical_trial_details_scrape(study_url)
                if study_info:
                    results.append(study_info)
                    time.sleep(2)  # Increased rate limiting
            
            return results
            
        except Exception as e:
            print(f"Error in ClinicalTrials.gov scrape search: {str(e)}")
            return []

    def _get_clinical_trial_details_scrape(self, url: str) -> Optional[SearchResult]:
        """
        Extract detailed information from a ClinicalTrials.gov study page using web scraping
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"Error accessing {url}: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract NCT ID - handle different URL formats
            try:
                if '/ct2/show/' in url:
                    nct_id = url.split('/')[-1]
                else:
                    # Try to find NCT pattern
                    import re
                    nct_match = re.search(r'(NCT\d{8})', url)
                    if nct_match:
                        nct_id = nct_match.group(1)
                    else:
                        # Last resort - check the page content for NCT ID
                        nct_elem = soup.find(string=re.compile(r'NCT\d{8}'))
                        if nct_elem:
                            nct_match = re.search(r'(NCT\d{8})', nct_elem)
                            nct_id = nct_match.group(1) if nct_match else "Unknown"
                        else:
                            nct_id = "Unknown"
            except Exception:
                nct_id = "Unknown"
            
            # Extract title - handle different page structures
            title = "Unknown Title"
            title_selectors = ["h1.tr-h1", "h1.ct-title", ".headline-title"]
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            # Extract status
            status = "Unknown"
            status_selectors = [".ct-recruitment-status div.ct-recruitment-status__label", 
                              ".statusLabel", 
                              "p:contains('Recruitment Status:')"]
            for selector in status_selectors:
                status_elem = None
                try:
                    if ':contains' in selector:
                        # Handle custom contains selector
                        text = selector.split(':contains(')[1].strip("')")
                        for p in soup.find_all('p'):
                            if text in p.get_text():
                                status_elem = p
                                break
                    else:
                        status_elem = soup.select_one(selector)
                except Exception:
                    continue
                    
                if status_elem:
                    status_text = status_elem.get_text().strip()
                    if "Status:" in status_text:
                        status = status_text.split("Status:")[1].strip()
                    else:
                        status = status_text
                    break
            
            # Extract summary - try different selectors
            summary = ""
            summary_selectors = ["#brief-summary div.tr-indent2", 
                               ".ct-body__section div.tr-indent1", 
                               "section#brief-summary"]
            for selector in summary_selectors:
                summary_elem = soup.select_one(selector)
                if summary_elem:
                    summary = summary_elem.get_text().strip()
                    break
            
            # Extract study type
            study_type = ""
            study_type_selectors = [
                lambda s: s.find(string="Study Type:"),
                lambda s: s.find("th", string="Study Type")
            ]
            for selector_func in study_type_selectors:
                study_type_label = selector_func(soup)
                if study_type_label:
                    if study_type_label.parent:
                        value_elem = None
                        if study_type_label.parent.name == "th":
                            # Handle table format
                            value_elem = study_type_label.parent.find_next("td")
                        else:
                            # Handle div format
                            value_elem = study_type_label.parent.find_next("div", class_="ct-data-elem__value")
                            if not value_elem:
                                value_elem = study_type_label.parent.find_next_sibling("div")
                        
                        if value_elem:
                            study_type = value_elem.get_text().strip()
                            break
            
            # Extract phase
            phase = ""
            phase_selectors = [
                lambda s: s.find(string="Phase:"),
                lambda s: s.find("th", string="Phase")
            ]
            for selector_func in phase_selectors:
                phase_label = selector_func(soup)
                if phase_label:
                    if phase_label.parent:
                        value_elem = None
                        if phase_label.parent.name == "th":
                            # Handle table format
                            value_elem = phase_label.parent.find_next("td")
                        else:
                            # Handle div format
                            value_elem = phase_label.parent.find_next("div", class_="ct-data-elem__value")
                            if not value_elem:
                                value_elem = phase_label.parent.find_next_sibling("div")
                        
                        if value_elem:
                            phase = value_elem.get_text().strip()
                            break
            
            # Extract conditions
            conditions = []
            conditions_selectors = ["#conditions", "section#conditions", "section:contains('Condition')"]
            for selector in conditions_selectors:
                conditions_section = None
                try:
                    if ':contains' in selector:
                        # Handle custom contains selector
                        text = selector.split(':contains(')[1].strip("')")
                        for section in soup.find_all('section'):
                            if text in section.get_text():
                                conditions_section = section
                                break
                    else:
                        conditions_section = soup.select_one(selector)
                except Exception:
                    continue
                    
                if conditions_section:
                    # Try to find conditions in list items
                    condition_items = conditions_section.select("li")
                    if condition_items:
                        conditions = [item.get_text().strip() for item in condition_items]
                    else:
                        # If no list items, try to get text content
                        conditions_text = conditions_section.get_text().strip()
                        # Remove section title if present
                        if ":" in conditions_text:
                            conditions_text = conditions_text.split(":", 1)[1].strip()
                        conditions = [cond.strip() for cond in conditions_text.split(",")]
                    break
            
            return SearchResult(
                title=title,
                link=url,
                snippet=summary[:200] + "..." if summary and len(summary) > 200 else summary,
                abstract=summary,
                nct_id=nct_id,
                status=status,
                study_type=study_type,
                phase=phase,
                conditions=conditions
            )
            
        except Exception as e:
            print(f"Error extracting details from {url}: {str(e)}")
            return None

    def _get_page_content(self, url: str, max_chars: int) -> str:
        """
        Fetch and extract text content from a webpage
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator=" ", strip=True)
            return text[:max_chars]
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

class GapAnalysisEngine:
    """
    Research Gap Analysis Engine using Flan-T5-Large model
    Identifies knowledge, methodological, dataset, and temporal gaps in research papers
    """
    
    def __init__(self):
        """Initialize the Flan-T5 model and tokenizer"""
        print("Loading Flan-T5-Large model for gap analysis...")
        self.model_name = "google/flan-t5-large"
        
        try:
            # Load tokenizer and model from Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Set device (GPU if available, else CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    def _create_few_shot_prompt(self, papers: List[SearchResult], topic: str) -> str:
        """
        Create a few-shot prompt with examples to guide the model
        
        Args:
            papers: List of SearchResult objects containing paper information
            topic: The research topic being analyzed
            
        Returns:
            Formatted prompt string with few-shot examples
        """
        
        # Format the input papers with more detail
        papers_text = ""
        for i, paper in enumerate(papers[:12], 1):  # Increased from 10 to 12 papers
            papers_text += f"\n--- Paper {i} ---\n"
            papers_text += f"Title: {paper.title}\n"
            
            # Add authors if available
            if paper.authors:
                papers_text += f"Authors: {', '.join(paper.authors[:3])}\n"  # First 3 authors
            
            # Add publication date
            if paper.published:
                papers_text += f"Published: {paper.published}\n"
            
            # Add abstract/summary with more characters
            if paper.abstract:
                papers_text += f"Abstract: {paper.abstract[:500]}...\n"  # Increased from 300
            elif paper.snippet:
                papers_text += f"Summary: {paper.snippet[:500]}...\n"
            
            # Add study type for clinical trials
            if paper.study_type:
                papers_text += f"Study Type: {paper.study_type}\n"
            if paper.phase:
                papers_text += f"Phase: {paper.phase}\n"
        
        # Enhanced few-shot prompt with detailed, insightful examples
        prompt = f"""You are a PhD-level research analyst with expertise in identifying critical research gaps. Analyze the papers deeply and provide SPECIFIC, ACTIONABLE gaps that researchers can build upon.

EXAMPLE 1:
Topic: Federated Learning Privacy
Papers:
Paper 1: Differential Privacy in Federated Learning (2022)
Abstract: We apply differential privacy mechanisms to federated learning, achieving epsilon=0.5 privacy guarantees with 3% accuracy loss on MNIST...
Paper 2: Secure Aggregation Protocols (2021)  
Abstract: Novel cryptographic protocols for secure parameter aggregation in federated settings, tested on 100 devices...
Paper 3: Privacy Budget Allocation (2020)
Abstract: Adaptive privacy budget allocation across federated learning rounds, optimizing utility-privacy tradeoff...

DEEP ANALYSIS:
{{
  "knowledge_gaps": [
    "CRITICAL: No empirical studies on privacy leakage through gradient updates in heterogeneous device environments (IoT + mobile + edge)",
    "UNEXPLORED: Privacy guarantees degrade with model size - no theoretical bounds for models >1B parameters in federated settings",
    "MISSING: Real-world privacy attack success rates beyond academic datasets - what about medical records, financial data?",
    "UNKNOWN: Privacy-utility tradeoffs in non-IID data distributions with extreme class imbalance (e.g., rare disease detection)"
  ],
  "methodological_gaps": [
    "NO FRAMEWORK: Combining differential privacy + secure aggregation + homomorphic encryption in single unified system - existing work addresses only 1-2",
    "MISSING EVALUATION: Longitudinal privacy analysis across 100+ training rounds - current studies stop at 10-20 rounds",
    "ABSENT: Privacy-preserving techniques for vertical federated learning (different features per client) - all work focuses on horizontal FL",
    "LACKING: Adaptive privacy mechanisms that adjust epsilon based on attack risk in real-time during training"
  ],
  "dataset_gaps": [
    "CRITICAL MISSING: Standardized privacy attack benchmark suite with diverse attack vectors (membership inference, model inversion, gradient leakage)",
    "NO BENCHMARK: Real federated datasets with known privacy violations - current benchmarks use synthetic privacy labels",
    "LACKING: Heterogeneous device capability datasets showing computation/communication/privacy tradeoffs across 1000+ devices",
    "ABSENT: Privacy audit trails from production federated learning deployments"
  ],
  "temporal_gaps": [
    "OUTDATED (pre-2023): Privacy analysis doesn't account for LLM-scale models (GPT-4, Llama-2) in federated settings",
    "OBSOLETE HARDWARE: Studies assume 2019-era mobile devices - modern edge TPUs and neural engines change privacy-performance dynamics",
    "MISSING 2024 CONTEXT: New privacy regulations (EU AI Act, US state laws) not reflected in federated learning design"
  ]
}}

EXAMPLE 2:
Topic: Explainable AI in Medical Diagnosis
Papers:
Paper 1: SHAP for Medical Imaging (2021)
Abstract: Applying SHAP values to explain CNN predictions in chest X-ray diagnosis, achieving correlation with radiologist attention...
Paper 2: LIME in Clinical Decision Support (2020)
Abstract: Local interpretable model-agnostic explanations for electronic health record-based predictions...
Paper 3: Attention Visualization in Diagnosis (2022)
Abstract: Visualizing transformer attention patterns for disease classification from medical images...

DEEP ANALYSIS:
{{
  "knowledge_gaps": [
    "CRITICAL: Zero studies on whether physicians ACTUALLY change treatment decisions based on AI explanations - all measure 'trust' not clinical outcomes",
    "UNEXPLORED: Explanation quality for rare diseases (<1% prevalence) where models have insufficient training data - current work focuses on common conditions",
    "MISSING: Conflicting explanation scenarios - what happens when SHAP suggests feature X but physician believes Y? No resolution frameworks exist",
    "UNKNOWN: Cognitive load of processing AI explanations during emergency medicine - explanations may slow critical decisions"
  ],
  "methodological_gaps": [
    "NO GOLD STANDARD: Evaluating explanation correctness requires ground truth (which features TRULY matter) - current methods use proxy metrics like 'plausibility'",
    "MISSING FRAMEWORK: Comparative explanation methods - should we use SHAP vs LIME vs attention? No decision tree exists for medical contexts",
    "ABSENT: Real-time explanation generation for time-critical diagnoses - current methods take 30+ seconds, unacceptable in ER settings",
    "LACKING: Multi-modal explanations combining imaging + EHR + genomics - existing work explains single modality only"
  ],
  "dataset_gaps": [
    "CRITICAL MISSING: Datasets with physician-annotated 'ground truth' explanations for 1000+ diagnoses - current datasets lack expert labels",
    "NO BENCHMARK: Longitudinal patient data showing how explanations affected treatment outcomes over months/years",
    "LACKING: Adversarial explanation datasets - cases where explanations are intentionally misleading yet appear valid",
    "ABSENT: Cross-hospital explanation generalization datasets - do explanations transfer between institutions?"
  ],
  "temporal_gaps": [
    "OUTDATED (pre-2023): Explanation methods designed for CNNs, not foundation models (MedPaLM, GPT-4 for medicine) with emergent reasoning",
    "OBSOLETE REGULATORY: FDA guidance on AI explainability from 2019 - new 2024 requirements for transparency not addressed",
    "MISSING CURRENT CONTEXT: Post-COVID telehealth adoption means remote explanation delivery - no studies on explaining AI over video consultations"
  ]
}}

NOW ANALYZE THIS RESEARCH AREA WITH THE SAME DEPTH AND SPECIFICITY:
Topic: {topic}
Papers:{papers_text}

REQUIREMENTS FOR YOUR ANALYSIS:
1. Each gap must be SPECIFIC with technical details, numbers, or clear scenarios - NOT generic statements
2. Identify CRITICAL gaps that block real-world deployment or advancement
3. Point out CONTRADICTIONS or CONFLICTS in existing research approaches
4. Consider PRACTICAL implications (cost, time, feasibility, scalability)
5. Include WHY each gap matters - what problem does it cause? What opportunities does it create?
6. Mention specific technologies, datasets, methods, or evaluation metrics that are missing
7. Be ACTIONABLE - researchers should know exactly what to investigate next
8. FIND AT LEAST 2-3 GAPS PER CATEGORY - there are ALWAYS gaps in research!

IMPORTANT: Even well-researched areas have gaps! Look for:
- Unstudied combinations of techniques
- Missing benchmarks or standardized evaluations  
- Lack of real-world deployment studies
- Outdated assumptions from older papers
- Unexplored edge cases or failure modes
- Missing cross-domain applications

You MUST respond with ONLY valid JSON in this exact format (no extra text):
{{
  "knowledge_gaps": ["gap1", "gap2", "gap3"],
  "methodological_gaps": ["gap1", "gap2", "gap3"],
  "dataset_gaps": ["gap1", "gap2", "gap3"],
  "temporal_gaps": ["gap1", "gap2"]
}}

Generate the JSON now:"""

        return prompt
    
    def analyze_gaps(self, papers: List[SearchResult], topic: str) -> Dict:
        """
        Analyze research papers to identify gaps
        
        Args:
            papers: List of SearchResult objects from searches
            topic: The research topic being analyzed
            
        Returns:
            Dictionary containing identified gaps in four categories
        """
        
        if not self.model or not self.tokenizer:
            return {
                "error": "Model not loaded",
                "knowledge_gaps": [],
                "methodological_gaps": [],
                "dataset_gaps": [],
                "temporal_gaps": []
            }
        
        print(f"Analyzing {len(papers)} papers for research gaps...")
        
        # Create the few-shot prompt
        prompt = self._create_few_shot_prompt(papers, topic)
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)
            
            # Generate the analysis with better parameters for detailed output
            outputs = self.model.generate(
                **inputs,
                max_length=1024,  # Increased from 512 for more detailed gaps
                min_length=200,   # Ensure substantial output
                num_beams=5,      # Increased from 4 for better quality
                temperature=0.8,  # Slightly higher for more creative insights
                do_sample=True,
                top_p=0.92,       # Slightly higher for more diversity
                repetition_penalty=1.2,  # Avoid repetitive gaps
                length_penalty=1.0
            )
            
            # Decode the output
            analysis_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print("Raw model output:", analysis_text[:200])
            
            # Try to parse as JSON
            try:
                gaps = json.loads(analysis_text)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract structured information
                gaps = self._parse_unstructured_output(analysis_text)
            
            # Ensure all required fields exist
            gaps.setdefault("knowledge_gaps", [])
            gaps.setdefault("methodological_gaps", [])
            gaps.setdefault("dataset_gaps", [])
            gaps.setdefault("temporal_gaps", [])
            
            return gaps
            
        except Exception as e:
            print(f"Error during gap analysis: {str(e)}")
            return {
                "error": str(e),
                "knowledge_gaps": [],
                "methodological_gaps": [],
                "dataset_gaps": [],
                "temporal_gaps": []
            }
    
    def _parse_unstructured_output(self, text: str) -> Dict:
        """
        Parse unstructured model output into structured format
        
        Args:
            text: Raw text output from the model
            
        Returns:
            Dictionary with gap categories
        """
        gaps = {
            "knowledge_gaps": [],
            "methodological_gaps": [],
            "dataset_gaps": [],
            "temporal_gaps": []
        }
        
        # Simple parsing logic - extract bullet points or numbered lists
        lines = text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if 'knowledge' in line.lower():
                current_category = 'knowledge_gaps'
            elif 'methodological' in line.lower() or 'method' in line.lower():
                current_category = 'methodological_gaps'
            elif 'dataset' in line.lower() or 'data' in line.lower():
                current_category = 'dataset_gaps'
            elif 'temporal' in line.lower() or 'time' in line.lower():
                current_category = 'temporal_gaps'
            elif current_category and (line.startswith('-') or line.startswith('•') or any(line.startswith(f"{i}.") for i in range(10))):
                # Clean the line
                cleaned = line.lstrip('-•0123456789. ').strip()
                if cleaned and len(cleaned) > 10:
                    gaps[current_category].append(cleaned)
        
        return gaps
    
    def format_gaps_for_display(self, gaps: Dict) -> str:
        """
        Format the gaps analysis for display in Streamlit
        
        Args:
            gaps: Dictionary of identified gaps
            
        Returns:
            Formatted markdown string
        """
        output = "## 🔍 Research Gap Analysis\n\n"
        
        if gaps.get("error"):
            output += f"⚠️ Error: {gaps['error']}\n\n"
        
        # Knowledge Gaps
        output += "### 📚 Knowledge Gaps\n"
        output += "*Questions or areas that haven't been studied yet*\n\n"
        if gaps.get("knowledge_gaps"):
            for i, gap in enumerate(gaps["knowledge_gaps"], 1):
                output += f"{i}. {gap}\n"
        else:
            output += "*No significant knowledge gaps identified*\n"
        output += "\n"
        
        # Methodological Gaps
        output += "### 🔬 Methodological Gaps\n"
        output += "*Research approaches or methods not tried*\n\n"
        if gaps.get("methodological_gaps"):
            for i, gap in enumerate(gaps["methodological_gaps"], 1):
                output += f"{i}. {gap}\n"
        else:
            output += "*No significant methodological gaps identified*\n"
        output += "\n"
        
        # Dataset Gaps
        output += "### 💾 Dataset Gaps\n"
        output += "*Missing or under-explored datasets*\n\n"
        if gaps.get("dataset_gaps"):
            for i, gap in enumerate(gaps["dataset_gaps"], 1):
                output += f"{i}. {gap}\n"
        else:
            output += "*No significant dataset gaps identified*\n"
        output += "\n"
        
        # Temporal Gaps
        output += "### ⏰ Temporal Gaps\n"
        output += "*Areas that are outdated or need updating*\n\n"
        if gaps.get("temporal_gaps"):
            for i, gap in enumerate(gaps["temporal_gaps"], 1):
                output += f"{i}. {gap}\n"
        else:
            output += "*No significant temporal gaps identified*\n"
        output += "\n"
        
        return output

class HypothesisGenerator:
    """
    Hypothesis Generation and Scoring Engine
    Converts research gaps into actionable hypotheses with novelty, impact, and feasibility scores
    """
    
    def __init__(self):
        """Initialize OpenAI client and embedding model"""
        print("Initializing Hypothesis Generator...")
        
        # OpenAI for hypothesis generation
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found. Hypothesis generation will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        # Load embedding model for novelty scoring
        try:
            print("Loading embedding model for novelty scoring...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("Embedding model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {str(e)}")
            self.embedding_model = None
    
    def _create_hypothesis_prompt(self, gaps: Dict, topic: str) -> str:
        """
        Create few-shot prompt for hypothesis generation
        
        Args:
            gaps: Dictionary of identified research gaps
            topic: Research topic
            
        Returns:
            Formatted prompt for GPT-4
        """
        
        # Format gaps for the prompt
        gaps_text = "IDENTIFIED RESEARCH GAPS:\n\n"
        
        if gaps.get("knowledge_gaps"):
            gaps_text += "🔬 Knowledge Gaps:\n"
            for i, gap in enumerate(gaps["knowledge_gaps"][:5], 1):
                gaps_text += f"{i}. {gap}\n"
            gaps_text += "\n"
        
        if gaps.get("methodological_gaps"):
            gaps_text += "🛠️ Methodological Gaps:\n"
            for i, gap in enumerate(gaps["methodological_gaps"][:5], 1):
                gaps_text += f"{i}. {gap}\n"
            gaps_text += "\n"
        
        if gaps.get("dataset_gaps"):
            gaps_text += "💾 Dataset Gaps:\n"
            for i, gap in enumerate(gaps["dataset_gaps"][:5], 1):
                gaps_text += f"{i}. {gap}\n"
            gaps_text += "\n"
        
        if gaps.get("temporal_gaps"):
            gaps_text += "⏰ Temporal Gaps:\n"
            for i, gap in enumerate(gaps["temporal_gaps"][:5], 1):
                gaps_text += f"{i}. {gap}\n"
            gaps_text += "\n"
        
        # Few-shot prompt with detailed examples
        prompt = f"""You are a world-class research strategist. Convert research gaps into actionable, fundable research hypotheses.

EXAMPLE INPUT:
Gap: "CRITICAL: No empirical studies on privacy leakage through gradient updates in heterogeneous device environments (IoT + mobile + edge)"

EXAMPLE OUTPUT:
{{
  "hypothesis_id": 1,
  "title": "Privacy-Preserving Gradient Sharing in Heterogeneous Federated Learning",
  "gap_addressed": "No empirical studies on privacy leakage in heterogeneous federated environments",
  "problem_statement": "Current federated learning privacy studies assume homogeneous devices (all mobile or all edge), but real deployments mix IoT sensors, mobile phones, and edge servers. Privacy guarantees derived for homogeneous settings may not hold when devices have vastly different computational capabilities, leading to asymmetric privacy leakage where weaker devices reveal more information through gradient updates.",
  "proposed_solution": "Design and implement a heterogeneous federated learning testbed with 1000+ devices (IoT sensors, smartphones, edge servers) to empirically measure privacy leakage across device types. Develop adaptive differential privacy mechanisms that adjust noise levels based on device computational capacity. Create theoretical framework proving privacy bounds for heterogeneous settings.",
  "expected_impact": "First empirical evidence of device-specific privacy risks in real federated deployments. Will enable $50M+ industry deployments of federated learning in healthcare and smart cities by providing provable privacy guarantees for mixed-device environments. Expected to reduce privacy leakage by 40-60% compared to one-size-fits-all approaches.",
  "methodology": "1) Deploy federated learning across 1000 heterogeneous devices, 2) Conduct membership inference and gradient inversion attacks tailored to each device type, 3) Measure information leakage using mutual information and attack success rates, 4) Develop and evaluate adaptive DP mechanisms, 5) Prove theoretical privacy bounds",
  "required_resources": {{
    "compute": "Access to 1000 heterogeneous devices (or simulation), 10 GPUs for attack models",
    "data": "Federated datasets (FEMNIST, Shakespeare, medical imaging if available)",
    "tools": "PyTorch, PySyft for federated learning, privacy attack libraries",
    "team": "2-3 researchers (1 privacy expert, 1 systems engineer, 1 ML researcher)",
    "timeline": "12-18 months",
    "estimated_cost": "$80K-120K (device access, compute, personnel)"
  }},
  "novelty_score": 9,
  "impact_score": 9,
  "feasibility_score": 7,
  "priority": "HIGH",
  "target_venue": "USENIX Security, IEEE S&P, NeurIPS (privacy track)",
  "potential_collaborators": "Federated learning research groups at Google, Meta, academic labs working on privacy-preserving ML"
}}

EXAMPLE INPUT 2:
Gap: "MISSING FRAMEWORK: Real-time carbon footprint tracking during training with <5% overhead"

EXAMPLE OUTPUT 2:
{{
  "hypothesis_id": 2,
  "title": "ZeroCarbon: Real-Time Energy and Carbon Tracking for ML Training",
  "gap_addressed": "No real-time carbon tracking framework with <5% overhead during model training",
  "problem_statement": "Existing carbon tracking tools (CodeCarbon, experiment-impact-tracker) add 20-30% training overhead, making them impractical for production use. Researchers cannot monitor carbon emissions in real-time during training, leading to wasteful hyperparameter searches and missed optimization opportunities. Need lightweight system that tracks energy/carbon with <5% overhead while providing actionable insights.",
  "proposed_solution": "Develop ZeroCarbon: a GPU kernel-level energy monitoring system that hooks into CUDA/PyTorch at minimal overhead. Use hardware performance counters instead of external power meters. Implement predictive modeling to estimate full training carbon cost after first 100 iterations. Provide real-time dashboard showing carbon/hour and projected total emissions with recommendations to stop wasteful runs.",
  "expected_impact": "Enable carbon-aware ML training at scale. If adopted by 10% of ML researchers, could reduce global ML training carbon by 5-8% (estimated 50K-100K tons CO2/year). Will become standard tool in MLOps pipelines at major tech companies. Expected to identify and stop 30-40% of wasteful training runs early.",
  "methodology": "1) Develop CUDA kernel hooks for energy monitoring, 2) Benchmark overhead across 20+ model architectures (ResNet, BERT, GPT), 3) Build predictive models for carbon estimation, 4) Create real-time dashboard with FastAPI backend, 5) Conduct user studies with 50+ ML practitioners, 6) Integrate with popular frameworks (PyTorch Lightning, HuggingFace Trainer)",
  "required_resources": {{
    "compute": "4-8 GPUs (A100/H100) for benchmarking, cloud credits for testing",
    "data": "Training logs from diverse ML workloads",
    "tools": "CUDA, PyTorch, FastAPI, React for dashboard",
    "team": "2-3 developers (1 systems engineer, 1 ML engineer, 1 full-stack dev)",
    "timeline": "6-9 months",
    "estimated_cost": "$40K-60K (compute, personnel)"
  }},
  "novelty_score": 8,
  "impact_score": 9,
  "feasibility_score": 8,
  "priority": "HIGH",
  "target_venue": "MLSys, SysML, NeurIPS (datasets/benchmarks track)",
  "potential_collaborators": "Green AI labs, cloud providers (AWS, Google Cloud), MLOps startups"
}}

NOW GENERATE HYPOTHESES FOR THIS RESEARCH AREA:

Topic: {topic}

{gaps_text}

Generate EXACTLY 5 HIGH-QUALITY, ACTIONABLE research hypotheses that:
1. Address the most CRITICAL gaps identified above
2. Are SPECIFIC with technical details, timelines, and resource estimates
3. Include realistic impact projections (numbers, percentages, cost savings)
4. Provide clear methodology and required resources
5. Can realistically be executed by a research team or PhD student
6. Have clear success metrics and target publication venues
7. Cover DIVERSE approaches - don't repeat similar ideas

CRITICAL: Generate EXACTLY 5 hypotheses, not 3-4. Each should be unique and address different gaps.

IMPORTANT: Your response must be ONLY valid JSON in this exact format:
{{
  "hypotheses": [
    {{
      "hypothesis_id": 1,
      "title": "...",
      "gap_addressed": "...",
      "problem_statement": "...",
      "proposed_solution": "...",
      "expected_impact": "...",
      "methodology": "...",
      "required_resources": {{...}},
      "novelty_score": 8,
      "impact_score": 9,
      "feasibility_score": 7,
      "priority": "HIGH",
      "target_venue": "...",
      "potential_collaborators": "..."
    }},
    {{
      "hypothesis_id": 2,
      ...
    }},
    {{
      "hypothesis_id": 3,
      ...
    }},
    {{
      "hypothesis_id": 4,
      ...
    }},
    {{
      "hypothesis_id": 5,
      ...
    }}
  ]
}}

Respond with ONLY the JSON containing EXACTLY 5 hypotheses, no additional text."""

        return prompt
    
    def _calculate_novelty_score(self, hypothesis_text: str, papers: List[SearchResult]) -> float:
        """
        Calculate novelty score by comparing hypothesis to existing papers using embeddings
        
        Args:
            hypothesis_text: The hypothesis text to evaluate
            papers: List of existing papers
            
        Returns:
            Novelty score from 1-10 (10 = highly novel)
        """
        
        if not self.embedding_model or not papers:
            return 7.0  # Default moderate novelty if can't calculate
        
        try:
            # Combine title and proposed solution for embedding
            hypothesis_embedding = self.embedding_model.encode([hypothesis_text])
            
            # Get embeddings for existing paper abstracts
            paper_texts = []
            for paper in papers[:15]:  # Limit to first 15 papers
                if paper.abstract:
                    paper_texts.append(paper.abstract[:500])
                elif paper.snippet:
                    paper_texts.append(paper.snippet[:500])
            
            if not paper_texts:
                return 7.0
            
            paper_embeddings = self.embedding_model.encode(paper_texts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(hypothesis_embedding, paper_embeddings)[0]
            
            # Novelty score: inverse of max similarity, scaled to 1-10
            # High similarity = low novelty, low similarity = high novelty
            max_similarity = np.max(similarities)
            novelty_score = (1 - max_similarity) * 10
            
            # Ensure score is between 1 and 10
            novelty_score = max(1.0, min(10.0, novelty_score))
            
            return round(novelty_score, 1)
            
        except Exception as e:
            print(f"Error calculating novelty score: {str(e)}")
            return 7.0
    
    async def generate_hypotheses(
        self, 
        gaps: Dict, 
        topic: str,
        papers: List[SearchResult] = None
    ) -> Dict:
        """
        Generate research hypotheses from identified gaps
        
        Args:
            gaps: Dictionary of research gaps from gap analysis
            topic: Research topic
            papers: Optional list of papers for novelty scoring
            
        Returns:
            Dictionary containing generated hypotheses with scores
        """
        
        if not self.client:
            return {
                "error": "OpenAI API not configured",
                "hypotheses": []
            }
        
        print(f"\n💡 Generating research hypotheses for: {topic}")
        
        # Create the prompt
        prompt = self._create_hypothesis_prompt(gaps, topic)
        
        try:
            # Generate hypotheses using GPT-4
            print("Calling GPT-4 to generate hypotheses...")
            response = self.client.chat.completions.create(
                model="gpt-4",  # Use GPT-4 for higher quality
                messages=[
                    {"role": "system", "content": "You are a world-class research strategist and grant proposal expert. Generate detailed, actionable research hypotheses. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher for creativity
                max_tokens=3000  # Enough for 3-5 detailed hypotheses
            )
            
            # Parse the response
            hypotheses_json = json.loads(response.choices[0].message.content)
            
            # Handle both single hypothesis and array formats
            if "hypotheses" in hypotheses_json:
                hypotheses_list = hypotheses_json["hypotheses"]
            elif isinstance(hypotheses_json, list):
                hypotheses_list = hypotheses_json
            else:
                # Wrap single hypothesis in array
                hypotheses_list = [hypotheses_json]
            
            # Calculate novelty scores using embeddings
            if papers and self.embedding_model:
                print("Calculating novelty scores using embeddings...")
                for hyp in hypotheses_list:
                    # Create text for novelty comparison
                    hyp_text = f"{hyp.get('title', '')} {hyp.get('proposed_solution', '')}"
                    novelty = self._calculate_novelty_score(hyp_text, papers)
                    hyp['calculated_novelty_score'] = novelty
                    # Update the novelty score if calculated is significantly different
                    if abs(hyp.get('novelty_score', 7) - novelty) > 2:
                        hyp['novelty_score'] = novelty
            
            print(f"✓ Generated {len(hypotheses_list)} hypotheses")
            
            return {
                "hypotheses": hypotheses_list,
                "count": len(hypotheses_list),
                "topic": topic
            }
            
        except Exception as e:
            print(f"Error generating hypotheses: {str(e)}")
            return {
                "error": str(e),
                "hypotheses": []
            }
    
    def format_hypotheses_for_display(self, hypotheses_data: Dict) -> str:
        """
        Format hypotheses for display in Streamlit
        
        Args:
            hypotheses_data: Dictionary containing hypotheses
            
        Returns:
            Formatted markdown string
        """
        
        if hypotheses_data.get("error"):
            return f"⚠️ Error: {hypotheses_data['error']}"
        
        hypotheses = hypotheses_data.get("hypotheses", [])
        if not hypotheses:
            return "No hypotheses generated"
        
        output = "## 💡 Generated Research Hypotheses\n\n"
        output += f"*Generated {len(hypotheses)} actionable research proposals*\n\n"
        output += "---\n\n"
        
        for i, hyp in enumerate(hypotheses, 1):
            # Header with title and priority
            priority = hyp.get('priority', 'MEDIUM')
            priority_emoji = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
            
            output += f"### {priority_emoji} Hypothesis {i}: {hyp.get('title', 'Untitled')}\n\n"
            
            # Scores display
            novelty = hyp.get('novelty_score', 'N/A')
            impact = hyp.get('impact_score', 'N/A')
            feasibility = hyp.get('feasibility_score', 'N/A')
            
            output += f"**📊 Scores:** Novelty: {novelty}/10 | Impact: {impact}/10 | Feasibility: {feasibility}/10\n\n"
            
            # Gap addressed
            if hyp.get('gap_addressed'):
                output += f"**🎯 Gap Addressed:** {hyp['gap_addressed']}\n\n"
            
            # Problem statement
            if hyp.get('problem_statement'):
                output += f"**❓ Problem Statement:**\n{hyp['problem_statement']}\n\n"
            
            # Proposed solution
            if hyp.get('proposed_solution'):
                output += f"**💡 Proposed Solution:**\n{hyp['proposed_solution']}\n\n"
            
            # Expected impact
            if hyp.get('expected_impact'):
                output += f"**🚀 Expected Impact:**\n{hyp['expected_impact']}\n\n"
            
            # Methodology
            if hyp.get('methodology'):
                output += f"**🔬 Methodology:**\n{hyp['methodology']}\n\n"
            
            # Required resources
            if hyp.get('required_resources'):
                output += f"**📋 Required Resources:**\n"
                resources = hyp['required_resources']
                if isinstance(resources, dict):
                    for key, value in resources.items():
                        output += f"- **{key.title()}:** {value}\n"
                else:
                    output += f"{resources}\n"
                output += "\n"
            
            # Target venue and collaborators
            if hyp.get('target_venue'):
                output += f"**📰 Target Venues:** {hyp['target_venue']}\n\n"
            
            if hyp.get('potential_collaborators'):
                output += f"**🤝 Potential Collaborators:** {hyp['potential_collaborators']}\n\n"
            
            output += "---\n\n"
        
        return output

class Agent:
    """AI agent wrapper for OpenAI ChatGPT model"""
    def __init__(self, name: str, model: str = "gpt-3.5-turbo"):
        self.name = name
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.model = model
            self.chat_history = []
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg)

    async def process(self, message: str) -> str:
        """
        Process a message using OpenAI ChatGPT model
        """
        if not self.client:
            return "Error: Agent not properly initialized"
        try:
            # Add user message to history
            self.chat_history.append({
                "role": "user",
                "content": message
            })
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract response text
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.chat_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        except Exception as e:
            error_msg = f"OpenAI API Error: {str(e)}"
            print(error_msg)
            return error_msg

class MultiAgentSystem:
    """
    System coordinating multiple agents for literature review
    """
    def __init__(self):
        # Initialize OpenAI client for GPT-4 gap analysis fallback
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            print("Warning: OPENAI_API_KEY not found. GPT-4 fallback unavailable.")
            self.client = None
        
        try:
            self.google_agent = Agent("Google_Search_Agent")
        except Exception as e:
            print(f"Failed to initialize Google agent: {str(e)}")
            self.google_agent = None
            
        try:
            self.arxiv_agent = Agent("Arxiv_Search_Agent")
        except Exception as e:
            print(f"Failed to initialize Arxiv agent: {str(e)}")
            self.arxiv_agent = None
            
        try:
            self.clinical_trials_agent = Agent("ClinicalTrials_Search_Agent")
        except Exception as e:
            print(f"Failed to initialize ClinicalTrials agent: {str(e)}")
            self.clinical_trials_agent = None
            
        try:
            self.report_agent = Agent("Report_Agent")
        except Exception as e:
            print(f"Failed to initialize Report agent: {str(e)}")
            self.report_agent = None
            
        self.search_tool = SearchTool()
        
        # Initialize Gap Analysis Engine
        try:
            self.gap_analyzer = GapAnalysisEngine()
        except Exception as e:
            print(f"Failed to initialize Gap Analysis Engine: {str(e)}")
            self.gap_analyzer = None
        
        # Initialize Hypothesis Generator
        try:
            self.hypothesis_generator = HypothesisGenerator()
        except Exception as e:
            print(f"Failed to initialize Hypothesis Generator: {str(e)}")
            self.hypothesis_generator = None

    async def run_gap_analysis(self, topic: str) -> Dict:
        """
        Run gap analysis on research papers for a given topic
        Collects papers and analyzes them for research gaps
        
        Args:
            topic: The research topic to analyze
            
        Returns:
            Dictionary containing gap analysis results and formatted output
        """
        print(f"\n🔍 Starting Gap Analysis for: {topic}")
        
        # Collect papers
        all_papers = await self._collect_papers(topic)
        print(f"Collected {len(all_papers)} papers total for analysis")
        
        # Analyze gaps using the collected papers
        return await self._run_gap_analysis_with_papers(all_papers, topic)
    
    async def _openai_gap_analysis_fallback(self, papers: List[SearchResult], topic: str) -> Dict:
        """
        Use OpenAI GPT-4 for comprehensive gap analysis
        
        Args:
            papers: List of research papers
            topic: Research topic
            
        Returns:
            Dictionary of research gaps
        """
        if not self.client:
            print("OpenAI client not available")
            return {
                "knowledge_gaps": ["OpenAI API key not configured"],
                "methodological_gaps": [],
                "dataset_gaps": [],
                "temporal_gaps": []
            }
        
        try:
            # Create detailed papers summary with MORE context
            papers_text = ""
            for i, paper in enumerate(papers[:15], 1):  # Increased from 10 to 15
                papers_text += f"\n{'='*60}\n"
                papers_text += f"Paper {i}: {paper.title}\n"
                
                if paper.authors:
                    papers_text += f"Authors: {', '.join(paper.authors[:5])}\n"
                
                if paper.published:
                    papers_text += f"Published: {paper.published}\n"
                
                # Use FULL abstract, not truncated
                if paper.abstract:
                    papers_text += f"Abstract: {paper.abstract}\n"  # Full abstract, no truncation
                elif paper.snippet:
                    papers_text += f"Summary: {paper.snippet}\n"
                
                if paper.study_type:
                    papers_text += f"Study Type: {paper.study_type}\n"
                if paper.phase:
                    papers_text += f"Phase: {paper.phase}\n"
            
            prompt = f"""You are a PhD-level research analyst with 20+ years experience identifying critical research gaps. Your job is to DEEPLY analyze these papers and find SPECIFIC, ACTIONABLE gaps that researchers can build upon.

RESEARCH TOPIC: {topic}

PAPERS TO ANALYZE:{papers_text}

{'='*80}

CRITICAL INSTRUCTIONS:
1. READ EVERY PAPER CAREFULLY - Don't just skim
2. FIND AT LEAST 3-5 GAPS PER CATEGORY - Research ALWAYS has gaps!
3. Be SPECIFIC with technical details, numbers, methods, datasets
4. Each gap should be ACTIONABLE - what exactly is missing?
5. Look for:
   - Unstudied combinations of techniques
   - Missing benchmarks or evaluation standards
   - Lack of real-world deployment/application studies
   - Outdated assumptions from older papers
   - Unexplored edge cases or failure modes
   - Missing cross-domain applications
   - Contradictions between papers
   - Limited dataset sizes or diversity
   - Methods that haven't been tried together
   - Temporal aspects (what's changed since publication?)

EXAMPLE OF GOOD GAPS (BE THIS SPECIFIC):
❌ BAD: "More research needed on privacy"
✅ GOOD: "CRITICAL: No empirical studies measuring privacy leakage in federated learning with >1000 heterogeneous devices (IoT + mobile + edge servers) - all existing work uses <100 homogeneous nodes"

❌ BAD: "Missing datasets"
✅ GOOD: "MISSING DATASET: No standardized benchmark for privacy attacks in federated learning with ground truth labels - researchers cannot compare attack success rates across studies"

NOW ANALYZE AND FIND GAPS IN THESE 4 CATEGORIES:

1. KNOWLEDGE GAPS - What questions are NOT answered by these papers?
   - What aspects of {topic} have NOT been studied?
   - What combinations haven't been explored?
   - What assumptions are untested?
   - What edge cases are ignored?

2. METHODOLOGICAL GAPS - What approaches are MISSING?
   - What research methods haven't been tried?
   - What evaluation metrics are absent?
   - What experimental designs are lacking?
   - What theoretical frameworks don't exist?

3. DATASET GAPS - What data is MISSING or INSUFFICIENT?
   - What datasets don't exist but should?
   - What's the coverage (size, diversity, quality)?
   - What domains are underrepresented?
   - What benchmarks are needed?

4. TEMPORAL GAPS - What's OUTDATED?
   - What has changed since these papers were published?
   - What new technologies/methods aren't considered?
   - What recent events/regulations aren't reflected?

YOU MUST FIND AT LEAST 2-3 SPECIFIC GAPS PER CATEGORY!

Respond with ONLY valid JSON (no markdown, no extra text):
{{
  "knowledge_gaps": ["SPECIFIC gap 1 with details", "SPECIFIC gap 2 with details", "SPECIFIC gap 3 with details"],
  "methodological_gaps": ["SPECIFIC gap 1 with details", "SPECIFIC gap 2 with details", "SPECIFIC gap 3 with details"],
  "dataset_gaps": ["SPECIFIC gap 1 with details", "SPECIFIC gap 2 with details"],
  "temporal_gaps": ["SPECIFIC gap 1 with details", "SPECIFIC gap 2 with details"]
}}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a world-class research analyst. You ALWAYS find specific, actionable research gaps. You respond with ONLY valid JSON, no markdown formatting, no extra text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Increased from 0.7 for more creativity
                max_tokens=2500   # Increased from 1500 for more detailed gaps
            )
            
            # Extract JSON from response (handle markdown code blocks if present)
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                # Find the JSON content between ``` markers
                lines = response_text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or (not line.strip().startswith("```")):
                        json_lines.append(line)
                response_text = '\n'.join(json_lines)
            
            gaps_json = json.loads(response_text)
            
            # Validate that we got gaps
            total_gaps = sum(len(gaps_json.get(k, [])) for k in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"])
            print(f"  ✓ GPT-4 extracted {total_gaps} total gaps")
            
            return gaps_json
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Response was: {response_text[:500]}...")
            return {
                "knowledge_gaps": ["JSON parsing failed - please try again"],
                "methodological_gaps": [],
                "dataset_gaps": [],
                "temporal_gaps": []
            }
        except Exception as e:
            print(f"OpenAI fallback error: {str(e)}")
            return {
                "knowledge_gaps": [f"Error: {str(e)}"],
                "methodological_gaps": [],
                "dataset_gaps": [],
                "temporal_gaps": []
            }
    
    async def run_hypothesis_generation(
        self, 
        gaps: Dict, 
        topic: str,
        papers: List[SearchResult] = None
    ) -> Dict:
        """
        Generate research hypotheses from identified gaps
        
        Args:
            gaps: Dictionary of research gaps
            topic: Research topic
            papers: Optional list of papers for novelty scoring
            
        Returns:
            Dictionary containing generated hypotheses
        """
        print(f"\n💡 Starting Hypothesis Generation for: {topic}")
        
        if not self.hypothesis_generator:
            return {
                "error": "Hypothesis generator not available",
                "hypotheses": []
            }
        
        # Generate hypotheses
        result = await self.hypothesis_generator.generate_hypotheses(gaps, topic, papers)
        
        # Format for display
        if result.get("hypotheses"):
            formatted_output = self.hypothesis_generator.format_hypotheses_for_display(result)
            result["formatted_output"] = formatted_output
        
        return result

    async def run_complete_analysis(self, topic: str) -> Dict:
        """
        Run complete research pipeline: Paper Collection → Literature Review → Gap Analysis → Hypothesis Generation
        All phases use the SAME papers for consistency
        
        Args:
            topic: Research topic
            
        Returns:
            Dictionary containing all results (review, gaps, hypotheses)
        """
        print(f"\n🚀 Starting COMPLETE ANALYSIS for: {topic}")
        print("=" * 80)
        
        results = {
            "topic": topic,
            "literature_review": None,
            "gap_analysis": None,
            "hypotheses": None,
            "papers": []
        }
        
        try:
            # Step 1: Collect papers ONCE for all phases
            print("\n📚 PHASE 1/4: Paper Collection")
            all_papers = await self._collect_papers(topic)
            results["papers"] = all_papers
            print(f"✓ Collected {len(all_papers)} papers")
            
            # Step 2: Generate Literature Review using collected papers
            print("\n📖 PHASE 2/4: Literature Review")
            literature_review = await self.run_literature_review(topic, papers=all_papers)
            results["literature_review"] = literature_review
            
            # Step 3: Run Gap Analysis using same papers
            print("\n🔍 PHASE 3/4: Gap Analysis")
            gap_results = await self._run_gap_analysis_with_papers(all_papers, topic)
            results["gap_analysis"] = gap_results
            
            # Step 4: Generate Hypotheses from gaps using same papers
            print("\n💡 PHASE 4/4: Hypothesis Generation")
            if gap_results.get("gaps"):
                hypothesis_results = await self.run_hypothesis_generation(
                    gaps=gap_results["gaps"],
                    topic=topic,
                    papers=all_papers
                )
                results["hypotheses"] = hypothesis_results
            else:
                print("⚠️ No gaps found, skipping hypothesis generation")
                results["hypotheses"] = {
                    "error": "No gaps identified for hypothesis generation",
                    "hypotheses": []
                }
            
            print("\n✅ COMPLETE ANALYSIS FINISHED")
            print("=" * 80)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {str(e)}")
            results["error"] = str(e)
            return results
    
    async def _collect_papers(self, topic: str) -> List[SearchResult]:
        """
        Collect papers from all sources (internal method for reuse)
        
        Args:
            topic: Research topic
            
        Returns:
            List of SearchResult objects
        """
        all_papers = []
        
        # ArXiv Search
        print(f"  → Collecting from arXiv...")
        arxiv_results = self.search_tool.arxiv_search(topic, max_results=7)
        all_papers.extend(arxiv_results)
        
        # Google Search
        print(f"  → Collecting from Google...")
        google_results = self.search_tool.google_search(topic, num_results=5)
        all_papers.extend(google_results)
        
        # Clinical Trials
        print(f"  → Collecting from ClinicalTrials.gov...")
        clinical_results = self.search_tool.clinicaltrials_search_beta_api(topic, max_results=3)
        if not clinical_results:
            clinical_results = self.search_tool.clinicaltrials_search_scrape(topic, max_results=3)
        all_papers.extend(clinical_results)
        
        return all_papers
    
    async def _run_gap_analysis_with_papers(self, papers: List[SearchResult], topic: str) -> Dict:
        """
        Run gap analysis with pre-collected papers (internal method)
        Uses BOTH Flan-T5 and GPT-4, then combines unique gaps
        
        Args:
            papers: Pre-collected papers
            topic: Research topic
            
        Returns:
            Gap analysis results
        """
        print(f"  → Analyzing {len(papers)} papers for research gaps...")
        
        if not papers or len(papers) == 0:
            return {
                "gaps": {
                    "error": "No papers found",
                    "knowledge_gaps": [],
                    "methodological_gaps": [],
                    "dataset_gaps": [],
                    "temporal_gaps": []
                },
                "formatted_output": "⚠️ No papers to analyze",
                "papers_analyzed": 0,
                "papers": []
            }
        
        # Run BOTH models in parallel for redundancy
        flan_t5_gaps = {}
        gpt4_gaps = {}
        
        # Method 1: Flan-T5-Large (local, free)
        if self.gap_analyzer:
            print("  → Running Flan-T5-Large analysis...")
            try:
                flan_t5_gaps = self.gap_analyzer.analyze_gaps(papers, topic)
                total_flan = sum(len(flan_t5_gaps.get(k, [])) for k in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"])
                print(f"  ✓ Flan-T5 found {total_flan} gaps")
            except Exception as e:
                print(f"  ⚠️ Flan-T5 error: {str(e)}")
                flan_t5_gaps = {
                    "knowledge_gaps": [],
                    "methodological_gaps": [],
                    "dataset_gaps": [],
                    "temporal_gaps": []
                }
        
        # Method 2: GPT-4 (always run for reliability)
        print("  → Running GPT-4 analysis...")
        try:
            gpt4_gaps = await self._openai_gap_analysis_fallback(papers, topic)
            total_gpt = sum(len(gpt4_gaps.get(k, [])) for k in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"])
            print(f"  ✓ GPT-4 found {total_gpt} gaps")
        except Exception as e:
            print(f"  ⚠️ GPT-4 error: {str(e)}")
            gpt4_gaps = {
                "knowledge_gaps": [],
                "methodological_gaps": [],
                "dataset_gaps": [],
                "temporal_gaps": []
            }
        
        # Combine and deduplicate gaps from both models
        combined_gaps = self._combine_and_deduplicate_gaps(flan_t5_gaps, gpt4_gaps)
        
        # Determine which model(s) contributed
        total_flan = sum(len(flan_t5_gaps.get(k, [])) for k in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"])
        total_gpt = sum(len(gpt4_gaps.get(k, [])) for k in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"])
        
        if total_flan > 0 and total_gpt > 0:
            combined_gaps["source"] = "Flan-T5-Large + GPT-4 (Combined)"
        elif total_gpt > 0:
            combined_gaps["source"] = "GPT-4"
        elif total_flan > 0:
            combined_gaps["source"] = "Flan-T5-Large"
        else:
            combined_gaps["source"] = "No gaps found"
        
        total_combined = sum(len(combined_gaps.get(k, [])) for k in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"])
        print(f"  ✓ Total unique gaps after combining: {total_combined}")
        
        formatted_output = self.gap_analyzer.format_gaps_for_display(combined_gaps) if self.gap_analyzer else "No formatter available"
        
        return {
            "gaps": combined_gaps,
            "formatted_output": formatted_output,
            "papers_analyzed": len(papers),
            "papers": papers
        }
    
    def _combine_and_deduplicate_gaps(self, gaps1: Dict, gaps2: Dict) -> Dict:
        """
        Combine gaps from two sources and remove duplicates
        
        Args:
            gaps1: First set of gaps (e.g., from Flan-T5)
            gaps2: Second set of gaps (e.g., from GPT-4)
            
        Returns:
            Combined and deduplicated gaps
        """
        combined = {
            "knowledge_gaps": [],
            "methodological_gaps": [],
            "dataset_gaps": [],
            "temporal_gaps": []
        }
        
        for category in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"]:
            # Get gaps from both sources
            gaps_set1 = gaps1.get(category, [])
            gaps_set2 = gaps2.get(category, [])
            
            # Combine them
            all_gaps = gaps_set1 + gaps_set2
            
            # Deduplicate using similarity (case-insensitive, normalized)
            unique_gaps = []
            seen_normalized = set()
            
            for gap in all_gaps:
                # Normalize: lowercase, remove extra spaces, remove punctuation
                normalized = gap.lower().strip()
                normalized = ' '.join(normalized.split())  # Remove extra whitespace
                
                # Check if we've seen something very similar
                is_duplicate = False
                for seen in seen_normalized:
                    # If 80% of words overlap, consider it duplicate
                    gap_words = set(normalized.split())
                    seen_words = set(seen.split())
                    if gap_words and seen_words:
                        overlap = len(gap_words & seen_words) / max(len(gap_words), len(seen_words))
                        if overlap > 0.8:  # 80% similarity threshold
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_gaps.append(gap)
                    seen_normalized.add(normalized)
            
            combined[category] = unique_gaps
        
        return combined
    
    async def run_complete_analysis(self, topic: str) -> Dict:
        """
        Run complete end-to-end analysis: Paper Collection → Literature Review → Gap Analysis → Hypothesis Generation
        All phases use the same set of papers for consistency
        
        Args:
            topic: Research topic
            
        Returns:
            Dict containing all analysis results
        """
        results = {
            "topic": topic,
            "papers": [],
            "literature_review": "",
            "gap_analysis": {},
            "hypotheses": {}
        }
        
        try:
            print("=" * 80)
            print(f"🚀 STARTING COMPLETE ANALYSIS FOR: {topic}")
            print("=" * 80)
            
            # PHASE 1: Collect papers once for all subsequent phases
            print("\n📄 PHASE 1/4: Collecting Papers")
            papers = await self._collect_papers(topic)
            results["papers"] = papers
            print(f"  ✓ Collected {len(papers)} papers total")
            
            # PHASE 2: Literature Review using collected papers
            print("\n📚 PHASE 2/4: Literature Review")
            literature_review = await self.run_literature_review(topic, papers=papers)
            results["literature_review"] = literature_review
            
            # PHASE 3: Gap Analysis using same papers
            print("\n🔍 PHASE 3/4: Gap Analysis")
            gap_results = await self._run_gap_analysis_with_papers(papers, topic)
            results["gap_analysis"] = gap_results
            
            # PHASE 4: Hypothesis Generation from gaps
            print("\n💡 PHASE 4/4: Hypothesis Generation")
            if gap_results.get("gaps"):
                hypothesis_results = await self.run_hypothesis_generation(
                    gaps=gap_results["gaps"],
                    topic=topic,
                    papers=papers
                )
                results["hypotheses"] = hypothesis_results
            else:
                print("⚠️ No gaps found, skipping hypothesis generation")
                results["hypotheses"] = {
                    "error": "No gaps identified for hypothesis generation",
                    "hypotheses": []
                }
            
            print("\n✅ COMPLETE ANALYSIS FINISHED")
            print("=" * 80)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {str(e)}")
            results["error"] = str(e)
            return results

    async def run_literature_review(self, topic: str, papers: List[SearchResult] = None) -> str:
        """
        Run a comprehensive literature review on a given topic
        
        Args:
            topic: Research topic
            papers: Optional pre-collected papers. If None, will collect new papers
            
        Returns:
            Markdown-formatted literature review
        """
        # If no papers provided, collect them
        if papers is None:
            print(f"Collecting papers for literature review on '{topic}'...")
            papers = await self._collect_papers(topic)
        
        print(f"Generating comprehensive literature review from {len(papers)} papers...")
        
        # Build context from papers
        context = f"Research Topic: {topic}\n\n"
        context += "=" * 80 + "\n"
        context += "PAPERS FOR ANALYSIS:\n"
        context += "=" * 80 + "\n\n"
        
        for i, paper in enumerate(papers, 1):
            context += f"Paper {i}: {paper.title}\n"
            if paper.authors:
                context += f"Authors: {', '.join(paper.authors)}\n"
            if paper.published:
                context += f"Published: {paper.published}\n"
            if paper.abstract:
                context += f"Abstract: {paper.abstract[:500]}...\n"
            elif paper.snippet:
                context += f"Content: {paper.snippet}\n"
            if paper.arxiv_url:
                context += f"URL: {paper.arxiv_url}\n"
            elif hasattr(paper, 'link') and paper.link:
                context += f"URL: {paper.link}\n"
            context += "\n"
        
        # Generate literature review using GPT
        print("Generating comprehensive literature review report...")
        
        if not self.report_agent:
            return "Error: Report agent not initialized. Please check your OpenAI API key."
            
        report_prompt = f"""
        Generate a comprehensive literature review on {topic} based on {len(papers)} research papers provided below.

        {context}

        Please provide a well-structured literature review that:
        1. Synthesizes the main findings and themes across ALL papers
        2. Discusses key research directions and trends
        3. Includes proper citations with author names and years
        4. Provides clickable links to the papers when available
        5. Concludes with future research directions based on the analysis
        6. Write in academic style but keep it accessible
        7. Be comprehensive - discuss ALL major papers, not just 2-3
        
        Format the review with clear sections using markdown headers (##).
        """

        report = await self.report_agent.process(report_prompt)
        return report

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def run_literature_review():
        system = MultiAgentSystem()
        topic = input("Enter a medical or scientific topic for literature review: ")
        print(f"\nRunning comprehensive literature review on: {topic}")
        print("This may take a few minutes...\n")
        
        report = await system.run_literature_review(topic)
        print("\n" + "="*80)
        print("LITERATURE REVIEW REPORT")
        print("="*80 + "\n")
        print(report)
        
        # Save report to file
        filename = f"literature_review_{topic.replace(' ', '_')}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to {filename}")
    
    async def test_search():
        search_tool = SearchTool()
        query = input("Enter a medical topic to test search: ")
        
        print("\nTesting ClinicalTrials.gov Beta API search...")
        api_results = search_tool.clinicaltrials_search_beta_api(query, max_results=2)
        for result in api_results:
            print(f"Title: {result.title}")
            print(f"NCT ID: {result.nct_id}")
            print(f"URL: {result.link}")
            print(f"Status: {result.status}")
            print(f"Phase: {result.phase}")
            print(f"Conditions: {result.conditions}")
            print("-" * 50)
        
        print("\nTesting ClinicalTrials.gov scrape search (fallback method)...")
        scrape_results = search_tool.clinicaltrials_search_scrape(query, max_results=2)
        for result in scrape_results:
            print(f"Title: {result.title}")
            print(f"NCT ID: {result.nct_id}")
            print(f"URL: {result.link}")
            print(f"Status: {result.status}")
            print("-" * 50)
    
    # Choose which function to run
    choice = input("Choose an option:\n1. Run full literature review\n2. Test search only\nYour choice (1/2): ")
    if choice == "1":
        asyncio.run(run_literature_review())
    else:
        asyncio.run(test_search())