# 🔬 ResearchForge - Complete Technical Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Workflow](#architecture--workflow)
3. [Technologies & Tools](#technologies--tools)
4. [AI Models & APIs](#ai-models--apis)
5. [Features & Capabilities](#features--capabilities)
6. [Code Structure](#code-structure)
7. [Data Flow](#data-flow)
8. [API Calls & Costs](#api-calls--costs)
9. [Performance Metrics](#performance-metrics)

---

## 🎯 System Overview

**ResearchForge** is an autonomous AI research assistant that automates the entire research discovery pipeline from literature review to experiment design. Built for the Gen AI Hackathon 2025, it combines multiple state-of-the-art AI models to deliver research insights in minutes instead of weeks.

### Core Purpose
- **Input:** Research topic (e.g., "quantum computing for drug discovery")
- **Output:** Complete research analysis with literature review, identified gaps, testable hypotheses, experiment designs, datasets, and publication-ready paper template
- **Time:** 2-3 minutes for complete analysis
- **Cost:** ~$0.52 per full analysis

### Key Differentiators
✅ Multi-model AI orchestration (GPT-4 + GPT-3.5 + Sentence-BERT)  
✅ Real-time paper collection from multiple sources  
✅ Embedding-based novelty scoring (objective measurement)  
✅ Complete experiment design with dataset recommendations  
✅ Automated research paper generation in Word format  

---

## 🏗️ Architecture & Workflow

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI (app.py)                │
│  - User Input Interface                                     │
│  - 5 Tabs: Lit Review | Gap Analysis | Hypotheses |        │
│            Experiments | Paper Writer                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           MULTI-AGENT SYSTEM (multi_agent_system.py)        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Paper        │  │ Literature   │  │ Gap Analysis │     │
│  │ Collector    │→ │ Reviewer     │→ │ Engine       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                 │                  │              │
│         ▼                 ▼                  ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Hypothesis   │  │ Experiment   │  │ Dataset      │     │
│  │ Generator    │→ │ Designer     │→ │ Fetcher      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          RESEARCH PAPER WRITER (paper_writer.py)            │
│  - Title & Abstract Generator                               │
│  - Section-by-Section Writer (8 sections)                   │
│  - Markdown to Word Converter                               │
└─────────────────────────────────────────────────────────────┘
```

### Complete Workflow (6 Phases)

#### **Phase 1: Paper Collection** (10-15 seconds)
```
User Query → arXiv API (7 papers)
           → Google Custom Search (5 papers)
           → ClinicalTrials.gov (3 papers)
           → Total: 10-15 papers collected
```

**Data Sources:**
- **arXiv:** Academic preprints (CS, Physics, Math, Bio)
- **Google Scholar:** Published papers via Custom Search API
- **ClinicalTrials.gov:** Medical research trials

**Collection Strategy:**
- Parallel async requests for speed
- 7-day cache to avoid duplicate API calls
- Metadata extraction: title, authors, abstract, publication date, URLs

#### **Phase 2: Literature Review** (30-40 seconds)
```
Papers → GPT-3.5-turbo → Comprehensive Review
         (Input: abstracts + titles)
         (Output: 700-1000 words)
```

**Process:**
1. Aggregate all paper abstracts
2. Create structured prompt for GPT-3.5
3. Generate 4-5 section review:
   - Introduction to field
   - Current methodologies
   - Key findings
   - Research trends
   - Identified limitations

**Model:** `gpt-3.5-turbo` (fast, cost-effective)  
**Token Limit:** 2000 output tokens  
**Temperature:** 0.7 (balanced creativity)

#### **Phase 3: Gap Analysis** (40-50 seconds)
```
Papers + Abstracts → GPT-4 → Identified Gaps
                      (Detailed analysis)
                      (4 gap categories)
```

**Gap Categories:**
1. **Knowledge Gaps** - Unknown/unexplored areas
2. **Methodological Gaps** - Missing techniques/approaches
3. **Dataset Gaps** - Lack of data or benchmarks
4. **Temporal Gaps** - Outdated research needing updates

**Process:**
1. Extract full abstracts (no truncation)
2. Analyze up to 15 papers with GPT-4
3. Aggressive prompting for comprehensive gap identification
4. Structured JSON output with categorized gaps

**Model:** `gpt-4` (high reasoning quality)  
**Token Limit:** 2500 output tokens  
**Temperature:** 0.8 (creative gap discovery)

**Note:** Originally used Flan-T5-Large (780M params), but it consistently returned 0 gaps due to model limitations. Switched to GPT-4-only for reliable results.

#### **Phase 4: Hypothesis Generation** (40-50 seconds)
```
Gaps + Papers → GPT-4 → 5 Hypotheses
                         (with scoring)
                         
Hypotheses → Sentence-BERT → Novelty Scores
             (embedding comparison)
             
All Scores → Weighted Formula → Overall Scores
             (sorted by quality)
```

**Hypothesis Components:**
- `hypothesis_id`: Unique identifier
- `title`: Concise hypothesis name (5-8 words)
- `problem_statement`: What problem it solves
- `proposed_solution`: How to address the problem
- `gap_addressed`: Which research gap it fills
- `expected_impact`: Potential outcomes
- `novelty_score`: 1-10 (embedding-based)
- `impact_score`: 1-10 (GPT-4 assessment)
- `feasibility_score`: 1-10 (GPT-4 assessment)
- `overall_score`: Weighted average
- `priority`: HIGH/MEDIUM (based on overall score)

**Scoring System:**
```
Overall Score = (Novelty × 0.4) + (Impact × 0.4) + (Feasibility × 0.2)

Novelty Score: Calculated via Sentence-BERT embeddings
               - Compare hypothesis to existing papers
               - Cosine similarity → 1-10 scale
               - Lower similarity = Higher novelty
               
Impact Score: GPT-4 assessment of potential impact
Feasibility Score: GPT-4 assessment of implementation ease
```

**Default Scores:** (if GPT-4 doesn't provide)
- Impact: 8/10
- Feasibility: 7/10
- Novelty: 7/10

**Sorting:** Hypotheses sorted by `overall_score` (descending) - Best hypotheses appear first

**Model:** `gpt-4` + `sentence-transformers/all-MiniLM-L6-v2`  
**Token Limit:** 4000 output tokens  
**Temperature:** 0.8 (creative hypothesis generation)

#### **Phase 5: Experiment Design** (50-60 seconds)
```
Selected Hypothesis → GPT-4 → 3 Experiment Plans
                               
Research Topic → Kaggle API → Dataset Recommendations
              → HuggingFace API → Alternative Datasets
                               
Experiments → GPT-4 → Metrics, Architectures, Baselines
```

**Experiment Components:**
Each experiment includes:
1. **Introduction & Background** - Context and motivation
2. **Detailed Methodology** - Step-by-step procedures (8-12 steps)
3. **Datasets** - Verified datasets from Kaggle/HuggingFace
4. **Evaluation Metrics** - 4-6 metrics with descriptions
5. **Model Architectures** - 3-4 suggested architectures
6. **Baseline Models** - Comparison models
7. **Expected Outcomes** - Performance targets
8. **Challenges & Solutions** - Potential issues
9. **Required Resources** - Compute, time, tools
10. **Difficulty Level** - Easy/Medium/Hard
11. **Estimated Time** - 2-3 months typical

**Dataset Fetcher:**
- **Kaggle API:** Searches datasets by topic + keywords
- **HuggingFace API:** Searches datasets hub
- **Verification:** Checks dataset size, downloads, usability scores
- **Output:** Top 2-3 datasets with direct links

**Model:** `gpt-4` (experiment design)  
**Token Limit:** 3500 output tokens  
**Temperature:** 0.7 (balanced creativity)

#### **Phase 6: Research Paper Generation** (60-90 seconds)
```
Hypothesis + Experiment + Papers → GPT-4 → Complete Paper
                                            (8 sections)
                                            
Markdown Content → python-docx → Word Document (.docx)
```

**Paper Sections:**
1. **Title** (10-12 words) - Declarative, professional
2. **Abstract** (~200 words) - Background, methods, expected results
3. **Introduction** (~550 words, 4 paragraphs) - Context, problem, objectives
4. **Literature Review** (~700 words, 4-5 paragraphs) - From collected papers
5. **Methodology** (~800 words, 5-6 paragraphs) - From selected experiment
6. **Expected Results** (~550 words, 4 paragraphs) - Anticipated outcomes
7. **Discussion** (~600 words, 4-5 paragraphs) - Interpretation, implications
8. **Conclusion** (~300 words, 3 paragraphs) - Summary, future work
9. **References** - Formatted citations (APA style)

**Total Length:** ~3,500 words (4-5 pages)  
**Format:** Markdown → Word .docx  
**Model:** `gpt-4` (section generation), `gpt-3.5-turbo` (fast sections)  

---

## 🛠️ Technologies & Tools

### Frontend
- **Streamlit 1.31.0** - Web UI framework
  - Tabs navigation
  - Progress bars
  - File downloads
  - Session state management
  - Custom CSS/HTML styling

### Backend
- **Python 3.11+** - Core language
- **AsyncIO** - Asynchronous operations for paper collection
- **aiohttp 3.9.3** - Async HTTP requests

### AI/ML Libraries
```
transformers >= 4.41.0    # Hugging Face models (Flan-T5)
torch >= 2.1.0            # PyTorch for model inference
sentence-transformers >= 5.0.0  # Sentence embeddings
scikit-learn >= 1.7.0     # Cosine similarity
scipy >= 1.16.0           # Scientific computing
sentencepiece == 0.1.99   # Tokenization
```

### API Clients
```
openai == 1.12.0          # OpenAI GPT-4 & GPT-3.5
arxiv == 2.0.0            # arXiv paper search
kaggle == 1.6.6           # Kaggle dataset API
httpx == 0.24.1           # HTTP client (compatible with OpenAI)
requests == 2.31.0        # HTTP library
```

### Data Processing
```
pandas == 2.2.0           # Data manipulation
beautifulsoup4 == 4.12.3  # HTML parsing (Google results)
python-dotenv == 1.0.1    # Environment variables
```

### Document Generation
```
python-docx == 1.1.0      # Word document creation
```

---

## 🤖 AI Models & APIs

### Primary AI Models

#### 1. **OpenAI GPT-4** (Primary Reasoning Engine)
```
Model: gpt-4
Context Window: 8,192 tokens
Output Limit: 2,500-4,000 tokens (varies by task)
Temperature: 0.7-0.8
Cost: $0.03/1K input tokens, $0.06/1K output tokens
```

**Used For:**
- Gap Analysis (main model)
- Hypothesis Generation (main model)
- Experiment Design
- Research Paper Writing (key sections)

**Why GPT-4:**
- Superior reasoning capabilities
- Better structured output
- More reliable JSON formatting
- Higher quality gap identification
- Realistic scoring (7-10 range)

#### 2. **OpenAI GPT-3.5-Turbo** (Fast Generation)
```
Model: gpt-3.5-turbo
Context Window: 4,096 tokens
Output Limit: 2,000 tokens
Temperature: 0.7
Cost: $0.0015/1K input tokens, $0.002/1K output tokens
```

**Used For:**
- Literature Review generation (fast, cost-effective)
- Research paper sections (non-critical parts)

**Why GPT-3.5:**
- 20x cheaper than GPT-4
- Sufficient for literature summarization
- Fast response times
- Good for routine text generation

#### 3. **Google Flan-T5-Large** (Local Model - Deprecated)
```
Model: google/flan-t5-large
Parameters: 780 million
Size: 3.13 GB
Device: CPU/CUDA
Status: Commented out (not used)
```

**Original Purpose:** Gap analysis  
**Issue:** Consistently returned 0 gaps (model limitation)  
**Current Status:** Code present but inactive, system uses GPT-4 only  

**Decision:** Keep code for future potential improvements, but rely on GPT-4 for production

#### 4. **Sentence-BERT** (Embedding Model)
```
Model: sentence-transformers/all-MiniLM-L6-v2
Size: 80 MB
Embedding Dimension: 384
Purpose: Novelty scoring
```

**Used For:**
- Generating embeddings for hypotheses
- Generating embeddings for paper abstracts
- Calculating cosine similarity
- Objective novelty measurement

**Process:**
1. Encode hypothesis text → 384-dim vector
2. Encode paper abstracts → 384-dim vectors
3. Calculate cosine similarity
4. Invert similarity → novelty score (1-10)

**Advantages:**
- Objective measurement (no AI bias)
- Fast computation
- Small model size
- Reliable results

### API Services

#### 1. **arXiv API**
```
Endpoint: http://export.arxiv.org/api/query
Rate Limit: 1 request per 3 seconds (respected)
Cost: Free
Max Results: 7 papers per query
```

**Data Retrieved:**
- Title
- Authors
- Abstract (full text)
- Publication date
- Categories
- arXiv URL
- PDF URL

#### 2. **Google Custom Search API**
```
Endpoint: https://www.googleapis.com/customsearch/v1
Rate Limit: 100 queries/day (free tier)
Cost: Free (100/day), $5/1000 queries after
Max Results: 5 papers per query
```

**Configuration:**
- Custom Search Engine ID required
- Focused on academic/research results
- Extracts: title, link, snippet

#### 3. **ClinicalTrials.gov API**
```
Endpoint: https://clinicaltrials.gov/api/query/full_studies
Rate Limit: No strict limit
Cost: Free
Max Results: 3 trials per query
```

**Data Retrieved:**
- Trial title
- Brief summary
- Detailed description
- Study type
- Status
- Start date
- NCT number

#### 4. **Kaggle API**
```
Authentication: kaggle.json credentials
Rate Limit: None documented
Cost: Free
Max Results: Top 3 datasets per query
```

**Dataset Information:**
- Dataset name
- Owner
- Size
- Download count
- Usability score
- Direct download URL

#### 5. **HuggingFace Hub API**
```
Endpoint: https://huggingface.co/api/datasets
Authentication: HF_TOKEN (optional)
Cost: Free
Max Results: Top 3 datasets per query
```

**Dataset Information:**
- Dataset name
- Description
- Downloads
- Tags
- Direct URL

---

## ✨ Features & Capabilities

### 1. **Topic Input**
- Users enter a research topic or question to initiate the automated research workflow.
- Serves as the starting point for the entire AI-driven research process.

**TOOLS:** Streamlit UI

---

### 2. **Paper Retrieval & Literature Review**
- Automatically searches ArXiv, Google Scholar, and other sources for relevant papers.
- Summarizes papers and extracts key insights, trends, and findings for easy analysis.

**TOOLS:** arXiv API, Google Custom Search API, ClinicalTrials.gov API, GPT-3.5-turbo

---

### 3. **Gap Analysis**
- Analyzes collected literature to identify missing methods, datasets, or unexplored areas.
- Highlights opportunities for novel research contributions.

**TOOLS:** GPT-4

---

### 4. **Hypothesis Generation & Novelty Scoring**
- Converts identified gaps into actionable research hypotheses.
- Assigns novelty and originality scores to prioritize promising ideas.

**TOOLS:** GPT-4, Sentence-BERT (all-MiniLM-L6-v2)

---

### 5. **Experiment Design**
- Suggests step-by-step experiment plans, including methodology, metrics, and datasets.
- Helps researchers quickly move from hypothesis to practical testing.

**TOOLS:** GPT-4, Kaggle API, HuggingFace API

---

### 6. **AI Research Paper Draft**
- Combines all findings into a structured, publication-ready draft.
- Covers abstract, methodology, results, and discussion based on generated insights.

**TOOLS:** GPT-4, GPT-3.5-turbo, python-docx

---

## 📂 Code Structure

### File Breakdown

#### **app.py** (1,257 lines)
```python
Main Application Entry Point

Key Functions:
- init_session_state()           # Initialize session variables
- get_score_badge()              # Generate colored badges
- parse_literature_review()      # Split review into sections
- convert_markdown_to_docx()     # Markdown → Word conversion
- run_async_operations()         # Async complete analysis
- main()                         # Main UI logic
- display_experiment_results()   # Show experiment details
- render_footer()                # Team credits footer

UI Components:
- Title & subtitle with gradient background
- Sidebar configuration panel
- Search input interface
- Progress bar with 4 phases
- 5 tabs with complete results
- Download buttons (TXT, CSV, JSON, DOCX)
- Hypothesis comparison table
- Score badges (color-coded)
- Collapsible sections
```

#### **multi_agent_system.py** (2,896 lines)
```python
Core Multi-Agent Research System

Classes:
1. SearchResult                  # Data class for papers
2. DatasetFetcher               # Kaggle + HuggingFace API
3. PaperCollectionAgent         # Collect from 3 sources
4. LiteratureReviewAgent        # GPT-3.5 review generation
5. GapAnalysisEngine            # GPT-4 gap analysis
6. HypothesisGenerator          # GPT-4 + embeddings
7. ExperimentGenerator          # GPT-4 experiment design
8. MultiAgentSystem             # Orchestrator

Key Methods:

PaperCollectionAgent:
- search_arxiv()                # Fetch arXiv papers
- search_google()               # Google Custom Search
- search_clinical_trials()      # ClinicalTrials.gov
- collect_papers()              # Parallel collection

LiteratureReviewAgent:
- generate_review()             # GPT-3.5 review

GapAnalysisEngine:
- analyze_gaps()                # Main gap analysis
- _analyze_with_openai()        # GPT-4 gap finder
- _format_gap_output()          # Pretty formatting

HypothesisGenerator:
- generate_hypotheses()         # Main generation
- _calculate_novelty_score()    # Sentence-BERT embeddings
- _create_hypothesis_prompt()   # Structured prompt

ExperimentGenerator:
- generate_experiments()        # 3 experiment plans
- _fetch_kaggle_datasets()      # Kaggle API
- _fetch_hf_datasets()          # HuggingFace API

MultiAgentSystem:
- run_complete_analysis()       # Full 6-phase pipeline
- generate_experiments_for_hypothesis()  # Experiment design
```

#### **paper_writer.py** (644 lines)
```python
Research Paper Generator

Class: ResearchPaperWriter

Section Generators:
- _prompt_title()               # Title prompt
- _prompt_abstract()            # Abstract prompt
- _prompt_introduction()        # Introduction prompt
- _prompt_literature()          # Literature review prompt
- _prompt_methodology()         # Methodology prompt
- _prompt_results()             # Results prompt
- _prompt_discussion()          # Discussion prompt
- _prompt_conclusion()          # Conclusion prompt

Main Methods:
- generate_section()            # Generate one section
- generate_complete_paper()     # All 8 sections
- _format_references()          # Format paper citations
```

#### **requirements.txt** (17 dependencies)
```
Core:
- streamlit==1.31.0
- python-dotenv==1.0.1

AI/ML:
- openai==1.12.0
- transformers>=4.41.0
- torch>=2.1.0
- sentence-transformers>=5.0.0
- sentencepiece==0.1.99
- scikit-learn>=1.7.0
- scipy>=1.16.0

Data Collection:
- arxiv==2.0.0
- beautifulsoup4==4.12.3
- requests==2.31.0
- aiohttp==3.9.3
- httpx==0.24.1

Data Processing:
- pandas==2.2.0
- kaggle==1.6.6

Document Generation:
- python-docx==1.1.0
```

---

## 🔄 Data Flow

### Complete Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ USER INPUT: "quantum computing for drug discovery"          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PAPER COLLECTION (Parallel Async)                  │
├─────────────────────────────────────────────────────────────┤
│ arXiv API         → 7 papers  [3-5 sec]                     │
│ Google Search     → 5 papers  [2-3 sec]                     │
│ ClinicalTrials    → 3 papers  [2-3 sec]                     │
│ ────────────────────────────────────                         │
│ Total: 10-15 papers collected                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: LITERATURE REVIEW                                  │
├─────────────────────────────────────────────────────────────┤
│ Input:  Paper abstracts (aggregated)                        │
│ Model:  GPT-3.5-turbo                                       │
│ Tokens: ~1500 input, 2000 output                           │
│ Time:   30-40 seconds                                       │
│ Output: 700-1000 word review (4-5 sections)                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: GAP ANALYSIS                                       │
├─────────────────────────────────────────────────────────────┤
│ Input:  15 papers (full abstracts)                          │
│ Model:  GPT-4                                               │
│ Tokens: ~2000 input, 2500 output                           │
│ Temp:   0.8 (creative gap discovery)                        │
│ Time:   40-50 seconds                                       │
│ Output: 4 gap categories with detailed descriptions         │
│         - Knowledge Gaps                                    │
│         - Methodological Gaps                               │
│         - Dataset Gaps                                      │
│         - Temporal Gaps                                     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: HYPOTHESIS GENERATION                              │
├─────────────────────────────────────────────────────────────┤
│ Step 1: GPT-4 generates 5 hypotheses                        │
│   Input:  Gaps + topic                                      │
│   Tokens: ~1000 input, 4000 output                          │
│   Temp:   0.8 (creative)                                    │
│   Time:   30-35 seconds                                     │
│                                                              │
│ Step 2: Sentence-BERT novelty scoring                       │
│   Model:  all-MiniLM-L6-v2                                  │
│   Method: Cosine similarity with papers                     │
│   Output: Novelty scores (1-10)                             │
│   Time:   5-10 seconds                                      │
│                                                              │
│ Step 3: Overall score calculation                           │
│   Formula: (Novelty×0.4) + (Impact×0.4) + (Feasibility×0.2)│
│   Sorting: Descending by overall_score                      │
│                                                              │
│ Final: 5 ranked hypotheses with complete metadata           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: EXPERIMENT DESIGN (User selects hypothesis)        │
├─────────────────────────────────────────────────────────────┤
│ Step 1: Generate 3 experiments (GPT-4)                      │
│   Input:  Hypothesis + topic                                │
│   Tokens: ~1200 input, 3500 output                          │
│   Temp:   0.7                                               │
│   Time:   35-40 seconds                                     │
│                                                              │
│ Step 2: Fetch datasets (Parallel)                           │
│   Kaggle API    → Top 3 datasets [5-7 sec]                 │
│   HuggingFace   → Top 3 datasets [3-5 sec]                 │
│                                                              │
│ Step 3: Enrich experiments                                  │
│   - Add metrics (4-6 per experiment)                        │
│   - Add architectures (3-4 models)                          │
│   - Add baselines                                           │
│   - Add challenges & solutions                              │
│                                                              │
│ Final: 3 complete experiment plans with datasets            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: PAPER GENERATION (User selects experiment)         │
├─────────────────────────────────────────────────────────────┤
│ Section-by-Section Generation:                              │
│                                                              │
│ 1. Title (GPT-4, 50 tokens)                   [3 sec]      │
│ 2. Abstract (GPT-4, 300 tokens)               [5 sec]      │
│ 3. Introduction (GPT-4, 800 tokens)           [12 sec]     │
│ 4. Literature Review (GPT-3.5, 1000 tokens)   [8 sec]      │
│ 5. Methodology (GPT-4, 1200 tokens)           [15 sec]     │
│ 6. Expected Results (GPT-4, 800 tokens)       [10 sec]     │
│ 7. Discussion (GPT-4, 900 tokens)             [12 sec]     │
│ 8. Conclusion (GPT-4, 500 tokens)             [7 sec]      │
│ 9. References (Python formatting)             [1 sec]      │
│                                                              │
│ Total Time: ~75 seconds                                     │
│ Total Tokens: ~5,550 output                                 │
│                                                              │
│ Conversion: Markdown → Word .docx                           │
│                                                              │
│ Final: 3,500-word paper in Word format                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 💰 API Calls & Costs

### Detailed Cost Breakdown

#### **Per Complete Analysis:**

| Phase | API/Model | Input Tokens | Output Tokens | Calls | Cost |
|-------|-----------|--------------|---------------|-------|------|
| **Paper Collection** | arXiv API | - | - | 1 | $0.00 |
| | Google Custom Search | - | - | 1 | $0.00* |
| | ClinicalTrials.gov | - | - | 1 | $0.00 |
| **Literature Review** | GPT-3.5-turbo | ~1,500 | ~2,000 | 1 | $0.02 |
| **Gap Analysis** | GPT-4 | ~2,000 | ~2,500 | 1 | $0.21 |
| **Hypothesis Generation** | GPT-4 | ~1,000 | ~4,000 | 1 | $0.27 |
| | Sentence-BERT | - | - | Local | $0.00 |
| **Total (Phases 1-4)** | | | | | **$0.50** |

*Free tier: 100 queries/day

#### **Per Experiment Design:**
| Phase | API/Model | Input Tokens | Output Tokens | Calls | Cost |
|-------|-----------|--------------|---------------|-------|------|
| Experiment Generation | GPT-4 | ~1,200 | ~3,500 | 1 | $0.25 |
| Kaggle Datasets | Kaggle API | - | - | 1 | $0.00 |
| HuggingFace Datasets | HF API | - | - | 1 | $0.00 |
| **Total** | | | | | **$0.25** |

#### **Per Research Paper:**
| Section | Model | Output Tokens | Cost |
|---------|-------|---------------|------|
| Title | GPT-4 | ~50 | $0.003 |
| Abstract | GPT-4 | ~300 | $0.018 |
| Introduction | GPT-4 | ~800 | $0.048 |
| Literature Review | GPT-3.5 | ~1,000 | $0.002 |
| Methodology | GPT-4 | ~1,200 | $0.072 |
| Results | GPT-4 | ~800 | $0.048 |
| Discussion | GPT-4 | ~900 | $0.054 |
| Conclusion | GPT-4 | ~500 | $0.030 |
| **Total** | | ~5,550 | **$0.28** |

### Total Cost Summary

| Workflow | Cost |
|----------|------|
| Complete Analysis Only (Phases 1-4) | $0.50 |
| + Experiment Design | $0.75 |
| + Research Paper | $1.03 |
| **Full Pipeline (All Features)** | **$1.03** |

### Rate Limits

| Service | Free Tier | Paid Tier | Current Usage |
|---------|-----------|-----------|---------------|
| Google Custom Search | 100/day | $5/1000 queries | 1 per analysis |
| OpenAI GPT-4 | Pay-as-you-go | No limit (rate limited) | 2-4 calls per analysis |
| OpenAI GPT-3.5 | Pay-as-you-go | No limit (rate limited) | 1-2 calls per analysis |
| arXiv API | 1/3 seconds | - | 1 per analysis |
| Kaggle API | Unlimited | - | 1 per experiment |
| HuggingFace API | Unlimited | - | 1 per experiment |

---

## ⚡ Performance Metrics

### Execution Times

| Phase | Time | Breakdown |
|-------|------|-----------|
| Paper Collection | 10-15s | arXiv (5s) + Google (4s) + Clinical (3s) |
| Literature Review | 30-40s | GPT-3.5 inference |
| Gap Analysis | 40-50s | GPT-4 inference |
| Hypothesis Generation | 40-50s | GPT-4 (35s) + Embeddings (10s) |
| **Total (Main Analysis)** | **120-155s** | **~2.5 minutes** |
| Experiment Design | 50-60s | GPT-4 + API calls |
| Research Paper | 75-90s | 8 GPT-4 calls + conversion |

### Resource Usage

| Resource | Requirement | Peak Usage |
|----------|-------------|------------|
| RAM | 16GB min | ~8GB (Sentence-BERT loaded) |
| CPU | 4 cores | ~60% during inference |
| GPU | Optional | Not utilized (CPU-only) |
| Storage | 5GB | ~3.2GB (model cache) |
| Network | Stable connection | ~10MB per analysis |

### Model Loading Times

| Model | Size | First Load | Cached |
|-------|------|------------|--------|
| Flan-T5-Large | 3.13GB | 60-90s | Instant |
| Sentence-BERT | 80MB | 5-10s | Instant |
| Total | 3.21GB | ~100s | Instant |

**Note:** Flan-T5 code present but not executed (GPT-4 only mode)

### Accuracy & Quality

| Metric | Target | Achieved |
|--------|--------|----------|
| Paper Collection Success Rate | >90% | ~95% |
| Gap Analysis Quality | High | High (GPT-4) |
| Hypothesis Novelty Accuracy | Objective | High (embedding-based) |
| Experiment Feasibility | Realistic | High (GPT-4 assessment) |
| Paper Coherence | Professional | High (GPT-4 writing) |

---

## 🔐 Security & Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=sk-proj-...          # Required
GOOGLE_API_KEY=AIzaSy...            # Required
GOOGLE_SEARCH_ENGINE_ID=...         # Required
HF_TOKEN=hf_...                     # Optional (faster downloads)
KAGGLE_USERNAME=...                 # Optional (dataset verification)
KAGGLE_KEY=...                      # Optional (dataset verification)
```

### Data Privacy
- ✅ No user data stored permanently
- ✅ Session-based storage only
- ✅ API keys in .env (gitignored)
- ✅ No logging of queries
- ✅ No external data transmission (except APIs)

### Caching Strategy
- ✅ Paper collection: 7-day cache (avoid duplicate API calls)
- ✅ AI models: Permanent cache (~/.cache/huggingface/)
- ✅ Session state: Streamlit session (cleared on restart)

---

## 📊 Output Examples

### Sample Hypothesis Output
```json
{
  "hypothesis_id": "H1",
  "title": "Privacy-Preserving Quantum Federated Learning",
  "problem_statement": "Current federated learning lacks quantum-resistant encryption",
  "proposed_solution": "Integrate post-quantum cryptography with federated training",
  "gap_addressed": "Methodological gap in quantum-safe FL protocols",
  "expected_impact": "Secure FL for quantum era, 10x privacy enhancement",
  "novelty_score": 8.7,
  "impact_score": 9.2,
  "feasibility_score": 7.5,
  "overall_score": 8.5,
  "priority": "HIGH"
}
```

### Sample Gap Analysis Output
```markdown
## 🔍 Research Gap Analysis

**Papers Analyzed:** 15  
**Model Used:** GPT-4  

### 📚 Knowledge Gaps
1. Limited understanding of quantum noise effects in federated settings
2. Lack of theoretical bounds for quantum FL convergence
3. Unknown scalability limits for 100+ quantum nodes

### 🔬 Methodological Gaps
1. No standardized benchmarks for quantum federated learning
2. Missing quantum-safe aggregation protocols
3. Inadequate evaluation metrics for quantum privacy

### 💾 Dataset Gaps
1. Absence of quantum-generated synthetic datasets
2. Limited public quantum circuit datasets for benchmarking
3. No standardized quantum FL test beds

### ⏰ Temporal Gaps
1. Pre-2023 quantum algorithms need updating for NISQ era
2. Outdated classical FL security assumptions
3. Privacy techniques not adapted for quantum threats
```

---

## 🎓 Team & Development

**Development Team:**
- Radia Riaz
- Amna Alvie
- Emaan Riaz
- Maheen Alvie

**Event:** Gen AI Hackathon 2025  
**Repository:** https://github.com/Radia-987/ResearchForge  
**Branch:** main  

---

## 📝 Summary

ResearchForge is a comprehensive autonomous AI research assistant that combines:
- **3 AI Models** (GPT-4, GPT-3.5, Sentence-BERT)
- **6 APIs** (OpenAI, arXiv, Google, ClinicalTrials, Kaggle, HuggingFace)
- **6 Automated Phases** (Collection → Review → Gaps → Hypotheses → Experiments → Paper)
- **~2.5 minutes** complete analysis time
- **~$0.50-$1.03** cost per complete workflow
- **3,500-word** publication-ready paper output

Built with Python, Streamlit, and state-of-the-art AI models to accelerate research discovery from weeks to minutes.

---

*Last Updated: November 23, 2025*  
*Documentation Version: 1.0*
