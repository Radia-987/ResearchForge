"""
AI Research Paper Co-Writer
Generates editable draft research papers section by section
"""

import os
from openai import OpenAI
from typing import Dict, List
import re
from datetime import datetime


class ResearchPaperWriter:
    """
    AI-powered research paper generator with section-by-section editing
    """
    
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def generate_section(self, section_name: str, context: Dict, custom_prompt: str = "") -> str:
        """
        Generate a specific section of the research paper
        
        Args:
            section_name: Name of section (title, abstract, introduction, etc.)
            context: Dictionary with all analysis data
            custom_prompt: Additional user instructions for generation
            
        Returns:
            Generated section text
        """
        if not self.client:
            return "Error: OpenAI API key not configured"
        
        # Get section-specific prompt
        prompt_func = {
            "title": self._prompt_title,
            "abstract": self._prompt_abstract,
            "introduction": self._prompt_introduction,
            "literature_review": self._prompt_literature,
            "methodology": self._prompt_methodology,
            "expected_results": self._prompt_results,
            "discussion": self._prompt_discussion,
            "conclusion": self._prompt_conclusion,
        }
        
        prompt_generator = prompt_func.get(section_name)
        if not prompt_generator:
            return f"Unknown section: {section_name}"
        
        prompt = prompt_generator(context)
        
        if custom_prompt:
            prompt += f"\n\nAdditional instructions: {custom_prompt}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert academic research paper writer. Write clear, professional academic prose with proper structure and citations. Use Markdown formatting for sections, emphasis, and lists."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating {section_name}: {str(e)}"
    
    def _prompt_title(self, context: Dict) -> str:
        """Generate prompt for paper title"""
        hypothesis = context.get('hypothesis', {})
        
        return f"""Generate a professional academic paper title for this research:

**Research Topic:** {context.get('topic', 'Unknown')}
**Hypothesis:** {hypothesis.get('title', 'Unknown')}
**Problem:** {hypothesis.get('problem_statement', 'Unknown')}

Requirements:
- 10-15 words maximum
- Clear and descriptive
- Include key methodology or approach if relevant
- Professional academic tone
- No questions, use declarative form

Return ONLY the title text, nothing else."""

    def _prompt_abstract(self, context: Dict) -> str:
        """Generate prompt for abstract"""
        hypothesis = context.get('hypothesis', {})
        experiment = context.get('experiment', {})
        
        exp_title = experiment.get('title', 'Not specified') if experiment else 'Not specified'
        
        return f"""Write a 200-250 word abstract for this research paper:

**Topic:** {context.get('topic', '')}
**Hypothesis:** {hypothesis.get('title', '')}
**Problem Statement:** {hypothesis.get('problem_statement', '')}
**Proposed Solution:** {hypothesis.get('proposed_solution', '')}
**Expected Impact:** {hypothesis.get('expected_impact', '')}
**Experiment:** {exp_title}

Structure the abstract with these elements:
1. **Background** (2 sentences): Context and importance
2. **Problem/Gap** (2 sentences): What's missing in current research
3. **Proposed Approach** (3 sentences): Your methodology and innovation
4. **Expected Contributions** (2 sentences): Anticipated impact and significance

Write in academic style. Use past tense as if research is complete (e.g., "We propose..." "We demonstrate...")."""

    def _prompt_introduction(self, context: Dict) -> str:
        """Generate prompt for introduction"""
        hypothesis = context.get('hypothesis', {})
        papers = context.get('papers', [])
        
        papers_sample = "\n".join([f"- {self._format_paper_cite(p)}" for p in papers[:8]])
        
        return f"""Write a comprehensive Introduction section (4-5 paragraphs, ~600-800 words):

**Research Topic:** {context.get('topic', '')}
**Hypothesis:** {hypothesis.get('title', '')}
**Problem Statement:** {hypothesis.get('problem_statement', '')}
**Gap Addressed:** {hypothesis.get('gap_addressed', '')}

**Related Papers to Reference:**
{papers_sample}

Structure:
### 1. Background (Paragraph 1)
- Broad context of the research area
- Why this topic is important and timely
- Real-world applications or impact

### 2. Current State (Paragraph 2)
- Overview of existing approaches
- Brief mention of related work (cite papers using format: "Author et al. (2023)")
- What has been achieved so far

### 3. Research Gap (Paragraph 3)
- Specific limitations in current approaches
- What is missing or underexplored
- Why this gap matters

### 4. Our Contribution (Paragraph 4)
- What this paper proposes
- Novel aspects of the approach
- Expected contributions

### 5. Paper Organization (Final paragraph)
- Brief outline: "Section 2 reviews..., Section 3 presents..., Section 4 discusses..."

Use academic language, include in-text citations like (Author et al., 2023), and maintain professional tone."""

    def _prompt_literature(self, context: Dict) -> str:
        """Generate prompt for literature review"""
        papers = context.get('papers', [])
        lit_review = context.get('literature_review', '')
        
        papers_list = "\n".join([f"{i+1}. {self._format_paper_cite(p)}" for i, p in enumerate(papers[:12])])
        
        return f"""Write a comprehensive Literature Review section (5-6 paragraphs, ~1000 words):

**Topic:** {context.get('topic', '')}

**Papers to Synthesize:**
{papers_list}

**Existing Analysis:**
{lit_review[:600]}

Structure:
### 2. Literature Review

**Paragraph 1: Overview**
- Introduce the research landscape
- Explain how you'll organize the review

**Paragraph 2-4: Thematic Areas** (organize by themes, NOT paper-by-paper)
- Theme 1: [First major approach/area] - discuss relevant papers
- Theme 2: [Second major approach/area] - discuss relevant papers  
- Theme 3: [Recent advances/emerging trends] - discuss latest work

**Paragraph 5: Synthesis and Gap**
- Compare and contrast approaches
- Identify patterns and trends
- Clearly articulate what's missing

**Paragraph 6: Transition**
- Connect gaps to your proposed work
- Set up methodology section

Important:
- Synthesize thematically, don't just list papers
- Include citations like (Author et al., 2023)
- Compare and contrast different approaches
- Use academic transition phrases
- Be critical but fair"""

    def _prompt_methodology(self, context: Dict) -> str:
        """Generate prompt for methodology"""
        hypothesis = context.get('hypothesis', {})
        experiment = context.get('experiment', {})
        
        # Extract experiment details
        exp_desc = experiment.get('description', 'Not specified') if experiment else 'Not specified'
        exp_steps = experiment.get('steps', []) if experiment else []
        datasets = experiment.get('datasets', {}) if experiment else {}
        metrics = experiment.get('metrics', []) if experiment else []
        
        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(exp_steps)])
        
        return f"""Write a detailed Methodology section (6-7 paragraphs, ~1000-1200 words):

**Hypothesis:** {hypothesis.get('title', '')}
**Proposed Solution:** {hypothesis.get('proposed_solution', '')}

**Experiment Details:**
- Description: {exp_desc}
- Steps: {steps_text}
- Datasets: {datasets}
- Metrics: {metrics}

Structure:
### 3. Methodology

**3.1 Overview** (1 paragraph)
- High-level description of research approach
- Justify methodology choice
- Overview of experimental pipeline

**3.2 Dataset** (2 paragraphs)
- Dataset source and characteristics
- Size, format, features
- Data preprocessing and cleaning
- Train/validation/test splits
- Any data augmentation or balancing

**3.3 Proposed Approach** (2 paragraphs)
- Detailed description of your method/algorithm/architecture
- Step-by-step explanation of the approach
- Key innovations or modifications
- Mathematical formulations if relevant (use simple notation)
- Diagram description (mention "Figure 1 shows...")

**3.4 Implementation Details** (1 paragraph)
- Tools, frameworks, libraries used
- Hyperparameters and configuration
- Hardware specifications
- Training procedure

**3.5 Evaluation Metrics** (1 paragraph)
- Explain each metric and why it's appropriate
- How success is defined
- Baseline comparisons

Write technically but clearly. Include enough detail for reproducibility."""

    def _prompt_results(self, context: Dict) -> str:
        """Generate prompt for expected results"""
        hypothesis = context.get('hypothesis', {})
        experiment = context.get('experiment', {})
        
        expected_outcomes = experiment.get('expected_outcomes', {}) if experiment else {}
        
        return f"""Write an Expected Results section (4-5 paragraphs, ~600-800 words):

**Hypothesis:** {hypothesis.get('title', '')}
**Novelty Score:** {hypothesis.get('novelty_score', 'N/A')}/10
**Impact Score:** {hypothesis.get('impact_score', 'N/A')}/10
**Expected Outcomes:** {expected_outcomes}

Structure:
### 4. Expected Results

**4.1 Anticipated Findings** (2 paragraphs)
- What you expect to observe
- Predicted performance metrics (be specific with ranges)
- Expected trends and patterns
- Success criteria

**4.2 Comparative Analysis** (1 paragraph)
- Expected comparison with baselines
- Performance improvements anticipated
- Where your approach should excel
- Potential trade-offs

**4.3 Validation of Hypothesis** (1 paragraph)
- How results would support your hypothesis
- What would constitute success
- Potential implications if hypothesis is validated

**4.4 Result Presentation Plan** (1 paragraph)
- Mention tables and figures you'll create
- "Table 1 will show...", "Figure 2 will compare..."
- How results will be visualized

Note: Write in future/conditional tense since experiment isn't complete yet:
- "We expect to observe..."
- "Results should demonstrate..."
- "The proposed approach is likely to..."

Be realistic but optimistic. Acknowledge uncertainty where appropriate."""

    def _prompt_discussion(self, context: Dict) -> str:
        """Generate prompt for discussion"""
        hypothesis = context.get('hypothesis', {})
        experiment = context.get('experiment', {})
        
        challenges = experiment.get('challenges', []) if experiment else []
        
        return f"""Write a Discussion section (5-6 paragraphs, ~800-1000 words):

**Hypothesis:** {hypothesis.get('title', '')}
**Expected Impact:** {hypothesis.get('expected_impact', '')}
**Potential Challenges:** {challenges}

Structure:
### 5. Discussion

**5.1 Interpretation of Results** (2 paragraphs)
- What the expected results mean
- Why the approach should work
- Theoretical justification
- Connection to hypothesis

**5.2 Implications** (1 paragraph)
- Practical applications
- Impact on the field
- Who benefits from this research
- Broader significance

**5.3 Comparison with Prior Work** (1 paragraph)
- How this advances beyond existing approaches
- Novel contributions
- Why this is an improvement

**5.4 Limitations** (1 paragraph - be honest!)
- Dataset limitations
- Methodological constraints
- Scope boundaries
- What this work doesn't address

**5.5 Threats to Validity** (1 paragraph)
- Potential biases
- Generalization concerns
- Assumptions made
- How you mitigate these

**5.6 Future Work** (1 paragraph)
- Natural extensions of this work
- Open questions
- Potential improvements
- Next research directions

Be critical and analytical. Discuss both strengths and limitations honestly. Use conditional language for expected results."""

    def _prompt_conclusion(self, context: Dict) -> str:
        """Generate prompt for conclusion"""
        hypothesis = context.get('hypothesis', {})
        
        return f"""Write a Conclusion section (3-4 paragraphs, ~400-500 words):

**Topic:** {context.get('topic', '')}
**Hypothesis:** {hypothesis.get('title', '')}
**Problem Addressed:** {hypothesis.get('problem_statement', '')}
**Proposed Solution:** {hypothesis.get('proposed_solution', '')}
**Expected Impact:** {hypothesis.get('expected_impact', '')}

Structure:
### 6. Conclusion

**Paragraph 1: Summary**
- Restate the problem concisely
- Summarize your proposed approach (2-3 sentences)
- Don't introduce new information

**Paragraph 2: Key Contributions**
- List 3-4 main contributions of this work
- What makes this research novel
- Expected advances to the field

**Paragraph 3: Expected Impact**
- Practical implications
- How this changes current practice
- Who benefits

**Paragraph 4: Closing Statement**
- Future vision
- Broader impact potential
- Strong closing sentence

Keep it concise and impactful. End on a positive, forward-looking note. No new citations needed."""

    def generate_references(self, papers: List) -> str:
        """
        Generate formatted references section
        
        Args:
            papers: List of paper objects
            
        Returns:
            Formatted references in APA style
        """
        refs = "## References\n\n"
        
        for i, paper in enumerate(papers, 1):
            title = getattr(paper, 'title', 'Unknown Title')
            authors = getattr(paper, 'authors', [])
            published = getattr(paper, 'published', '')
            link = getattr(paper, 'link', '') or getattr(paper, 'arxiv_url', '')
            
            # Extract year
            year = self._extract_year(published)
            
            # Format authors (APA style)
            if not authors:
                author_str = "Unknown Author"
            elif len(authors) == 1:
                author_str = authors[0]
            elif len(authors) <= 3:
                author_str = ", ".join(authors[:-1]) + ", & " + authors[-1]
            else:
                author_str = authors[0] + " et al."
            
            # Format citation
            citation = f"[{i}] {author_str} ({year}). *{title}*."
            
            if link:
                citation += f" Retrieved from {link}"
            
            refs += citation + "\n\n"
        
        return refs
    
    def generate_complete_paper(self, context: Dict) -> str:
        """
        Generate a complete research paper in ONE API call
        
        Args:
            context: Dictionary with all analysis data
            
        Returns:
            Complete paper as Markdown string
        """
        if not self.client:
            return "Error: OpenAI API key not configured"
        
        # Extract context
        hypothesis = context.get('hypothesis', {})
        experiment = context.get('experiment', {})
        papers = context.get('papers', [])
        lit_review = context.get('literature_review', '')
        
        # Format papers for citation
        papers_list = "\n".join([f"{i+1}. {self._format_paper_cite(p)}" for i, p in enumerate(papers[:15])])
        
        # Format experiment details
        exp_steps = experiment.get('steps', []) if experiment else []
        steps_text = "\n".join([f"- {step}" for step in exp_steps[:8]])
        
        datasets = experiment.get('datasets', {}) if experiment else {}
        metrics = experiment.get('metrics', []) if experiment else []
        expected_outcomes = experiment.get('expected_outcomes', {}) if experiment else {}
        challenges = experiment.get('challenges', []) if experiment else []
        
        # Create comprehensive single prompt
        comprehensive_prompt = f"""Generate a CONCISE academic research paper draft.

CONTEXT: {context.get('topic', '')} | Hypothesis: {hypothesis.get('title', '')}
Problem: {hypothesis.get('problem_statement', '')[:200]}
Solution: {hypothesis.get('proposed_solution', '')[:200]}

Papers: {papers_list[:500]}

SECTIONS:

## TITLE (10-12 words)

## Abstract (200 words)
Background, Problem, Approach, Contributions

## 1. Introduction (4 paragraphs, ~550 words)
- Paragraph 1: Background & importance
- Paragraph 2: Current state with citations
- Paragraph 3: Research gap
- Paragraph 4: Our contribution & paper structure

## 2. Literature Review (4-5 paragraphs, ~700 words)
- Organize by themes (not paper-by-paper)
- Compare and contrast approaches
- Synthesize findings
- Identify gaps
- Include citations: (Author et al., 2023)

## 3. Methodology (5-6 paragraphs, ~800 words)
- 3.1 Overview
- 3.2 Dataset details (source, size, preprocessing)
- 3.3 Proposed approach: {steps_text[:400]}
- 3.4 Implementation (tools, frameworks)
- 3.5 Evaluation metrics: {metrics}

## 4. Expected Results (4 paragraphs, ~550 words)
- Anticipated findings with metrics
- Comparison with baselines
- Hypothesis validation
- Result presentation plan
Use future tense: "We expect..."

## 5. Discussion (4-5 paragraphs, ~600 words)
- Interpretation of results
- Practical implications
- Comparison with prior work
- Limitations
- Future work

## 6. Conclusion (3 paragraphs, ~300 words)
- Summary of approach
- Key contributions
- Expected impact

GUIDELINES:
- Professional academic language
- Citations: (Author et al., 2023)
- Markdown formatting
- Target: ~3500 words total
- Be detailed but avoid redundancy"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are an expert academic writer. Generate well-structured, detailed research paper drafts with proper academic prose and citations."},
                    {"role": "user", "content": comprehensive_prompt}
                ],
                temperature=0.7,
                max_tokens=3500
            )
            
            paper_content = response.choices[0].message.content.strip()
            
            # Add references section
            papers = context.get('papers', [])
            references = self.generate_references(papers) if papers else "\n## References\n\nNo references available."
            
            # Add footer
            timestamp = datetime.now().strftime("%Y-%m-%d")
            footer = f"\n\n---\n\n*Generated: {timestamp} | ResearchForge AI Assistant | Status: Draft Template*"
            
            complete_paper = paper_content + "\n\n" + references + footer
            
            return complete_paper
            
        except Exception as e:
            return f"Error generating paper: {str(e)}\n\nPlease check your API key and try again."
    
    def export_to_markdown(self, sections: Dict, references: str) -> str:
        """
        Combine all sections into a complete Markdown document
        
        Args:
            sections: Dictionary of section_name: content
            references: Formatted references
            
        Returns:
            Complete Markdown document
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        doc = f"""# {sections.get('title', 'Research Paper')}

**Generated:** {timestamp}  
**Status:** Draft Template - Add Your Experimental Results

---

## Abstract

{sections.get('abstract', '')}

---

## 1. Introduction

{sections.get('introduction', '')}

---

{sections.get('literature_review', '')}

---

{sections.get('methodology', '')}

---

{sections.get('expected_results', '')}

---

{sections.get('discussion', '')}

---

{sections.get('conclusion', '')}

---

{references}

---

*This paper was generated using ResearchForge AI Assistant with GPT-3.5-turbo. Please add your experimental results and edit as needed before submission.*
"""
        
        return doc
    
    def _format_paper_cite(self, paper) -> str:
        """Format a paper for citation"""
        title = getattr(paper, 'title', 'Unknown')
        authors = getattr(paper, 'authors', [])
        published = getattr(paper, 'published', '')
        
        author_str = authors[0] if authors else "Unknown"
        year = self._extract_year(published)
        
        return f"{title} ({author_str} et al., {year})"
    
    def _extract_year(self, date_string) -> str:
        """Extract year from date string"""
        if not date_string:
            return "n.d."
        
        match = re.search(r'20\d{2}', str(date_string))
        return match.group() if match else "n.d."
