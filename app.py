import streamlit as st
import asyncio
import pandas as pd
from typing import List
import os
from dotenv import load_dotenv
from multi_agent_system import MultiAgentSystem, SearchResult

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔍",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables"""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

async def run_async_operations(system, search_query, max_results=3):
    """Handle async complete analysis operation"""
    # Always run full pipeline: Paper Collection → Literature Review → Gap Analysis → Hypothesis Generation
    complete_results = await system.run_complete_analysis(search_query)
    return {"type": "complete", "data": complete_results}

def main():
    """Main application function"""
    st.title("📚 AI Research Assistant - Complete Analysis")
    st.markdown("*Automated Literature Review • Gap Analysis • Hypothesis Generation*")
    
    init_session_state()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### Analysis Pipeline")
        st.info("""
        **What happens when you search:**
        1. 📄 Collect 10-15 research papers
        2. 📚 Literature Review (GPT-3.5)
        3. 🔍 Gap Analysis (Flan-T5 + GPT-4)
        4. 💡 Hypothesis Generation (GPT-4)
        """)

        # API Status
        st.header("🔌 API Status")
        if os.getenv('OPENAI_API_KEY') and os.getenv('GOOGLE_SEARCH_ENGINE_ID'):
            st.success("✅ All APIs configured")
        else:
            st.error("❌ APIs not configured. Please check .env file")
        
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("• OpenAI GPT-4 & GPT-3.5")
        st.markdown("• Google Flan-T5-Large")
        st.markdown("• Sentence-BERT")

    # Main search interface
    st.markdown("## 🔍 Enter Your Research Topic")
    
    search_query = st.text_input(
        "Research Topic",
        placeholder="e.g., 'machine learning for drug discovery' or 'quantum computing applications'",
        label_visibility="collapsed"
    )
    
    search_button = st.button("🚀 Start Complete Analysis", type="primary", use_container_width=True)

    # Handle search
    if search_button and search_query:
        with st.spinner("🔄 Running Complete Analysis... This may take 2-3 minutes"):
            try:
                system = MultiAgentSystem()
                
                # Always run complete analysis
                result = asyncio.run(run_async_operations(
                    system, 
                    search_query,
                    max_results=5  # Not used for complete analysis
                ))
                
                # Display complete analysis results
                complete_data = result["data"]
                
                st.success("✅ Complete Analysis Finished!")
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📚 Literature Review", 
                    "🔍 Gap Analysis", 
                    "💡 Hypotheses",
                    "🧪 Experiment Design"
                ])
                
                with tab1:
                    st.header("Literature Review")
                    if complete_data.get("literature_review"):
                        st.markdown(complete_data["literature_review"])
                    else:
                        st.warning("No literature review available")
                
                with tab2:
                    st.header("Research Gap Analysis")
                    gap_data = complete_data.get("gap_analysis", {})
                    
                    # Show which model was used
                    gaps_dict = gap_data.get("gaps", {})
                    if gaps_dict.get("source"):
                        if "Combined" in gaps_dict["source"]:
                            st.success(f"🤖 Analysis by: {gaps_dict['source']}")
                        elif "GPT-4" in gaps_dict["source"]:
                            st.info(f"🤖 Analysis by: {gaps_dict['source']}")
                        else:
                            st.success(f"🤖 Analysis by: {gaps_dict['source']}")
                    
                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Papers Analyzed", gap_data.get("papers_analyzed", 0))
                    with col2:
                        total_gaps = sum(len(gaps_dict.get(k, [])) for k in ["knowledge_gaps", "methodological_gaps", "dataset_gaps", "temporal_gaps"])
                        st.metric("Total Gaps Found", total_gaps)
                    with col3:
                        st.metric("Categories", 4)
                    
                    # Display formatted gap analysis
                    st.markdown(gap_data.get("formatted_output", "No gap analysis available"))
                
                with tab3:
                    st.header("Research Hypotheses")
                    hyp_data = complete_data.get("hypotheses", {})
                    
                    if hyp_data.get("hypotheses"):
                        # Show summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Hypotheses Generated", len(hyp_data.get("hypotheses", [])))
                        with col2:
                            hypotheses_list = hyp_data.get("hypotheses", [])
                            avg_score = sum(h.get("overall_score", 0) for h in hypotheses_list) / max(len(hypotheses_list), 1)
                            st.metric("Avg Overall Score", f"{avg_score:.2f}")
                        with col3:
                            avg_novelty = sum(h.get("novelty_score", 0) for h in hypotheses_list) / max(len(hypotheses_list), 1)
                            st.metric("Avg Novelty Score", f"{avg_novelty:.2f}")
                        
                        # Display hypotheses with selection
                        st.markdown("### Select a Hypothesis for Experiment Design")
                        
                        for idx, hyp in enumerate(hypotheses_list, 1):
                            with st.expander(f"{'🔴' if hyp.get('priority') == 'HIGH' else '🟡'} Hypothesis {idx}: {hyp.get('title', 'Untitled')}", expanded=(idx==1)):
                                # Display hypothesis details
                                novelty = hyp.get('novelty_score', 'N/A')
                                impact = hyp.get('impact_score', 'N/A')
                                feasibility = hyp.get('feasibility_score', 'N/A')
                                
                                st.markdown(f"**📊 Scores:** Novelty: {novelty}/10 | Impact: {impact}/10 | Feasibility: {feasibility}/10")
                                
                                if hyp.get('gap_addressed'):
                                    st.markdown(f"**🎯 Gap Addressed:** {hyp['gap_addressed']}")
                                
                                if hyp.get('problem_statement'):
                                    st.markdown(f"**❓ Problem Statement:** {hyp['problem_statement']}")
                                
                                if hyp.get('proposed_solution'):
                                    st.markdown(f"**💡 Proposed Solution:** {hyp['proposed_solution']}")
                                
                                if hyp.get('expected_impact'):
                                    st.markdown(f"**🚀 Expected Impact:** {hyp['expected_impact']}")
                                
                                # Selection button
                                if st.button(f"✅ Select This Hypothesis for Experiments", key=f"select_hyp_{idx}"):
                                    st.session_state.selected_hypothesis = hyp
                                    st.session_state.selected_hypothesis_idx = idx
                                    st.success(f"✅ Selected Hypothesis {idx} for experiment design!")
                                    st.info("👉 Go to the 'Experiment Design' tab to see suggested experiments")
                    else:
                        st.warning("No hypotheses generated")
                
                with tab4:
                    st.header("🧪 Experiment Design & Dataset Recommendations")
                    
                    # Check if a hypothesis has been selected
                    if hasattr(st.session_state, 'selected_hypothesis') and st.session_state.selected_hypothesis:
                        selected_hyp = st.session_state.selected_hypothesis
                        hyp_idx = st.session_state.get('selected_hypothesis_idx', 1)
                        
                        st.success(f"✅ Designing experiments for: **{selected_hyp.get('title')}**")
                        
                        # Display selected hypothesis summary
                        with st.expander("📋 Selected Hypothesis Summary", expanded=False):
                            st.markdown(f"**Problem:** {selected_hyp.get('problem_statement', 'N/A')}")
                            st.markdown(f"**Solution:** {selected_hyp.get('proposed_solution', 'N/A')}")
                        
                        st.markdown("---")
                        
                        # Placeholder for future experiment generation
                        st.info("""
                        ### 🚧 Experiment Design (Coming Soon)
                        
                        This feature will automatically generate:
                        
                        **1. Suggested Experiments (1-3)**
                        - Detailed methodology
                        - Step-by-step approach
                        - Control vs treatment setup
                        
                        **2. Dataset Recommendations**
                        - Relevant datasets with links
                        - Dataset characteristics (size, format, domain)
                        - Preprocessing requirements
                        
                        **3. Evaluation Metrics**
                        - Primary metrics (accuracy, F1, etc.)
                        - Secondary metrics
                        - Baseline comparisons
                        
                        **4. Implementation Roadmap**
                        - Week-by-week plan
                        - Milestones and checkpoints
                        - Expected challenges
                        
                        **5. Resource Requirements**
                        - Compute infrastructure
                        - Storage needs
                        - Estimated timeline
                        """)
                        
                        # Button to generate experiments (placeholder for future)
                        if st.button("🚀 Generate Experiment Plan (Coming Soon)", disabled=True):
                            st.warning("This feature is under development")
                    else:
                        st.info("""
                        ### 👈 No Hypothesis Selected
                        
                        Please go to the **'Hypotheses'** tab and select a hypothesis 
                        by clicking the **"✅ Select This Hypothesis for Experiments"** button.
                        
                        Once selected, this tab will show:
                        - Suggested experiments
                        - Dataset recommendations  
                        - Evaluation metrics
                        - Implementation roadmap
                        """)
                
                # Add to search history
                st.session_state.search_history.append({
                    "query": search_query,
                    "type": "Complete Analysis",
                    "timestamp": pd.Timestamp.now()
                })
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(f"Error details: {type(e).__name__}")

    # Show search history
    if st.session_state.search_history:
        st.header("Search History")
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(
            history_df,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time",
                    format="D MMM, YYYY, HH:mm"
                )
            }
        )

if __name__ == "__main__":
    main()