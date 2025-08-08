# app.py - Main Streamlit Application
import streamlit as st
import time
from datetime import datetime
from langchain.llms import Ollama
from utils.vector_store import VectorStore
from agents.orchestrator import AgentOrchestrator
from utils.document_loader import DocumentLoader
import json

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-status {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

@st.cache_resource
def initialize_system():
    """Initialize the RAG system"""
    try:
        llm = Ollama(model="llama3.2:3b")
        vector_store = VectorStore()
        orchestrator = AgentOrchestrator(llm, vector_store)
        return orchestrator, "success"
    except Exception as e:
        return None, f"Error: {str(e)}"

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Agentic RAG Research Assistant</h1>
        <p>Multi-Agent AI System for Intelligent Research Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        # Initialize system
        if st.button("🔄 Initialize System"):
            with st.spinner("Initializing AI agents..."):
                orchestrator, status = initialize_system()
                if status == "success":
                    st.session_state.orchestrator = orchestrator
                    st.success("✅ System initialized!")
                else:
                    st.error(f"❌ {status}")
        
        # System status
        if st.session_state.orchestrator:
            st.success("🟢 System Ready")
            st.info("🤖 4 AI Agents Active")
        else:
            st.warning("🟡 System Not Initialized")
        
        st.markdown("---")
        
        # Document management
        st.header("📚 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Research Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDFs, text files, or Word documents for analysis"
        )
        
        if uploaded_files and st.button("📥 Process Documents"):
            if st.session_state.orchestrator:
                with st.spinner("Processing documents..."):
                    try:
                        loader = DocumentLoader()
                        documents = loader.process_files(uploaded_files)
                        st.session_state.orchestrator.add_documents(documents)
                        st.session_state.documents_loaded = True
                        st.success(f"✅ Processed {len(documents)} document chunks")
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
            else:
                st.warning("Please initialize system first")
        
        # Document status
        if st.session_state.documents_loaded:
            st.success("📚 Documents Loaded")
        else:
            st.info("📚 No documents loaded")
        
        st.markdown("---")
        
        # Query history
        st.header("📝 Query History")
        if st.session_state.query_history:
            for i, (query, timestamp) in enumerate(st.session_state.query_history[-5:], 1):
                st.text(f"{i}. {query[:30]}...")
        else:
            st.info("No queries yet")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Research Query")
        
        # Sample queries
        with st.expander("💡 Try these sample queries"):
            sample_queries = [
                "What are the latest trends in AI safety research?",
                "How do large language models handle bias and fairness?", 
                "Compare different approaches to machine learning robustness",
                "What are the ethical implications of AI in healthcare?",
                "Summarize recent developments in transformer architectures"
            ]
            
            for query in sample_queries:
                if st.button(f"📋 {query}", key=f"sample_{query[:20]}"):
                    st.session_state.current_query = query
        
        # Query input
        query = st.text_area(
            "Enter your research question:",
            value=st.session_state.get('current_query', ''),
            height=100,
            help="Ask any research question about your documents"
        )
        
        # Processing options
        col_a, col_b = st.columns(2)
        with col_a:
            show_agent_progress = st.checkbox("Show Agent Progress", value=True)
        with col_b:
            detailed_report = st.checkbox("Detailed Report", value=True)
        
        # Process query
        if st.button("🚀 Start Research", type="primary") and query:
            if not st.session_state.orchestrator:
                st.error("❌ Please initialize the system first")
            elif not st.session_state.documents_loaded:
                st.warning("⚠️ No documents loaded. Upload documents first for better results.")
            else:
                # Add to history
                st.session_state.query_history.append(
                    (query, datetime.now().strftime("%H:%M:%S"))
                )
                
                # Create progress containers
                if show_agent_progress:
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                    agent_logs = st.empty()
                
                # Progress callback function
                def progress_callback(progress, message):
                    if show_agent_progress:
                        progress_bar.progress(progress)
                        status_container.markdown(f"""
                        <div class="agent-status">
                            <strong>🤖 {message}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.5)  # Visual feedback
                
                # Execute query
                try:
                    start_time = time.time()
                    
                    with st.spinner("AI agents are processing your query..."):
                        result = st.session_state.orchestrator.process_query(
                            query, 
                            progress_callback=progress_callback if show_agent_progress else None
                        )
                    
                    execution_time = time.time() - start_time
                    
                    # Clear progress indicators
                    if show_agent_progress:
                        progress_bar.progress(1.0)
                        status_container.markdown("""
                        <div class="success-message">
                            ✅ <strong>Research Complete!</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display results
                    st.success(f"✅ Research completed in {execution_time:.1f} seconds")
                    
                    # Show the report
                    st.markdown("## 📊 Research Report")
                    st.markdown("---")
                    st.markdown(result)
                    
                    # Execution summary
                    if detailed_report:
                        summary = st.session_state.orchestrator.get_execution_summary()
                        
                        with st.expander("📈 Execution Details"):
                            col_x, col_y, col_z = st.columns(3)
                            
                            with col_x:
                                st.metric("Success Rate", f"{summary['success_rate']*100:.1f}%")
                            with col_y:
                                st.metric("Agents Used", f"{summary['successful']}/{summary['total_agents']}")
                            with col_z:
                                st.metric("Processing Time", f"{execution_time:.1f}s")
                            
                            st.json(summary['execution_log'])
                    
                    # Download options
                    st.download_button(
                        label="📄 Download Report",
                        data=result,
                        file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    
                    if show_agent_progress:
                        status_container.markdown(f"""
                        <div class="error-message">
                            ❌ <strong>Processing failed:</strong> {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.header("ℹ️ System Information")
        
        # Agent status
        if st.session_state.orchestrator:
            st.markdown("### 🤖 Active Agents")
            agents = [
                ("Query Agent", "🔍", "Analyzes research questions"),
                ("Search Agent", "📚", "Finds relevant documents"), 
                ("Analysis Agent", "🧠", "Synthesizes information"),
                ("Report Agent", "📝", "Generates final reports")
            ]
            
            for name, emoji, description in agents:
                with st.container():
                    st.markdown(f"""
                    **{emoji} {name}**  
                    *{description}*
                    """)
        
        st.markdown("---")
        
        # Performance tips
        st.markdown("### 💡 Performance Tips")
        st.markdown("""
        - Upload relevant documents first
        - Ask specific, focused questions
        - Use domain-specific terminology
        - Check agent progress for debugging
        """)
        
        st.markdown("---")
        
        # Technical details
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **Technology Stack:**
            - LLM: Ollama + Llama3.2:3b
            - Vector DB: ChromaDB
            - Framework: LangChain
            - UI: Streamlit
            
            **Architecture:**
            - Multi-agent system
            - RAG (Retrieval Augmented Generation)
            - Vector similarity search
            - Automated report generation
            """)

if __name__ == "__main__":
    main()