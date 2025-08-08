"""
Agentic RAG Assistant - Day 1 Foundation
Basic RAG implementation without agents
"""

import os
import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import tempfile
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def load_document(self, file_path: str, file_type: str):
        """Load document based on file type"""
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path)
            elif file_type in ["docx", "doc"]:
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def process_documents(self, documents):
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

class RAGSystem:
    """Main RAG system implementation"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.doc_processor = DocumentProcessor()
        
    def initialize(self):
        """Initialize all components"""
        try:
            # Initialize embeddings
            st.info("Loading embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize LLM (Ollama)
            st.info("Connecting to Ollama...")
            self.llm = Ollama(
                model="llama3.2:3b",
                temperature=0.1
            )
            
            # Test LLM connection
            test_response = self.llm("Hello")
            st.success("✅ Ollama connection successful!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
            return False
    
    def create_vectorstore(self, documents):
        """Create vector store from documents"""
        try:
            st.info("Creating vector embeddings...")
            
            # Process documents into chunks
            chunks = self.doc_processor.process_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )
            
            st.success(f"✅ Vector store created with {len(chunks)} chunks!")
            return True
            
        except Exception as e:
            st.error(f"❌ Vector store creation failed: {str(e)}")
            logger.error(f"Vector store error: {str(e)}")
            return False
    
    def query(self, question: str):
        """Query the RAG system"""
        if not self.qa_chain:
            return "❌ System not initialized. Please upload documents first."
        
        try:
            st.info("🔍 Searching documents...")
            response = self.qa_chain({"query": question})
            
            answer = response["result"]
            sources = response["source_documents"]
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return f"❌ Error processing query: {str(e)}"

def main():
    """Streamlit main function"""
    st.set_page_config(
        page_title="Agentic RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Agentic RAG Assistant")
    st.subheader("Basic RAG Implementation")
    st.write("""This is a basic implementation of a Retrieval-Augmented Generation (RAG) system.
You can upload documents, create a knowledge base, and ask questions about the content.""")
    
    # Initialize session state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        st.session_state.initialized = False
        st.session_state.documents_loaded = False
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            if not st.session_state.initialized:
                if st.session_state.rag_system.initialize():
                    st.session_state.initialized = True
                else:
                    st.stop()
            
            # Process uploaded files
            all_documents = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Load document
                    file_type = uploaded_file.name.split(".")[-1].lower()
                    documents = st.session_state.rag_system.doc_processor.load_document(
                        tmp_file_path, file_type
                    )
                    all_documents.extend(documents)
                    
                finally:
                    # Clean up temp file
                    os.unlink(tmp_file_path)
            
            # Create vector store
            if st.session_state.rag_system.create_vectorstore(all_documents):
                st.session_state.documents_loaded = True
                st.rerun()
    
    # Main interface
    if st.session_state.documents_loaded:
        st.success("📚 Documents loaded successfully! You can now ask questions.")
        
        # Query interface
        question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings in the documents?"
        )
        
        if question and st.button("🔍 Search"):
            with st.spinner("Processing your question..."):
                result = st.session_state.rag_system.query(question)
                
                if isinstance(result, dict):
                    st.subheader("📝 Answer")
                    st.write(result["answer"])
                    
                    if result["sources"]:
                        st.subheader("📚 Sources")
                        for i, source in enumerate(result["sources"][:3], 1):
                            with st.expander(f"Source {i}"):
                                st.write(source.page_content[:500] + "...")
                                if hasattr(source, 'metadata'):
                                    st.write("**Metadata:**", source.metadata)
                else:
                    st.error(result)
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What are the main topics discussed in the documents?",
            "Can you summarize the key findings?",
            "What are the most important conclusions?",
            "What methodologies are mentioned?"
        ]
        
        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.text_input("Ask a question about your documents:", value=q, key="auto_question")
    
    else:
        st.info("""
        👋 Welcome to the Agentic RAG Assistant!
        
        **Getting Started:**
        1. Make sure Ollama is running with llama3.2:3b model
        2. Upload your documents using the sidebar
        3. Click 'Process Documents' to create the knowledge base
        4. Start asking questions!
        
        **Supported file types:** PDF, TXT, DOCX
        """)
        
        # Quick test button
        if st.button("🧪 Test Ollama Connection"):
            test_system = RAGSystem()
            if test_system.initialize():
                st.success("✅ System ready for document processing!")

if __name__ == "__main__":
    main()