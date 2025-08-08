# agents/orchestrator.py - Improved version with better error handling
from typing import Dict, List, Optional, Callable
from agents.base_agent import BaseAgent, AgentMessage
from agents.query_agent import QueryUnderstandingAgent
from agents.search_agent import SearchAgent
from agents.analysis_agent import AnalysisAgent  
from agents.report_agent import ReportGenerationAgent
import time
import traceback

class ImprovedAgentOrchestrator:
    """Enhanced orchestrator with better error handling and fallback mechanisms"""
    
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        
        # Test LLM availability first
        self.llm_available = self._test_llm()
        
        # Initialize all agents
        self.agents: Dict[str, BaseAgent] = {
            "query_agent": QueryUnderstandingAgent("query_agent", llm),
            "search_agent": SearchAgent("search_agent", llm, vector_store),
            "analysis_agent": AnalysisAgent("analysis_agent", llm),
            "report_agent": ReportGenerationAgent("report_agent", llm)
        }
        
        # Workflow configuration with fallback options
        self.workflow = [
            ("query_agent", "Analyzing your question...", True),  # Can fallback
            ("search_agent", "Searching relevant documents...", False),  # Essential
            ("analysis_agent", "Analyzing findings...", True),  # Can fallback
            ("report_agent", "Generating final report...", True)  # Can fallback
        ]
        
        self.execution_log = []
        self.fallback_mode = not self.llm_available
    
    def _test_llm(self) -> bool:
        """Test if LLM is working properly"""
        try:
            response = self.llm("Hello")
            return len(response.strip()) > 0
        except Exception as e:
            print(f"⚠️ LLM test failed: {str(e)}")
            return False
    
    def process_query(self, user_query: str, progress_callback: Optional[Callable] = None) -> str:
        """Execute complete agent workflow with enhanced error handling"""
        print(f"\n🤖 Starting Agentic RAG processing for: '{user_query}'")
        print("=" * 60)
        
        if self.fallback_mode:
            print("⚠️ Running in fallback mode (LLM unavailable)")
        
        # Document count info
        if hasattr(self.vector_store, 'collection') and self.vector_store.collection:
            doc_count = self.vector_store.collection.count()
            print(f"📚 Processing with {doc_count} documents in vector store")
        
        # Initialize workflow
        current_message = AgentMessage("user", "query_agent", user_query, "query")
        
        # Execute each agent in sequence
        for i, (agent_name, status_message, can_fallback) in enumerate(self.workflow):
            if progress_callback:
                progress_callback((i + 1) / len(self.workflow), status_message)
            
            print(f"\n🔄 Step {i+1}/4: {status_message}")
            
            agent = self.agents[agent_name]
            
            try:
                if self.fallback_mode and can_fallback:
                    # Use fallback for LLM-dependent agents
                    current_message = self._fallback_execution(agent_name, current_message)
                else:
                    # Normal execution
                    current_message = agent.execute(current_message)
                
                # Log successful execution
                self.execution_log.append({
                    "agent": agent_name,
                    "status": "success" if not self.fallback_mode or not can_fallback else "fallback",
                    "message_type": current_message.message_type
                })
                
                print(f"✅ {agent_name} completed successfully")
                
            except Exception as e:
                print(f"❌ Error in {agent_name}: {str(e)}")
                
                # Log error and attempt recovery
                self.execution_log.append({
                    "agent": agent_name,
                    "status": "error",
                    "error": str(e)
                })
                
                # Attempt recovery
                current_message = self._recover_from_error(agent_name, current_message, str(e))
        
        print(f"\n🎉 Processing complete!")
        print("=" * 60)
        
        return current_message.content
    
    def _fallback_execution(self, agent_name: str, message: AgentMessage) -> AgentMessage:
        """Provide fallback functionality when LLM is not available"""
        
        if agent_name == "query_agent":
            # Simple query analysis without LLM
            user_query = message.content
            simple_analysis = {
                "original_query": user_query,
                "main_topic": "research query",
                "sub_topics": user_query.split()[:3],  # First 3 words as topics
                "query_type": "exploratory",
                "entities": [word for word in user_query.split() if len(word) > 3],
                "search_strategy": "keyword-based search",
                "expected_answer_type": "summary",
                "refined_query": user_query
            }
            print("🔄 Using simple keyword-based query analysis")
            return AgentMessage(agent_name, "search_agent", simple_analysis, "query_analysis")
            
        elif agent_name == "analysis_agent":
            # Simple analysis without LLM
            search_results = message.content
            documents = search_results.get("documents", [])
            
            simple_analysis = {
                "key_findings": [f"Found {len(documents)} relevant documents"],
                "themes": ["Information retrieval", "Document analysis"],
                "evidence": [{"finding": "Documents retrieved", "evidence": f"{len(documents)} documents", "source": "vector_search"}],
                "information_gaps": ["Limited analysis without LLM"],
                "synthesis": f"Retrieved {len(documents)} documents related to the query. Manual review recommended for detailed analysis.",
                "confidence_score": 0.6,
                "document_count": len(documents)
            }
            
            analysis_results = {
                "query_analysis": search_results["query_analysis"],
                "search_metadata": search_results.get("search_metadata", {"total_documents": len(documents)}),
                "analysis": simple_analysis,
                "processed_documents": documents[:5]
            }
            
            print("🔄 Using simple document counting analysis")
            return AgentMessage(agent_name, "report_agent", analysis_results, "analysis_complete")
            
        elif agent_name == "report_agent":
            # Simple report without LLM
            analysis_results = message.content
            query = analysis_results["query_analysis"]["original_query"]
            doc_count = len(analysis_results.get("processed_documents", []))
            
            simple_report = f"""# Research Report: {query}

## Executive Summary
Document retrieval completed for the research query: "{query}"

## Search Results  
- **Documents Found**: {doc_count}
- **Search Method**: Vector similarity search
- **Processing Mode**: Fallback mode (limited LLM analysis)

## Key Documents Retrieved
"""
            
            for i, doc in enumerate(analysis_results.get("processed_documents", [])[:3], 1):
                content = doc.get("content", "")[:200]
                simple_report += f"\n{i}. {content}...\n"
            
            simple_report += f"""
## Limitations
- Full semantic analysis unavailable due to LLM connectivity issues
- Manual review of retrieved documents recommended
- Consider re-running when LLM service is available

## Next Steps
1. Review the retrieved documents manually
2. Check LLM connectivity and re-run for full analysis
3. Consider refining the search query if results are not relevant

---
*Generated in fallback mode - Limited functionality*
"""
            
            print("🔄 Using simple template-based report")
            return AgentMessage(agent_name, "user", simple_report, "final_report")
        
        # Default fallback
        return message
    
    def _recover_from_error(self, agent_name: str, message: AgentMessage, error: str) -> AgentMessage:
        """Attempt to recover from agent errors"""
        
        if agent_name == "report_agent":
            # Generate minimal error report
            error_report = f"""# Research Report - Error Recovery

## Query
{message.content.get('query_analysis', {}).get('original_query', 'Unknown query')}

## Status
❌ Report generation failed due to: {error}

## Available Information
Search and analysis were attempted but report generation encountered an error.

## Recommendation
Please try again or check system configuration.

---
*Error recovery report*
"""
            return AgentMessage(agent_name, "user", error_report, "error_report")
        
        elif "analysis" in agent_name:
            # Return search results as-is for report generation
            return message
        
        # For other agents, pass message through
        return message
    
    def add_documents(self, documents: List[str]):
        """Add documents to the vector store"""
        if self.vector_store:
            self.vector_store.add_documents(documents)
            print(f"📚 Added {len(documents)} documents to vector store")
    
    def get_execution_summary(self) -> Dict:
        """Get summary of last execution"""
        successful_agents = [log for log in self.execution_log if log["status"] in ["success", "fallback"]]
        failed_agents = [log for log in self.execution_log if log["status"] == "error"]
        
        return {
            "total_agents": len(self.workflow),
            "successful": len(successful_agents),
            "failed": len(failed_agents),
            "success_rate": len(successful_agents) / len(self.workflow) if self.workflow else 0,
            "execution_log": self.execution_log,
            "fallback_mode": self.fallback_mode,
            "llm_available": self.llm_available
        }
    
    def get_system_status(self) -> Dict:
        """Get detailed system status"""
        status = {
            "llm_available": self.llm_available,
            "fallback_mode": self.fallback_mode,
            "vector_store_available": self.vector_store is not None,
            "agents_initialized": len(self.agents),
            "document_count": 0
        }
        
        if hasattr(self.vector_store, 'collection') and self.vector_store.collection:
            try:
                status["document_count"] = self.vector_store.collection.count()
            except:
                status["document_count"] = "Unknown"
        
        return status

# For backward compatibility
AgentOrchestrator = ImprovedAgentOrchestrator