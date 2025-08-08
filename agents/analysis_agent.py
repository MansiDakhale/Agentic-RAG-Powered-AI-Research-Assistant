# agents/analysis_agent.py
from agents.base_agent import BaseAgent, AgentMessage
import json
from typing import List, Dict

class AnalysisAgent(BaseAgent):
    """Processes retrieved documents and synthesizes information"""
    
    def __init__(self, name: str, llm):
        super().__init__(name, llm)
        self.analysis_prompt = """
        You are an AI Research Analysis Agent. Your job is to analyze retrieved documents and synthesize key information.
        
        Original Query: {original_query}
        Query Type: {query_type}
        Expected Answer Type: {expected_answer_type}
        
        Retrieved Documents:
        {documents}
        
        Please analyze these documents and provide:
        
        1. KEY FINDINGS: Main insights relevant to the query
        2. THEMES: Common patterns or themes across documents
        3. EVIDENCE: Specific facts, statistics, or quotes that support findings
        4. GAPS: What information might be missing or unclear
        5. SYNTHESIS: How the information comes together to answer the query
        
        Structure your response as JSON:
        {{
            "key_findings": ["finding1", "finding2", "finding3"],
            "themes": ["theme1", "theme2"],
            "evidence": [
                {{"finding": "...", "evidence": "...", "source": "..."}},
                {{"finding": "...", "evidence": "...", "source": "..."}}
            ],
            "information_gaps": ["gap1", "gap2"],
            "synthesis": "comprehensive analysis combining all findings",
            "confidence_score": 0.85
        }}
        """
    
    def execute(self, message: AgentMessage) -> AgentMessage:
        """Analyze retrieved documents and synthesize information"""
        self.log_activity("Starting document analysis")
        
        search_results = message.content
        query_analysis = search_results["query_analysis"]
        documents = search_results["documents"]
        
        # Prepare documents for analysis
        doc_text = self._prepare_documents(documents)
        
        # Generate analysis
        analysis = self._analyze_documents(query_analysis, doc_text)
        
        # Prepare final analysis results
        analysis_results = {
            "query_analysis": query_analysis,
            "search_metadata": {
                "total_documents": len(documents),
                "search_queries": search_results["search_queries"]
            },
            "analysis": analysis,
            "processed_documents": documents[:5]  # Keep top 5 for reference
        }
        
        self.log_activity(f"Analysis complete with confidence: {analysis.get('confidence_score', 'N/A')}")
        
        return self.send_message("report_agent", analysis_results, "analysis_complete")
    
    def _prepare_documents(self, documents: List[Dict]) -> str:
        """Format documents for analysis"""
        formatted_docs = []
        for i, doc in enumerate(documents[:8]):  # Limit to top 8 documents
            formatted_docs.append(f"Document {i+1} (Rank {doc['rank']}):\n{doc['content']}\n")
        
        return "\n".join(formatted_docs)
    
    def _analyze_documents(self, query_analysis: Dict, documents: str) -> Dict:
        """Perform the actual analysis using LLM"""
        prompt = self.analysis_prompt.format(
            original_query=query_analysis["original_query"],
            query_type=query_analysis.get("query_type", "exploratory"),
            expected_answer_type=query_analysis.get("expected_answer_type", "summary"),
            documents=documents
        )
        
        try:
            analysis_response = self.llm(prompt)
            analysis = json.loads(analysis_response)
            
            # Add metadata
            analysis["document_count"] = len(documents.split("Document"))
            analysis["analysis_timestamp"] = self._get_timestamp()
            
            return analysis
            
        except json.JSONDecodeError:
            # Fallback analysis
            return self._create_fallback_analysis(query_analysis, documents)
    
    def _create_fallback_analysis(self, query_analysis: Dict, documents: str) -> Dict:
        """Create basic analysis if JSON parsing fails"""
        return {
            "key_findings": ["Analysis of retrieved documents completed"],
            "themes": ["General information found"],
            "evidence": [],
            "information_gaps": ["Detailed analysis may be limited"],
            "synthesis": f"Based on the retrieved documents, information related to '{query_analysis['original_query']}' has been found and processed.",
            "confidence_score": 0.6,
            "document_count": len(documents.split("Document")),
            "analysis_timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()

# Test the analysis agent
if __name__ == "__main__":
    from langchain.llms import Ollama
    
    llm = Ollama(model="llama3.2:3b")
    agent = AnalysisAgent("test_analysis_agent", llm)
    
    # Mock search results
    test_search_results = {
        "query_analysis": {
            "original_query": "AI safety trends",
            "query_type": "exploratory",
            "expected_answer_type": "summary"
        },
        "search_queries": ["AI safety", "alignment research"],
        "documents": [
            {"content": "AI alignment research focuses on ensuring AI systems behave as intended.", "rank": 1},
            {"content": "Robustness in AI systems is crucial for preventing unexpected failures.", "rank": 2}
        ]
    }
    
    test_message = AgentMessage("search_agent", "analysis_agent", test_search_results, "search_results")
    result = agent.execute(test_message)
    
    print("Analysis Result:")
    print(json.dumps(result.content["analysis"], indent=2))