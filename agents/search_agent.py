# agents/search_agent.py
from pandas import unique
from agents.base_agent import BaseAgent, AgentMessage
from typing import List, Dict
import json

class SearchAgent(BaseAgent):
    """Retrieves relevant documents and information from vector store"""
    
    def __init__(self, name: str, llm, vector_store):
        super().__init__(name, llm, vector_store)
        self.search_prompt = """
        Based on the query analysis, generate optimized search queries for document retrieval.
        
        Query Analysis: {analysis}
        
        Generate 3-5 different search queries that would help find relevant information:
        1. Primary search query (main topic)
        2. Secondary search queries (sub-topics and entities)
        3. Alternative phrasings or related concepts
        
        Return as JSON array: ["query1", "query2", "query3", ...]
        """
    
    def execute(self, message: AgentMessage) -> AgentMessage:
        """Search for relevant documents based on query analysis"""
        self.log_activity("Starting document search")
        
        query_analysis = message.content
        
        # Generate multiple search queries
        search_queries = self._generate_search_queries(query_analysis)
        
        # Perform searches
        all_results = []
        for query in search_queries:
            self.log_activity(f"Searching for: {query}")
            results = self.vector_store.search(query, n_results=3)
            all_results.extend(results)
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query_analysis)
        
        # Prepare search results
        search_results = {
            "query_analysis": query_analysis,
            "search_queries": search_queries,
            "documents": ranked_results[:10],  # Top 10 results
            "total_found": len(unique_results)
        }
        
        self.log_activity(f"Found {len(unique_results)} unique documents")
        
        return self.send_message("analysis_agent", search_results, "search_results")
    
    def _generate_search_queries(self, analysis: Dict) -> List[str]:
        """Generate multiple search queries from analysis"""
        queries = []
        
        # Primary query
        queries.append(analysis.get("refined_query", analysis["original_query"]))
        
        # Topic-based queries
        main_topic = analysis.get("main_topic", "")
        if main_topic:
            queries.append(main_topic)
        
        # Entity-based queries
        for entity in analysis.get("entities", []):
            queries.append(entity)
        
        # Sub-topic queries
        for subtopic in analysis.get("sub_topics", []):
            queries.append(subtopic)
        
        # Use LLM to generate additional queries if needed
        if len(queries) < 3:
            prompt = self.search_prompt.format(analysis=json.dumps(analysis))
            try:
                llm_queries = json.loads(self.llm(prompt))
                queries.extend(llm_queries[:3])
            except:
                pass
        
        return list(set(queries))  # Remove duplicates
    
    def _deduplicate_results(self, results: List) -> List:
        """Remove duplicate documents based on content + source."""
        seen = set()
        unique = []
        for doc in results:
            doc_id = (getattr(doc, 'page_content', str(doc)) + str(getattr(doc, 'metadata', {})))
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
        return unique

    def _rank_results(self, results: List, analysis: Dict) -> List[Dict]:
        """Rank results by relevance, keeping metadata."""
        ranked_docs = []
        for i, doc in enumerate(results):
            content = getattr(doc, 'page_content', str(doc))
            metadata = getattr(doc, 'metadata', {})
            ranked_docs.append({
                "content": content,
                "rank": i + 1,
                "relevance_score": 1.0 / (i + 1),
                "metadata": metadata
            })
        return ranked_docs


# Test the agent
if __name__ == "__main__":
    from langchain.llms import Ollama
    from utils.vector_store import VectorStore
    
    llm = Ollama(model="llama3.1:8b")
    vector_store = VectorStore()
    
    agent = SearchAgent("test_search_agent", llm, vector_store)
    
    # Mock query analysis
    test_analysis = {
        "original_query": "AI safety trends",
        "main_topic": "artificial intelligence safety",
        "sub_topics": ["alignment", "robustness", "interpretability"],
        "entities": ["AI", "safety", "research"],
        "refined_query": "latest artificial intelligence safety research trends"
    }
    
    test_message = AgentMessage("query_agent", "search_agent", test_analysis, "query_analysis")
    result = agent.execute(test_message)
    print("Search Results:")
    print(f"Found documents: {result.content['total_found']}")