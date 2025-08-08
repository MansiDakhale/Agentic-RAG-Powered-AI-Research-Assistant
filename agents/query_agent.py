# agents/query_agent.py
from agents.base_agent import BaseAgent, AgentMessage
import json

class QueryUnderstandingAgent(BaseAgent):
    """Analyzes user queries and extracts intent, entities, and search strategy"""
    
    def __init__(self, name: str, llm):
        super().__init__(name, llm)
        self.analysis_prompt = """
        You are a Query Analysis Agent. Analyze the user's research query and provide structured information.
        
        User Query: {query}
        
        Please provide a JSON response with the following structure:
        {{
            "main_topic": "primary subject of the query",
            "sub_topics": ["list", "of", "related", "topics"],
            "query_type": "factual|analytical|comparative|exploratory",
            "entities": ["key", "entities", "to", "search", "for"],
            "search_strategy": "how to approach the search",
            "expected_answer_type": "summary|detailed_analysis|comparison|trends",
            "refined_query": "optimized version of the original query for search"
        }}
        
        Focus on being precise and actionable for the next agents in the pipeline.
        """
    
    def execute(self, message: AgentMessage) -> AgentMessage:
        """Analyze the query and extract structured information"""
        self.log_activity("Analyzing user query")
        
        user_query = message.content
        
        # Generate analysis
        prompt = self.analysis_prompt.format(query=user_query)
        analysis_response = self.llm(prompt)
        
        try:
            # Parse JSON response
            analysis = json.loads(analysis_response)
            
            # Add original query for reference
            analysis["original_query"] = user_query
            
            self.log_activity(f"Identified topic: {analysis.get('main_topic', 'Unknown')}")
            self.log_activity(f"Query type: {analysis.get('query_type', 'Unknown')}")
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            analysis = {
                "original_query": user_query,
                "main_topic": "general research",
                "sub_topics": [],
                "query_type": "exploratory",
                "entities": [],
                "search_strategy": "broad search",
                "expected_answer_type": "summary",
                "refined_query": user_query
            }
        
        return self.send_message("search_agent", analysis, "query_analysis")

# Test the agent
if __name__ == "__main__":
    from langchain.llms import Ollama
    
    llm = Ollama(model="llama3.2:3b")
    agent = QueryUnderstandingAgent("test_query_agent", llm)
    
    test_message = AgentMessage("user", "query_agent", 
                               "What are the latest trends in AI safety research?", 
                               "query")
    
    result = agent.execute(test_message)
    print("Query Analysis Result:")
    print(json.dumps(result.content, indent=2))