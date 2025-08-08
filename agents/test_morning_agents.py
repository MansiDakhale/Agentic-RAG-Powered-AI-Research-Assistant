# test_morning_agents.py
from langchain.llms import Ollama
from utils.vector_store import VectorStore
from agents.query_agent import QueryUnderstandingAgent
from agents.search_agent import SearchAgent
from agents.base_agent import AgentMessage

def test_morning_pipeline():
    """Test the query and search agents together"""
    
    # Initialize components
    llm = Ollama(model="llama3.1:8b")
    vector_store = VectorStore()
    
    # Add some test documents to vector store
    test_docs = [
        "AI safety research focuses on alignment, robustness, and interpretability of large language models.",
        "Recent trends in machine learning include transformer architectures and few-shot learning.",
        "Ethical AI considerations include bias mitigation, fairness, and transparency in decision making."
    ]
    vector_store.add_documents(test_docs)
    
    # Initialize agents
    query_agent = QueryUnderstandingAgent("query_agent", llm)
    search_agent = SearchAgent("search_agent", llm, vector_store)
    
    # Test pipeline
    user_query = "What are the current trends in AI safety?"
    print(f"Testing with query: {user_query}")
    
    # Step 1: Query analysis
    query_msg = AgentMessage("user", "query_agent", user_query, "query")
    analysis_result = query_agent.execute(query_msg)
    
    print("\n=== Query Analysis ===")
    print(f"Main topic: {analysis_result.content.get('main_topic')}")
    
    # Step 2: Document search  
    search_result = search_agent.execute(analysis_result)
    
    print("\n=== Search Results ===")
    print(f"Documents found: {search_result.content['total_found']}")
    print(f"Top result: {search_result.content['documents'][0]['content'][:100]}...")
    
    print("\n✅ Morning pipeline working!")

if __name__ == "__main__":
    test_morning_pipeline()