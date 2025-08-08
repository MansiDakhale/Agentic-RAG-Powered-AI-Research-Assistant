# agents/report_agent.py
from agents.base_agent import BaseAgent, AgentMessage
import json
from typing import Dict
from datetime import datetime

class ReportGenerationAgent(BaseAgent):
    """Generates final comprehensive research reports"""
    
    def __init__(self, name: str, llm):
        super().__init__(name, llm)
        self.report_prompt = """
        You are a Research Report Generation Agent. Create a comprehensive, well-structured research report.
        
        Original Query: {original_query}
        Analysis Results: {analysis}
        
        Create a professional research report with the following structure:
        
        # Research Report: [Title based on query]
        
        ## Executive Summary
        Brief overview of key findings (2-3 sentences)
        
        ## Research Question
        Restate the original research question clearly
        
        ## Key Findings
        - Finding 1: [Evidence and source]
        - Finding 2: [Evidence and source] 
        - Finding 3: [Evidence and source]
        
        ## Analysis & Insights
        Detailed analysis of the themes and patterns found
        
        ## Supporting Evidence
        Specific quotes, statistics, or data points that support the findings
        
        ## Information Gaps & Limitations
        What information was missing or unclear
        
        ## Conclusions
        Summary of what the research reveals about the original question
        
        ## Sources & Methodology
        Brief note on sources analyzed and search approach
        
        ---
        *Report generated on {timestamp}*
        *Confidence Score: {confidence}/10*
        
        Make the report professional, well-formatted, and actionable. Use markdown formatting for clarity.
        """
    
    def execute(self, message: AgentMessage) -> AgentMessage:
        """Generate the final research report"""
        self.log_activity("Generating final research report")
        
        analysis_results = message.content
        analysis = analysis_results["analysis"]
        query_analysis = analysis_results["query_analysis"]
        
        # Generate the report
        report = self._generate_report(query_analysis, analysis)
        
        # Add metadata
        report_data = {
            "report_content": report,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "query": query_analysis["original_query"],
                "confidence_score": analysis.get("confidence_score", 0.7),
                "documents_analyzed": analysis_results["search_metadata"]["total_documents"],
                "processing_chain": ["query_agent", "search_agent", "analysis_agent", "report_agent"]
            }
        }
        
        self.log_activity("Research report completed successfully")
        
        return self.send_message("user", report, "final_report")
    
    def _generate_report(self, query_analysis: Dict, analysis: Dict) -> str:
        """Generate the formatted research report"""
        
        # Prepare confidence score (0-1 to 0-10 scale)
        confidence = int(analysis.get("confidence_score", 0.7) * 10)
        
        prompt = self.report_prompt.format(
            original_query=query_analysis["original_query"],
            analysis=json.dumps(analysis, indent=2),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            confidence=confidence
        )
        
        try:
            report = self.llm(prompt)
            
            # Add agent processing info
            report += f"\n\n---\n**Agent Processing Summary:**\n"
            report += f"- Query analyzed for: {query_analysis.get('main_topic', 'N/A')}\n"
            report += f"- Documents processed: {analysis.get('document_count', 'N/A')}\n"
            report += f"- Key themes identified: {len(analysis.get('themes', []))}\n"
            
            return report
            
        except Exception as e:
            # Fallback report
            return self._create_fallback_report(query_analysis, analysis)
    
    def _create_fallback_report(self, query_analysis: Dict, analysis: Dict) -> str:
        """Create a basic report if main generation fails"""
        
        findings = analysis.get("key_findings", ["No specific findings available"])
        synthesis = analysis.get("synthesis", "Analysis completed but detailed synthesis unavailable")
        
        report = f"""# Research Report: {query_analysis["original_query"]}

## Executive Summary
Research analysis has been completed for the query: "{query_analysis["original_query"]}"

## Key Findings
"""
        
        for i, finding in enumerate(findings[:5], 1):
            report += f"{i}. {finding}\n"
        
        report += f"""
## Analysis Summary
{synthesis}

## Methodology
- Documents analyzed through vector similarity search
- Multi-agent processing pipeline
- Automated synthesis and report generation

---
*Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return report

# Test the report agent
if __name__ == "__main__":
    from langchain.llms import Ollama
    
    llm = Ollama(model="llama3.1:8b")
    agent = ReportGenerationAgent("test_report_agent", llm)
    
    # Mock analysis results
    test_analysis_results = {
        "query_analysis": {"original_query": "What are AI safety trends?"},
        "analysis": {
            "key_findings": ["AI safety research is growing", "Focus on alignment and robustness"],
            "themes": ["Safety", "Research trends"],
            "synthesis": "AI safety is an active area of research with focus on alignment.",
            "confidence_score": 0.8
        },
        "search_metadata": {"total_documents": 5}
    }
    
    test_message = AgentMessage("analysis_agent", "report_agent", test_analysis_results, "analysis_complete")
    result = agent.execute(test_message)
    
    print("Generated Report:")
    print(result.content)