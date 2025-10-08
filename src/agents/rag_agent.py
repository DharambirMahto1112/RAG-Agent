"""
Main RAG Agent using LangGraph for orchestration.
"""
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain.schema import BaseMessage
from langsmith import Client
from src.nodes.decision_node import DecisionNode
from src.nodes.weather_node import WeatherNode
from src.nodes.rag_node import RAGNode
from src.nodes.fallback_node import FallbackNode
from src.services.config import get_settings

class RAGAgent:
    """Main agent that orchestrates weather and document queries."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize nodes
        self.decision_node = DecisionNode()
        self.weather_node = WeatherNode()
        self.rag_node = RAGNode()
        self.fallback_node = FallbackNode()
        
        # Initialize LangSmith client if API key is provided
        self.langsmith_client = None
        if self.settings.langsmith_api_key:
            self.langsmith_client = Client(
                api_key=self.settings.langsmith_api_key,
                api_url=self.settings.langsmith_endpoint
            )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the state graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("decision", self._decision_wrapper)
        workflow.add_node("weather", self._weather_wrapper)
        workflow.add_node("rag", self._rag_wrapper)
        workflow.add_node("fallback", self._fallback_wrapper)
        
        # Set entry point
        workflow.set_entry_point("decision")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "decision",
            self._route_decision,
            {
                "weather": "weather",
                "rag": "rag",
                "fallback": "fallback"
            }
        )
        
        # Add edges to end
        workflow.add_edge("weather", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("fallback", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _decision_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for decision node with logging."""
        try:
            if self.langsmith_client:
                # Log to LangSmith
                self.langsmith_client.create_run(
                    name="decision_node",
                    run_type="chain",
                    inputs={"query": state.get("query", "")},
                    project_name=self.settings.langsmith_project
                )
            
            return self.decision_node.process(state)
        except Exception as e:
            state["error"] = f"Decision node error: {str(e)}"
            return state
    
    def _weather_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for weather node with logging."""
        try:
            if self.langsmith_client:
                self.langsmith_client.create_run(
                    name="weather_node",
                    run_type="chain",
                    inputs={"query": state.get("query", "")},
                    project_name=self.settings.langsmith_project
                )
            
            return self.weather_node.process(state)
        except Exception as e:
            state["error"] = f"Weather node error: {str(e)}"
            return state
    
    def _rag_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for RAG node with logging."""
        try:
            if self.langsmith_client:
                self.langsmith_client.create_run(
                    name="rag_node",
                    run_type="chain",
                    inputs={"query": state.get("query", "")},
                    project_name=self.settings.langsmith_project
                )
            
            return self.rag_node.process(state)
        except Exception as e:
            state["error"] = f"RAG node error: {str(e)}"
            return state
    
    def _fallback_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for fallback node with logging."""
        try:
            if self.langsmith_client:
                self.langsmith_client.create_run(
                    name="fallback_node",
                    run_type="chain",
                    inputs={"query": state.get("query", "")},
                    project_name=self.settings.langsmith_project
                )
            
            return self.fallback_node.process(state)
        except Exception as e:
            state["error"] = f"Fallback node error: {str(e)}"
            return state
    
    def _route_decision(self, state: Dict[str, Any]) -> str:
        """Route to appropriate node based on decision."""
        return self.decision_node.should_continue(state)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the agent.
        
        Args:
            query: User query string
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            # Initialize state
            # Detect if we have any documents indexed for RAG routing decisions
            has_docs = False
            try:
                has_docs = self.rag_node.rag_service.has_documents()
            except Exception:
                has_docs = False

            initial_state = {
                "query": query,
                "messages": [],
                "response": "",
                "response_type": "",
                "error": None,
                "has_documents": has_docs
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Log the complete run to LangSmith
            if self.langsmith_client:
                self.langsmith_client.create_run(
                    name="rag_agent_complete",
                    run_type="chain",
                    inputs={"query": query},
                    outputs={"response": final_state.get("response", "")},
                    project_name=self.settings.langsmith_project
                )
            
            return final_state
            
        except Exception as e:
            return {
                "query": query,
                "response": f"❌ Error processing query: {str(e)}",
                "response_type": "error",
                "error": str(e)
            }
    
    def add_document(self, pdf_path: str) -> bool:
        """
        Add a PDF document to the knowledge base.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            return self.rag_node.rag_service.process_document_and_store(pdf_path)
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent's capabilities."""
        return {
            "capabilities": [
                "Weather queries for any city",
                "Document Q&A using RAG",
                "Intelligent query routing",
                "LangSmith logging and evaluation"
            ],
            "supported_formats": ["PDF"],
            "vector_database": "Qdrant",
            "llm_provider": "Groq (OpenAI-compatible)",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
