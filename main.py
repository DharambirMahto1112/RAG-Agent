"""
Main entry point for the RAG Agent application.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.agents.rag_agent import RAGAgent
from src.services.config import get_settings

def main():
    """Main function to run the RAG Agent."""
    print("🤖 RAG Agent - Weather & Document Q&A Assistant")
    print("=" * 50)
    
    try:
        # Initialize agent
        agent = RAGAgent()
        print("✅ Agent initialized successfully!")
        
        # Display agent info
        info = agent.get_agent_info()
        print("\n📋 Agent Capabilities:")
        for capability in info["capabilities"]:
            print(f"  • {capability}")
        
        print(f"\n🔧 Configuration:")
        print(f"  • Vector Database: {info['vector_database']}")
        print(f"  • LLM Provider: {info['llm_provider']}")
        print(f"  • Embedding Model: {info['embedding_model']}")
        
        # Interactive chat loop
        print("\n💬 Interactive Chat (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process query
                print("🤔 Thinking...")
                response = agent.process_query(user_input)
                
                # Display response
                print(f"\n🤖 Agent: {response.get('response', 'No response generated')}")
                
                # Show additional info for document responses
                if response.get('response_type') == 'document' and response.get('sources'):
                    print("\n📚 Sources:")
                    for i, source in enumerate(response['sources'], 1):
                        print(f"  {i}. {source['source']} (Page {source['page_range']}, Score: {source['score']:.3f})")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("\nPlease check your configuration and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
