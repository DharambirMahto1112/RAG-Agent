"""
Streamlit UI for the RAG Agent application.
"""
import streamlit as st
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.rag_agent import RAGAgent
from src.services.config import get_settings

# Page configuration
st.set_page_config(
    page_title="RAG Agent - Weather & Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #000000; /* Ensure agent reply text is black */
    }
    .weather-response {
        border-left-color: #ff7f0e;
    }
    .document-response {
        border-left-color: #2ca02c;
    }
    .error-response {
        border-left-color: #d62728;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = RAGAgent()
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            st.session_state.agent = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def display_chat_history():
    """Display chat history."""
    for i, message in enumerate(st.session_state.chat_history):
        with st.container():
            if message["type"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                response_type = message.get("response_type", "unknown")
                css_class = f"{response_type}-response"
                
                st.markdown(f"""
                <div class="response-box {css_class}">
                    <strong>Agent:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show additional info for document responses
                if response_type == "document" and "sources" in message:
                    with st.expander("📚 Sources"):
                        for j, source in enumerate(message["sources"], 1):
                            st.markdown(f"""
                            **{j}. {source['source']}** (Page {source['page_range']})
                            *Score: {source['score']:.3f}*
                            
                            {source['text'][:200]}...
                            """)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Agent</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent Weather & Document Q&A Assistant**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Configuration")
        
        # Check configuration
        settings = get_settings()
        from pathlib import Path
        has_local_model = Path(settings.local_model_path).exists()
        has_openai = bool(settings.openai_api_key)
        has_groq = bool(settings.groq_api_key)
        has_ollama = False
        configured = has_local_model or has_openai or has_groq
        if has_groq:
            config_status = "✅ Configured (Groq API)"
        elif has_local_model:
            config_status = "✅ Configured (Local LLM)"
        elif has_openai:
            config_status = "✅ Configured (OpenAI)"
        else:
            config_status = "❌ Not configured"
        st.markdown(f"**Status:** {config_status}")
        
        if not configured:
            st.error("Model not configured. Set GROQ_API_KEY (recommended), or a local model or OpenAI key in .env")
            st.markdown("""
            Options:
            - GROQ_API_KEY: use Groq OpenAI-compatible endpoint (fastest to set up)
            - LOCAL_MODEL_PATH: path to local model directory (legacy)
            - OPENAI_API_KEY (optional): only if using OpenAI
            - OPENWEATHER_API_KEY (optional): only if using weather
            - QDRANT_URL (optional)
            """)
        
        st.markdown("## 📁 Document Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=['pdf'],
            help="Upload a PDF file to add it to the knowledge base",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Show file info
            st.info(f"📄 Selected: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            if st.button("📚 Process Document", type="primary"):
                with st.spinner("Processing PDF document..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process document
                        if st.session_state.agent:
                            success = st.session_state.agent.add_document(temp_path)
                            if success:
                                st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
                                st.session_state.uploaded_files.append(uploaded_file.name)
                                
                                # Show processing details
                                st.info("📊 Document has been added to the knowledge base. You can now ask questions about its content!")
                            else:
                                st.error("❌ Failed to process document. Please check if the PDF contains readable text.")
                        else:
                            st.error("❌ Agent not initialized. Please check your configuration.")
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")
                        # Clean up temp file on error
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        # Show uploaded files
        if st.session_state.uploaded_files:
            st.markdown("### 📚 Processed Documents")
            for file in st.session_state.uploaded_files:
                st.markdown(f"• {file}")
        
        # Agent info
        if st.session_state.agent:
            st.markdown("## 🤖 Agent Info")
            info = st.session_state.agent.get_agent_info()
            st.markdown("**Capabilities:**")
            for capability in info["capabilities"]:
                st.markdown(f"• {capability}")
    
    # Main chat interface
    st.markdown("## 💬 Chat")
    
    # Display chat history
    display_chat_history()
    
    # Clear input if flagged (must be done BEFORE widget instantiation)
    if st.session_state.get("clear_input"):
        st.session_state.user_input = ""
        del st.session_state["clear_input"]

    # Chat input
    user_input = st.text_input(
        "Ask me anything about weather or your documents:",
        placeholder="e.g., 'What's the weather in Paris?' or 'What does the document say about...?'",
        key="user_input"
    )
    
    # Process query (only on button click to avoid loops)
    if st.button("Send", type="primary"):
        if st.session_state.user_input and st.session_state.agent:
            # Add user message to history
            st.session_state.chat_history.append({
                "type": "user",
                "content": st.session_state.user_input
            })
            
            # Process query
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.process_query(st.session_state.user_input)
                    
                    # Add agent response to history
                    st.session_state.chat_history.append({
                        "type": "agent",
                        "content": response.get("response", "No response generated"),
                        "response_type": response.get("response_type", "unknown"),
                        "sources": response.get("sources", []),
                        "confidence": response.get("confidence", 0)
                    })
                    
                    # Flag to clear input on next render (cannot modify after widget instantiation)
                    st.session_state.clear_input = True
                    
                    # Rerun to update display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.session_state.chat_history.append({
                        "type": "agent",
                        "content": f"❌ Error: {str(e)}",
                        "response_type": "error"
                    })
                    # Flag to clear input on error as well
                    st.session_state.clear_input = True
                    st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        Built with LangGraph, LangChain, Streamlit, and Qdrant
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
