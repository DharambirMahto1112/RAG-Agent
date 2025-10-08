# 🤖 RAG Agent - Weather & Document Q&A Assistant (Groq + Qdrant)

A RAG (Retrieval-Augmented Generation) agent built with LangGraph, LangChain, and Streamlit that provides intelligent responses to weather queries and document-based questions. The LLM is provided by Groq's OpenAI-compatible API. Vector search uses Qdrant. Weather data comes from OpenWeather.

## 🌟 Features

- **🌤️ Weather Intelligence**: Real-time weather data for any city worldwide
- **📚 PDF Document Q&A**: Upload PDF documents and ask questions about their content using advanced RAG
- **🧠 Groq LLM**: Uses Groq's OpenAI-compatible API (fast setup)
- **🧠 Smart Routing**: Automatically determines whether queries are weather or document-related
- **🔍 Vector Search**: Advanced semantic search using Qdrant vector database
- **📊 LangSmith Integration**: Comprehensive logging and evaluation
- **🎨 Modern UI**: Beautiful Streamlit interface with chat functionality
- **🧪 Comprehensive Testing**: Full test suite with pytest

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   LangGraph     │    │   Services      │
│                 │    │   Agent         │    │                 │
│  • Chat Interface│   │                 │    │  • Weather API  │
│  • File Upload  │◄──►│  • Decision Node│◄──►│  • RAG Service  │
│  • Response Display│   │  • Weather Node │    │  • Config Mgmt  │
│                 │    │  • RAG Node     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   External APIs │
                       │                 │
                       │  • OpenWeather  │
                       │  • OpenAI       │
                       │  • Qdrant       │
                       │  • LangSmith    │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key (recommended)
- OpenWeatherMap API Key (only for weather)
- Qdrant (optional, defaults to localhost)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG_Agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example configuration
   cp config.env.example .env
   
   # Edit .env with your keys
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_BASE_URL=https://api.groq.com/openai/v1
   GROQ_MODEL=openai/gpt-oss-20b
   
   OPENWEATHER_API_KEY=your_openweather_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   QDRANT_URL=http://localhost:6333
   ```

5. **Set up Qdrant (optional)**
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or install locally
   pip install qdrant-client
   ```

## 🎯 Usage

### Option 1: Streamlit Web UI (Recommended)

```bash
python run_streamlit.py
```

Then open your browser to `http://localhost:8501`

### PDF Document Processing

1. **Upload PDF**: Use the file uploader in the Streamlit UI to upload your PDF documents
2. **Automatic Processing**: The system will extract text, generate embeddings, and store in Qdrant
3. **Ask Questions**: Query the document content using natural language
4. **Get Sources**: Receive answers with page references and confidence scores

### Option 2: Command Line Interface

```bash
python main.py
```

### Option 3: Programmatic Usage

```python
from src.agents.rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent()

# Add a document
agent.add_document("path/to/document.pdf")

# Process queries
response = agent.process_query("What's the weather in London?")
print(response["response"])

response = agent.process_query("What does the document say about AI?")
print(response["response"])
```

## 📁 Project Structure

```
RAG_Agent/
├── src/
│   ├── agents/
│   │   └── rag_agent.py          # Main LangGraph agent
│   ├── nodes/
│   │   ├── decision_node.py      # Query classification
│   │   ├── weather_node.py       # Weather processing
│   │   ├── rag_node.py          # Document Q&A
│   │   └── fallback_node.py     # Unknown queries
│   ├── services/
│   │   ├── config.py            # Configuration management
│   │   ├── weather_service.py   # OpenWeatherMap integration
│   │   └── rag_service.py      # Document processing & RAG
│   ├── tests/
│   │   ├── test_weather_service.py
│   │   ├── test_rag_service.py
│   │   └── test_decision_node.py
│   └── ui/
│       └── streamlit_app.py     # Web interface
├── main.py                      # CLI entry point
├── run_streamlit.py            # Streamlit launcher
├── run_tests.py                # Test runner
├── requirements.txt            # Dependencies
├── config.env.example         # Environment template
└── README.md                  # This file
```

## 🧪 Testing

Run the complete test suite:

```bash
python run_tests.py
```

Or run specific tests:

```bash
# All tests
pytest src/tests/ -v

# Specific test file
pytest src/tests/test_weather_service.py -v

# With coverage
pytest src/tests/ --cov=src --cov-report=html
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Groq API key (OpenAI-compatible) | Yes (LLM) | - |
| `GROQ_BASE_URL` | Groq base URL | No | `https://api.groq.com/openai/v1` |
| `GROQ_MODEL` | Groq model id | No | `openai/gpt-oss-20b` |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key | No (if no weather) | - |
| `LANGSMITH_API_KEY` | LangSmith API key for logging | No | - |
| `QDRANT_URL` | Qdrant server URL | No | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key | No | - |
| `LANGSMITH_PROJECT` | LangSmith project name | No | `RAG_Agent_Project` |
| `DEBUG` | Enable debug mode | No | `False` |

### API Keys Setup

1. **Groq API Key**
   - Visit `https://console.groq.com/keys`
   - Create a new API key
   - Add to `.env` file as `GROQ_API_KEY`

2. **OpenWeatherMap API Key**
   - Visit [OpenWeatherMap](https://openweathermap.org/api)
   - Sign up for free account
   - Get your API key
   - Add to `.env` file

3. **LangSmith API Key (Optional)**
   - Visit [LangSmith](https://smith.langchain.com/)
   - Create account and get API key
   - Add to `.env` file for logging

## 📊 LangSmith Integration

The agent automatically logs all interactions to LangSmith when configured:

- **Run Tracking**: Every query execution is logged
- **Node Performance**: Individual node performance metrics
- **Error Tracking**: Detailed error logging and debugging
- **Evaluation**: Response quality and relevance scoring

## 🎨 Streamlit Features

- **💬 Interactive Chat**: Real-time conversation interface
- **📁 Document Upload**: Drag-and-drop PDF upload
- **🎯 Smart Responses**: Context-aware answer generation
- **📚 Source Citations**: Transparent source attribution
- **🎨 Modern UI**: Clean, responsive design
- **📊 Confidence Scores**: Response quality indicators

## 🔍 Query Examples

### Weather Queries
```
"What's the weather in London?"
"Temperature in Paris"
"Is it raining in New York?"
"Weather forecast for Tokyo"
"How hot is it in Dubai?"
```

### PDF Document Queries
```
"What does the PDF document say about machine learning?"
"Summarize the key points from the document"
"Explain the methodology section"
"What are the main findings in the PDF?"
"Find information about neural networks in the document"
"What is mentioned about AI ethics in the PDF?"
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd RAG_Agent
   
   # Activate virtual environment
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **API Key Errors**
   ```bash
   # Check your .env file
   cat .env
   
   # Verify API keys are valid
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

3. **Qdrant Connection Issues**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/collections
   
   # Start Qdrant with Docker
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Streamlit Issues**
   ```bash
   # Clear Streamlit cache
   streamlit cache clear
   
   # Run with debug mode
   streamlit run src/ui/streamlit_app.py --logger.level debug
   ```

## 📈 Performance Optimization

### For Production Deployment

1. **Qdrant Configuration**
   ```yaml
   # qdrant-config.yaml
   storage:
     performance:
       max_optimization_threads: 4
       indexing_threshold: 20000
   ```

2. **Embedding Model Selection**
   - Current: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
   - Alternative: `sentence-transformers/all-mpnet-base-v2` (768 dims, better quality)

3. **Chunk Size Optimization**
   ```python
   # In rag_service.py
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,      # Adjust based on content
       chunk_overlap=200,    # Maintain context
   )
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - State machine framework
- [Streamlit](https://streamlit.io/) - Web application framework
- [Qdrant](https://qdrant.tech/) - Vector database
- [OpenWeatherMap](https://openweathermap.org/) - Weather data API
- [OpenAI](https://openai.com/) - Language models

## 📞 Support

For questions, issues, or contributions:

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-repo/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📧 **Contact**: [your-email@example.com](mailto:your-email@example.com)

---

**Built with ❤️ using LangGraph, LangChain, and Streamlit**
