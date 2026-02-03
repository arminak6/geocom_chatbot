# 🔥 Firecrawl MCP Chatbot

A professional AI-powered chatbot that intelligently scrapes and analyzes websites using Firecrawl MCP and AWS Bedrock LLM. Built with Streamlit for an intuitive user interface.

## ✨ Features

- **🤖 Intelligent Scraping**: Automatically decides when to scrape websites vs. using cached content
- **🎯 Smart URL Detection**: Extracts and processes URLs from natural language queries
- **🔍 Deep Dive Analysis**: Discovers and analyzes multiple related pages for comprehensive answers
- **📊 Source Citations**: Provides clear references to all scraped sources
- **💬 Conversational Context**: Maintains chat history for contextual follow-up questions
- **⚡ Async Processing**: Fast, non-blocking operations for better performance

## 🏗️ Architecture

This project follows a clean, modular architecture with clear separation of concerns:

```
geocom_chatbot/
├── config.py          # Configuration management
├── utils.py           # Utility functions (URL extraction, text processing)
├── chatbot_core.py    # Core AI & business logic
├── app.py             # Streamlit UI layer
└── main.py            # Entry point wrapper
```

### Module Responsibilities

- **`config.py`** - Loads configuration from `config.json`, manages environment variables
- **`utils.py`** - Reusable utilities for text processing, URL extraction, JSON cleaning
- **`chatbot_core.py`** - LLM integration, Firecrawl MCP, scraping logic, deep dive functionality
- **`app.py`** - Streamlit interface, session management, user interactions
- **`main.py`** - Application entry point

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [Firecrawl](https://www.firecrawl.dev/) instance running (self-hosted or cloud)
- AWS account with Bedrock access
- Required Python packages (see Installation)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd geocom_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit langchain-aws mcp
   ```

3. **Configure the application**
   
   Create a `config.json` file in the project root:
   ```json
   {
     "AWS_REGION": "eu-central-1",
     "MODEL_ID": "openai.gpt-oss-20b-1:0",
     "FIRECRAWL_API_URL": "http://localhost:3002",
     "AWS_ACCESS_KEY_ID": "your-aws-access-key",
     "AWS_SECRET_ACCESS_KEY": "your-aws-secret-key"
   }
   ```

   **Note**: You can also set AWS credentials via environment variables instead of including them in `config.json`.

### Running the Application

```bash
streamlit run app.py
```

Or use the main.py wrapper:
```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

## 💡 Usage

### Basic Query
Ask about any website by including the URL in your question:
```
"What does https://example.com do?"
```

### Follow-up Questions
Continue asking questions about the same website without repeating the URL:
```
"What are their main products?"
"Do they have a contact page?"
```

### Deep Dive
Click the **"Go deeper on this site"** button to:
- Discover and rank relevant subpages
- Scrape up to 10 most relevant pages
- Get comprehensive analysis with multiple source citations

### Reset Session
Click **"Reset (new website)"** in the sidebar to start fresh with a different website.

## 🎯 How It Works

### Intelligent Decision Making
The chatbot uses a two-stage LLM approach:

1. **Planner Agent**: Decides whether to scrape, use cache, or answer directly
2. **Answer Agent**: Generates responses using scraped content and conversation history

### Web Scraping Modes

- **Single Page**: Scrapes one specific URL
- **Crawl**: Follows links and scrapes multiple pages (up to 20 pages, depth 3)
- **Deep Dive**: Maps subpages, scores them for relevance, and scrapes the top 10

### URL Scoring
Combines two approaches for ranking subpage relevance:
- **LLM-based scoring** (70% weight): Uses AI to evaluate relevance
- **Heuristic scoring** (30% weight): Keyword matching in URLs

## 📋 Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `AWS_REGION` | `eu-central-1` | AWS region for Bedrock |
| `MODEL_ID` | `openai.gpt-oss-20b-1:0` | LLM model to use |
| `FIRECRAWL_API_URL` | `http://localhost:3002` | Firecrawl instance URL |
| `AWS_ACCESS_KEY_ID` | - | AWS access key (optional if using env vars) |
| `AWS_SECRET_ACCESS_KEY` | - | AWS secret key (optional if using env vars) |

## 🛠️ Development

### Project Structure

```
├── config.py              # Configuration loader
├── utils.py               # Utility functions
│   ├── extract_json_from_model_output()
│   ├── strip_reasoning_tags()
│   └── extract_url_from_text()
├── chatbot_core.py        # Core business logic
│   ├── System prompts
│   ├── MCP integration
│   ├── Firecrawl functions
│   ├── LLM functions
│   ├── URL scoring
│   └── Main chat logic
├── app.py                 # Streamlit UI
│   ├── Session management
│   ├── UI rendering
│   └── Event handlers
└── main.py                # Entry point
```

### Adding Features

**Modify UI**: Edit `app.py`
```python
# Change title, add new buttons, modify layout
```

**Update AI Behavior**: Edit `chatbot_core.py`
```python
# Modify system prompts, scraping logic, or URL scoring
```

**Add Utilities**: Edit `utils.py`
```python
# Add new text processing or parsing functions
```

**Change Configuration**: Edit `config.py` or `config.json`
```python
# Add new configuration options
```

## 🧪 Testing

### Test Individual Modules

```python
# Test URL extraction
from utils import extract_url_from_text
url = extract_url_from_text("Visit www.example.com")
assert url == "https://www.example.com"

# Test configuration
import config
print(config.AWS_REGION)

# Test heuristic scoring
from chatbot_core import heuristic_score_url
score = heuristic_score_url("https://example.com/about", "Who are they?")
```

## 🤝 Contributing

Contributions are welcome! The modular architecture makes it easy to:

1. Fork the repository
2. Create a feature branch
3. Make your changes in the appropriate module
4. Test thoroughly
5. Submit a pull request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- [Firecrawl](https://www.firecrawl.dev/) - Web scraping infrastructure
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - LLM services
- [Streamlit](https://streamlit.io/) - Web application framework
- [LangChain](https://www.langchain.com/) - LLM integration

## 📧 Contact

[Add your contact information here]

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the correct directory
cd c:\sinergia_ak\chatbot\geocom_chatbot
```

**Firecrawl Connection Failed**
- Verify Firecrawl is running at the configured URL
- Check `FIRECRAWL_API_URL` in `config.json`

**AWS Authentication Issues**
- Verify AWS credentials are set correctly
- Check IAM permissions for Bedrock access
- Ensure the region supports your chosen model

**Module Not Found**
```bash
# Install missing dependencies
pip install streamlit langchain-aws mcp
```

## 📊 Performance

- **Single page scraping**: ~2-5 seconds
- **Deep dive analysis**: ~30-60 seconds (depends on number of pages)
- **LLM response time**: ~1-3 seconds
- **Concurrent requests**: Supported via async operations

---

**Built with ❤️ using modern Python architecture**
