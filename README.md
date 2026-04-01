# Firecrawl MCP Chatbot

A Streamlit chatbot that analyzes websites using Firecrawl MCP and AWS Bedrock.

## Features

- Intelligent scraping vs. cached-content decisions
- URL extraction from natural-language questions
- Deep-dive analysis across multiple related pages
- Source citations in answers
- Session-aware follow-up questions
- Async processing for scraping and answering

## Project Structure

```text
geocom_chatbot/
|-- src/
|   |-- app.py
|   |-- chatbot_core.py
|   |-- config.py
|   |-- mcp_firecrawl.py
|   `-- utils.py
|-- config.json.example
|-- compose.yaml
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Prerequisites

- Python 3.8+
- Node.js 18+ for Firecrawl MCP
- A Firecrawl instance
- AWS account with Bedrock access

## Configuration

The app can read configuration from either:

- Environment variables
- `config.json` in the repo root

Environment variables take precedence over `config.json`.

Example `config.json`:

```json
{
  "AWS_REGION": "eu-central-1",
  "MODEL_ID": "openai.gpt-oss-120b-1:0",
  "FIRECRAWL_API_URL": "http://localhost:3002",
  "AWS_ACCESS_KEY_ID": "your-aws-access-key-id-here",
  "AWS_SECRET_ACCESS_KEY": "your-aws-secret-access-key-here"
}
```

Important settings:

- `AWS_REGION`
- `MODEL_ID`
- `FIRECRAWL_API_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the config template if you want file-based configuration:

```bash
# Windows
copy config.json.example config.json

# macOS/Linux
cp config.json.example config.json
```

3. Start the app:

```bash
streamlit run src/app.py
```

The app is available at `http://localhost:8501`.

## Run with Docker

1. Copy the environment template:

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

2. Update `.env` with your real values.

If Firecrawl is running on your host machine, this default is usually correct:

```bash
FIRECRAWL_API_URL=http://host.docker.internal:3002
```

3. Build and start the container:

```bash
docker compose up --build
```

4. Open `http://localhost:8501`

The Docker image includes Python, Node.js, and a preinstalled `firecrawl-mcp` binary.

## Usage

Ask about any site by including a URL, for example:

```text
What does https://example.com do?
```

You can then ask follow-up questions without repeating the URL.

Use the deep-dive button to expand analysis across more pages of the same site.

## Development Notes

- UI lives in `src/app.py`
- Core chatbot logic lives in `src/chatbot_core.py`
- Firecrawl MCP helpers live in `src/mcp_firecrawl.py`
- Utility helpers live in `src/utils.py`
- Configuration logic lives in `src/config.py`

## Quick Test Snippets

```python
from src.utils import extract_url_from_text

url = extract_url_from_text("Visit www.example.com")
assert url == "https://www.example.com"
```

```python
from src import config

print(config.AWS_REGION)
```

## Troubleshooting

### Firecrawl connection failed

- Verify Firecrawl is running
- Check `FIRECRAWL_API_URL`
- If you are using Docker and Firecrawl runs on your host, prefer `http://host.docker.internal:3002`

### AWS authentication issues

- Verify your credentials are correct
- Check IAM permissions for Bedrock
- Confirm the configured AWS region supports your model

### `npx` or Node.js not found

Install Node.js 18+ and verify:

```bash
node --version
npx --version
```
