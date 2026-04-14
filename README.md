# QA Assistant Chatbot

A Streamlit chatbot for QA professionals powered by Claude and traced with LangSmith.

## Setup

### 1. Enter the project directory
```bash
cd qa-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=qa-chatbot
```

- **ANTHROPIC_API_KEY**: Get from https://console.anthropic.com/
- **LANGCHAIN_API_KEY**: Get from https://smith.langchain.com/ (Settings → API Keys)
- **LANGCHAIN_PROJECT**: The project name that will appear in LangSmith

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked API calls — no real keys needed.

## Project Structure

```
qa-chatbot/
├── .env                  # API keys (never committed)
├── .env.example          # Template for required env vars
├── .gitignore
├── requirements.txt
├── app.py                # Streamlit UI and session state
├── chatbot.py            # Claude API logic and LangSmith tracing
├── tests/
│   └── test_chatbot.py   # Unit tests for get_response
└── README.md
```

## LangSmith Tracing

Every call to Claude is traced automatically via the `@traceable` decorator in `chatbot.py`.
Each trace is tagged with:
- `session_id` — unique per browser session
- `timestamp` — ISO-8601 UTC time of the call
- `project` — value of `LANGCHAIN_PROJECT` env var

View traces at: https://smith.langchain.com/projects/qa-chatbot
