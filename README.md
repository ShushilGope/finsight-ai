# 📊 FinSight AI — Finance Analyst Agentic Workflow

A multi-agent equity research assistant built with **CrewAI**, **Gemini**, **LangChain**, and **Streamlit**.

---

## Architecture

```
User input (ticker / query)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  CrewAI Orchestrator  (sequential process)          │
│                                                     │
│  Research Agent ──context──► Analyst Agent          │
│       │                           │                 │
│  SerperWebSearch            SerperFilings           │
│  SerperNews                 ScrapeWebsite           │
│  ScrapeWebsite                    │                 │
│                           context ▼                 │
│                      Validator Agent  ◄─── NEW      │
│                      (fact-checker)                 │
│                           │                         │
│                      context ▼                      │
│                      Writer Agent                   │
│                      FileWriterTool                 │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Final Report  (Markdown / .md download)
```

### Agents
| Agent | Role | Tools |
|-------|------|-------|
| Research Agent | Gather live price, news, earnings, analyst ratings | SerperWebSearch, SerperNews, ScrapeWebsite |
| Analyst Agent | Valuation, DCF, bull/base/bear scenarios | SerperFilings, ScrapeWebsite |
| Validator Agent | Fact-check figures, catch hallucinations | SerperFilings, ScrapeWebsite |
| Writer Agent | Produce structured Markdown report | ScrapeWebsite |

---

## Project Structure

```
finance_analyst/
├── app.py                  # Streamlit frontend (entry point)
├── crew.py                 # CrewAI orchestrator + non-blocking CrewRunner
├── requirements.txt
├── .env.example            # Copy to .env and fill in keys
├── agents/
│   └── finance_agents.py   # Agent factory functions
├── tasks/
│   └── finance_tasks.py    # Task definitions with context chaining
├── tools/
│   └── finance_tools.py    # Custom BaseTool wrappers + retry logic
├── utils/
│   └── config.py           # LLM factory, env validation, @lru_cache
├── .streamlit/
│   ├── config.toml         # Theme + server config
│   └── secrets.toml        # API keys (NEVER commit)
└── output/                 # Generated reports land here
```

---

## Quick Start (Local)

### 1. Clone & install

```bash
git clone <your-repo>
cd finance_analyst
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and fill in GOOGLE_API_KEY and SERPER_API_KEY
```

Get your keys:
- **Gemini**: https://aistudio.google.com/ (free tier available)
- **Serper**: https://serper.dev/ (2,500 free searches/month)

### 3. Run Streamlit

```bash
streamlit run app.py
```

### 4. CLI usage (no UI)

```bash
python crew.py AAPL "What is the risk from AI chip regulations?"
```

---

## Cloud Deployment

### Streamlit Community Cloud (free)

1. Push to a public GitHub repo (check `.gitignore` — never commit `.env` or `secrets.toml`).
2. Go to https://share.streamlit.io → New app → select `app.py`.
3. Under **Settings → Secrets**, paste:

```toml
GOOGLE_API_KEY = "your_key_here"
SERPER_API_KEY = "your_key_here"
```

4. Deploy.

### Railway / Render

Set `GOOGLE_API_KEY` and `SERPER_API_KEY` as environment variables in the platform dashboard.
Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

---

## Configuration

All settings can be adjusted via the Streamlit sidebar or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_MODEL` | `gemini-1.5-flash` | Model: `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash` |
| `LLM_TEMPERATURE` | `0.2` | 0 = deterministic, 1 = creative |
| `MAX_ITER` | `8` | Max agent iterations (prevents runaway loops) |
| `OUTPUT_DIR` | `output` | Directory for saved reports |

---

## Key Design Decisions

### Why a Validator agent?
Finance LLMs can hallucinate numbers. A dedicated critic agent cross-checks every figure from the Analysis agent against raw research data *before* the report is written. This catches errors like a price target below the current price, or a P/E ratio that contradicts the sourced data.

### Why `queue.Queue` + `threading.Thread`?
Streamlit's execution model reruns the full script on each interaction. Running `crew.kickoff()` synchronously would block the event loop and freeze the UI. The `CrewRunner` class pushes `AgentLog` objects into a `Queue`, which `app.py` drains on each `st.rerun()` cycle — giving live log streaming without blocking.

### Why `@lru_cache` on `get_llm()`?
CrewAI instantiates each agent fresh per run, which would create multiple `ChatGoogleGenerativeAI` objects. Caching by `(model, temperature)` avoids redundant HTTP handshakes and keeps memory flat.

---

## Disclaimer

Reports generated by FinSight AI are for **informational and educational purposes only**. They do not constitute financial advice. Always verify information independently before making investment decisions.
