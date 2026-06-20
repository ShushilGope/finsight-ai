# ◈ FinSight — Multi-Agent AI Equity Research

A finance research assistant where four AI agents — Research, Analysis, Validation, and Writing — collaborate in sequence to turn a stock ticker into a structured equity report. Built with **CrewAI**, **Google Gemini**, and **Streamlit**.

**Live app:** https://fininsight-ai.streamlit.app/

---

## Why four agents instead of one prompt?

A single LLM call asked to "research and analyze a stock" tends to blend fact-finding with speculation, and has no mechanism to catch its own errors. Splitting the work into four specialized agents — each with a narrow job and its own tools — gives the pipeline two things a single prompt can't:

1. **Separation of concerns.** The Research agent only gathers data with citations. The Analyst only reasons over that data. Neither is trying to do both at once.
2. **A built-in fact-checker.** The Validator agent sits between Analysis and Writing specifically to catch hallucinated figures before they reach the final report — see [The Validator](#the-validator-why-it-exists) below.

---

## Architecture

```
                         User input (ticker + optional focus question)
                                          │
                                          ▼
                         ┌──────────────────────────────────┐
                         │   CrewAI Orchestrator             │
                         │   (Process.sequential)            │
                         └──────────────────────────────────┘
                                          │
        ┌─────────────────┬──────────────┴───────────┬─────────────────┐
        ▼                 ▼                          ▼                 ▼
┌───────────────┐ ┌───────────────┐         ┌───────────────┐ ┌───────────────┐
│   RESEARCH    │ │   ANALYST     │         │   VALIDATOR   │ │   WRITER      │
│   Agent       │─▶   Agent        │────────▶│   Agent       │─▶   Agent       │
│               │ │               │         │               │ │               │
│ Web + news    │ │ Valuation,    │         │ Cross-checks  │ │ Drafts final  │
│ search, scrape│ │ bull/base/bear│         │ every figure  │ │ Markdown      │
│               │ │ scenarios     │         │ against       │ │ report, flags │
│               │ │               │         │ research data │ │ unverified    │
│               │ │ outputs       │         │               │ │ data inline   │
│               │ │ structured    │         │ outputs       │ │               │
│               │ │ JSON too      │         │ "Flagged      │ │               │
│               │ │ (for charts)  │         │ Figures" list │ │               │
└───────────────┘ └───────────────┘         └───────────────┘ └───────────────┘
                                                                       │
                                                                       ▼
                                                          Markdown report + structured
                                                          data → rendered in Streamlit
                                                          (report, charts, live diagram)
```

Each arrow is CrewAI's `context=[...]` mechanism — every agent receives the **full output** of every upstream agent, not just a summary. The Validator sees both the raw research and the analyst's claims; the Writer sees all three.

### Agent reference

| Agent | Role | Tools | Key output |
|---|---|---|---|
| **Research** | Gather live price, earnings, news, analyst ratings, competitor data | `SerperWebSearchTool`, `SerperNewsSearchTool`, `ScrapeWebsiteTool` | Cited Markdown research dump |
| **Analyst** | Valuation, financial health, bull/base/bear price targets | `SerperFilingsSearchTool`, `ScrapeWebsiteTool` | Written memo **+** structured JSON (`StructuredAnalysis` schema) |
| **Validator** | Cross-checks every number in the Analyst's output against the Research agent's raw data | `SerperFilingsSearchTool`, `ScrapeWebsiteTool` | Status (PASS / PASS WITH CAVEATS / FAIL) + explicit **Flagged Figures** list |
| **Writer** | Synthesizes everything into a client-ready report | `ScrapeWebsiteTool` | Final Markdown report with required sections, tables, and ⚠️ markers on unverified figures |

---

## The Validator: why it exists

LLMs asked for financial analysis will confidently state numbers that sound plausible but were never actually confirmed by a source — a classic hallucination failure mode, and a dangerous one in a finance context.

The Validator agent's only job is to be skeptical: it re-reads the Analyst's claims against the Research agent's cited data and produces an explicit **Flagged Figures** list — every number it couldn't trace back to a source.

Critically, a `FAIL` status does **not** halt the pipeline. Blocking the report entirely on any uncertainty would make the tool useless for real-world data, where some figures are always estimates or unavailable. Instead, the Writer agent is instructed to:

1. Open the report with a **Data Notice** banner stating how many figures are unverified
2. Prefix every specific flagged figure inline with ⚠️ wherever it appears

This way the report stays useful while being honest about its own confidence — visible directly in the UI as a gold banner plus inline warning icons throughout the rendered report.

---

## Structured output: how the charts work

By default, CrewAI agents only produce free text — great for a report, useless for a chart. The Analyst task uses CrewAI's `output_pydantic` parameter to force a second, parallel output: a typed object (`StructuredAnalysis`, defined in `schemas/analysis_schema.py`) containing the bull/base/bear price targets and valuation ratios as actual numbers, not prose.

```python
class StructuredAnalysis(BaseModel):
    ticker: str
    current_price: float
    scenarios: list[ScenarioTarget]       # exactly Bull, Base, Bear
    valuation_ratios: list[ValuationRatio]
    recommendation: str
    conviction: str
```

`crew.py` pulls this object off the Analyst task's output after the run and passes it to the Streamlit UI, which renders it with Plotly. If the model's output doesn't parse cleanly into the schema on a given run, the UI quietly skips the charts — the written report is never affected by a charting failure.

---

## The live architecture diagram

The Streamlit UI renders the same 4-agent diagram shown above as live SVG, with the currently-executing node glowing gold and completed nodes turning sage green — built specifically so a demo audience can see "what's happening under the hood" in real time, not just a generic spinner.

**The interesting bug this surfaced:** CrewAI's `step_callback` (which fires on every intermediate LLM step) doesn't reliably expose which agent is running — its payload shape has varied across CrewAI versions, and in this version it has no `.agent` attribute at all. Relying on it silently broke the live diagram.

The fix was to stop trying to parse the unreliable per-step object, and instead add a second hook, `task_callback`, which CrewAI *does* guarantee fires once per completed task with a stable `.agent` string. Since the four tasks always run in the same fixed order, "which agent is active right now" becomes a simple lookup: *the first stage that hasn't completed yet* — no fragile parsing required.

---

## Project structure

```
finance_analyst/
├── app.py                   # Streamlit frontend — UI, live diagram, charts, log streaming
├── crew.py                  # Orchestrator: builds the Crew, runs it in a background thread
├── finance_agents.py        # Agent factory functions (role, goal, backstory, tools, llm)
├── finance_tasks.py         # Task definitions — descriptions, expected_output, context chaining
├── finance_tools.py         # Custom CrewAI tools wrapping the Serper.dev API + retry logic
├── config.py                # Gemini LLM factory (@lru_cache), env validation
├── schemas/
│   └── analysis_schema.py   # Pydantic schema for the Analyst's structured output
├── requirements.txt
├── runtime.txt              # Pins Python 3.11 for Streamlit Cloud
├── .streamlit/
│   ├── config.toml          # Theme (soft cream/gold palette)
│   └── secrets.toml         # API keys — never committed, see .gitignore
└── output/                  # Generated reports land here (gitignored)
```

> Note: this is a flat layout (no `agents/`/`tasks/`/`tools/`/`utils/` subpackages) — all modules import each other directly by filename, e.g. `from finance_tasks import build_tasks`.

---

## Quick start (local)

```bash
git clone https://github.com/ShushilGope/finsight-ai.git
cd finsight-ai
python3.11 -m venv .venv && source .venv/bin/activate   # Python 3.11 specifically — see note below
pip install -r requirements.txt
```

**Set your API keys:**

```bash
cat > .env << 'EOF'
GOOGLE_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key
GEMINI_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.2
MAX_ITER=8
EOF
```

- **Gemini key**: https://aistudio.google.com/
- **Serper key**: https://serper.dev/ (2,500 free searches/month)

**Run it:**

```bash
streamlit run app.py
```

**Or run headless via CLI** (faster for debugging — prints the raw report to terminal):

```bash
python crew.py AAPL "what's the risk from rising rates?"
```

### Why Python 3.11 specifically

`crewai` and several of its dependencies (`tiktoken` in particular) don't yet have prebuilt wheels for the newest Python releases, and fail to compile from source on 3.13+. `runtime.txt` pins Streamlit Cloud to 3.11 for the same reason — match it locally to avoid dependency hell.

---

## Deployment (Streamlit Community Cloud)

1. Push to GitHub — confirm `.env`, `.streamlit/secrets.toml`, and `.venv/` are excluded via `.gitignore`.
2. On [share.streamlit.io](https://share.streamlit.io): New app → select the repo → main file `app.py`.
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GOOGLE_API_KEY = "your_key"
   SERPER_API_KEY = "your_key"
   ```
4. Deploy.

**Security note:** the sidebar's API key fields are intentionally never pre-filled with the host's real keys — they start empty and only let a visitor optionally supply their *own* key. Pre-filling them with `value=os.getenv(...)` would leak the host's secret to every visitor, which is exactly what shipped in an earlier version of this app before being caught and fixed.

---

## Key design decisions

**Why CrewAI's native `LLM` class instead of LangChain's `ChatGoogleGenerativeAI`?**
Early versions of this project routed Gemini through `langchain-google-genai`. CrewAI's LLM calls go through LiteLLM internally regardless, which expects a `gemini/<model>` provider-prefixed string — using CrewAI's own `LLM` class directly avoids an unnecessary translation layer and a dependency that wasn't adding value.

**Why `queue.Queue` + `threading.Thread` instead of just calling `crew.kickoff()` directly?**
Streamlit reruns the entire script on every interaction. A synchronous, multi-minute `kickoff()` call would freeze the UI for the whole run. `CrewRunner` kicks off the crew in a background thread and pushes `AgentLog` entries into a thread-safe queue; the main Streamlit thread polls that queue on a `st.rerun()` loop, giving live log streaming without blocking.

**Why capture full tracebacks instead of just `str(exception)`?**
An earlier version only stored the exception's string message, which made debugging silent failures (e.g. a parsing issue inside a callback) nearly impossible from the UI alone. `crew.py` now captures `traceback.format_exc()` on any failure and surfaces it in an expandable panel in the UI.

**Why does the Writer always run, even on a Validator `FAIL`?**
Blocking the entire pipeline on any single unverifiable figure would make the tool unusable against real-world financial data, which always has some gaps. Transparency (flagging the gap) beats refusal (hiding the report) for this use case.

---

## Configuration reference

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Also supports `gemini-2.5-pro`, `gemini-3-flash` |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic; adjustable via UI slider |
| `MAX_ITER` | `8` | Caps agent tool-call iterations to prevent runaway loops |

---

## Disclaimer

Reports generated by FinSight are for informational and educational purposes only and do not constitute financial advice. The pipeline includes a fact-checking step, but LLM-generated financial data should always be independently verified before being used for real investment decisions.
