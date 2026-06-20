"""
app.py  —  Streamlit frontend for the Finance Analyst Agentic AI
────────────────────────────────────────────────────────────────
Design direction: AI equity-research terminal. Near-black surface,
mint/teal signature accent, monospace for data, Space Grotesk for
display type. Signature element: a live 4-node agent pipeline that
lights up as Research → Analyst → Validator → Writer each fire.

Functional architecture unchanged from prior version:
  • crew.kickoff() runs in a daemon thread (CrewRunner) so Streamlit stays responsive.
  • AgentLog objects flow through a queue.Queue → polled by st.rerun() loop.
  • st.session_state holds: runner, logs, result, running flag.
  • Secrets: reads from st.secrets first, falls back to os.environ / .env file.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from crew import AgentLog, CrewRunner

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinSight — AI Equity Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
# Color:  base #08090C · surface #13151A · border #1F2229
#         text #EDEEF0 · muted #7A8089 · accent #5EEAD4 (mint)
#         up #34D399 · down #FB7185
# Type:   display = Space Grotesk · body = Inter · data = JetBrains Mono

PIPELINE_STAGES = [
    ("Senior Financial Research Analyst", "research", "Research"),
    ("Principal Financial Analyst",       "analyst",  "Analysis"),
    ("Independent Research Validator",    "validator", "Validate"),
    ("Senior Financial Writer",           "writer",   "Report"),
]

MODELS = {
    "gemini-2.5-flash · fast, recommended": "gemini-2.5-flash",
    "gemini-2.5-pro · slower, sharper reasoning": "gemini-2.5-pro",
    "gemini-3-flash · latest frontier, preview": "gemini-3-flash",
}

MOCK_TICKER_ITEMS = [
    ("NVDA", "+2.4%", "up"), ("AAPL", "−0.6%", "down"), ("MSFT", "+1.1%", "up"),
    ("TSLA", "−3.2%", "down"), ("GOOGL", "+0.8%", "up"), ("AXISBANK", "+1.9%", "up"),
    ("AMZN", "+0.3%", "up"), ("META", "−1.4%", "down"), ("BTC", "+4.7%", "up"),
    ("HDFCBANK", "+0.5%", "up"), ("NFLX", "−0.2%", "down"), ("AMD", "+3.1%", "up"),
]


def _inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --base: #08090C;
        --surface: #13151A;
        --surface-raised: #181B22;
        --border: #1F2229;
        --text: #EDEEF0;
        --muted: #7A8089;
        --accent: #5EEAD4;
        --accent-dim: #2DD4BF;
        --up: #34D399;
        --down: #FB7185;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--base) !important;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] label { color: var(--muted) !important; font-size: 0.8rem; }

    /* ── Typography ── */
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
    .mono { font-family: 'JetBrains Mono', monospace; }

    /* ── Kill default Streamlit chrome we don't want ── */
    div[data-testid="stDecoration"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* ── Ticker marquee (signature element) ── */
    .ticker-wrap {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        white-space: nowrap;
        margin-bottom: 1.75rem;
        position: relative;
    }
    .ticker-wrap::before, .ticker-wrap::after {
        content: ""; position: absolute; top: 0; bottom: 0; width: 36px; z-index: 2;
    }
    .ticker-wrap::before { left: 0; background: linear-gradient(90deg, var(--surface), transparent); }
    .ticker-wrap::after  { right: 0; background: linear-gradient(270deg, var(--surface), transparent); }
    .ticker-track {
        display: inline-flex;
        animation: scroll-ticker 38s linear infinite;
        padding: 0.65rem 0;
    }
    @keyframes scroll-ticker {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .ticker-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        padding: 0 1.4rem;
        color: var(--muted);
        border-right: 1px solid var(--border);
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .ticker-item b { color: var(--text); font-weight: 500; }
    .tick-up   { color: var(--up); }
    .tick-down { color: var(--down); }

    @media (prefers-reduced-motion: reduce) {
        .ticker-track { animation: none; }
    }

    /* ── Hero ── */
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        line-height: 1.1;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 1.6rem;
    }

    /* ── Command bar (pill input row) ── */
    .cmdbar-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stTextInput"] input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
        padding: 0.65rem 0.9rem !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: var(--muted); opacity: 0.7; }

    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--accent) !important;
        color: #06211C !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        height: 100%;
        transition: filter 0.15s ease;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { filter: brightness(1.08); }
    div[data-testid="stButton"] button[kind="primary"]:disabled {
        background: var(--border) !important; color: var(--muted) !important;
    }

    /* ── Agent pipeline (signature visualization) ── */
    .pipeline-wrap {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
    }
    .pipeline-row { display: flex; align-items: center; justify-content: space-between; }
    .pipe-node { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; flex: 1; }
    .pipe-dot {
        width: 14px; height: 14px; border-radius: 50%;
        background: var(--border); border: 2px solid var(--border);
        transition: all 0.3s ease;
    }
    .pipe-dot.active {
        background: var(--accent); border-color: var(--accent);
        box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.18);
        animation: pulse-dot 1.4s ease-in-out infinite;
    }
    .pipe-dot.done { background: var(--up); border-color: var(--up); }
    @keyframes pulse-dot {
        0%, 100% { box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.18); }
        50%      { box-shadow: 0 0 0 8px rgba(94, 234, 212, 0.06); }
    }
    .pipe-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .pipe-label.active { color: var(--accent); }
    .pipe-label.done { color: var(--up); }
    .pipe-connector {
        flex: 1.4; height: 2px; background: var(--border);
        position: relative; top: -19px; max-width: 90px;
    }
    .pipe-connector.done { background: var(--up); }

    /* ── Live log console ── */
    .console {
        background: var(--base);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        max-height: 320px;
        overflow-y: auto;
    }
    .log-line {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: var(--up);
        padding: 0.2rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .log-line .ts { color: var(--muted); margin-right: 0.5rem; }
    .log-line .ag { color: var(--accent); }
    .log-line.err { color: var(--down); }

    /* ── Stat chips ── */
    .stat-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
    .stat-chip {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        flex: 1;
        min-width: 120px;
    }
    .stat-chip-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    .stat-chip-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text);
    }

    /* ── Recommendation badge ── */
    .rec-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.02em;
    }
    .rec-buy  { background: rgba(52, 211, 153, 0.12); color: var(--up); border: 1px solid rgba(52, 211, 153, 0.35); }
    .rec-hold { background: rgba(94, 234, 212, 0.10); color: var(--accent); border: 1px solid rgba(94, 234, 212, 0.3); }
    .rec-sell { background: rgba(251, 113, 133, 0.12); color: var(--down); border: 1px solid rgba(251, 113, 133, 0.35); }

    /* ── Report container ── */
    .report-shell {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 2.2rem 2.4rem;
    }
    .report-shell h2 {
        font-size: 1.25rem;
        color: var(--accent);
        margin-top: 1.6rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    .report-shell h2:first-child { margin-top: 0; }
    .report-shell h3 { font-size: 1.05rem; color: var(--text); margin-top: 1.2rem; }
    .report-shell p, .report-shell li { color: #D4D6DA; line-height: 1.7; font-size: 0.95rem; }
    .report-shell table {
        border-collapse: collapse; width: 100%; margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
    }
    .report-shell th {
        background: var(--surface-raised); color: var(--accent);
        padding: 0.55rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border);
    }
    .report-shell td { padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); color: var(--text); }
    .report-shell strong { color: var(--text); }
    .report-shell hr { border-color: var(--border); }

    /* ── Tabs ── */
    [data-testid="stTabs"] button { font-family: 'Space Grotesk', sans-serif; color: var(--muted); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent) !important; }
    [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

    /* ── Misc Streamlit overrides ── */
    [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
    div[data-testid="stDownloadButton"] button {
        background: var(--surface-raised) !important; color: var(--text) !important;
        border: 1px solid var(--border) !important; border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    div[data-testid="stDownloadButton"] button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)


def _render_ticker_marquee() -> None:
    items_html = "".join(
        f'<span class="ticker-item"><b>{sym}</b><span class="tick-{direction}">{chg}</span></span>'
        for sym, chg, direction in MOCK_TICKER_ITEMS
    )
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker-track">{items_html}{items_html}</div></div>',
        unsafe_allow_html=True,
    )


def _render_pipeline(active_agent_name: str | None, completed_stages: set[str]) -> None:
    nodes_html = ""
    for i, (full_name, key, label) in enumerate(PIPELINE_STAGES):
        is_done = key in completed_stages
        is_active = (active_agent_name or "").lower() in full_name.lower() and not is_done
        dot_cls = "done" if is_done else ("active" if is_active else "")
        label_cls = "done" if is_done else ("active" if is_active else "")

        nodes_html += f'<div class="pipe-node"><div class="pipe-dot {dot_cls}"></div><div class="pipe-label {label_cls}">{label}</div></div>'
        if i < len(PIPELINE_STAGES) - 1:
            conn_cls = "done" if is_done else ""
            nodes_html += f'<div class="pipe-connector {conn_cls}"></div>'

    st.markdown(f'<div class="pipeline-wrap"><div class="pipeline-row">{nodes_html}</div></div>', unsafe_allow_html=True)


def _completed_stages_from_logs(logs: list[AgentLog], current_agent: str | None) -> set[str]:
    """An agent stage counts as done once a *later* stage's agent appears in the logs."""
    seen_order: list[str] = []
    for log in logs:
        for full_name, key, _ in PIPELINE_STAGES:
            if full_name.lower() in log.agent.lower() and key not in seen_order:
                seen_order.append(key)
    if not seen_order:
        return set()
    return set(seen_order[:-1]) if len(seen_order) > 1 else set()


def _current_agent_from_logs(logs: list[AgentLog]) -> str | None:
    for log in reversed(logs):
        for full_name, _, _ in PIPELINE_STAGES:
            if full_name.lower() in log.agent.lower():
                return full_name
    return None


def _inject_secrets() -> None:
    for key in ("GOOGLE_API_KEY", "SERPER_API_KEY"):
        if key not in os.environ:
            val = st.secrets.get(key, "")
            if val:
                os.environ[key] = val


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.1rem;'
            'font-weight:700;margin-bottom:0.2rem;">◈ FinSight</div>'
            '<div style="color:var(--muted);font-size:0.78rem;margin-bottom:1.4rem;">Configuration</div>',
            unsafe_allow_html=True,
        )

        with st.expander("API keys", expanded=not bool(os.getenv("GOOGLE_API_KEY"))):
            google_key = st.text_input(
                "Google Gemini API key", value=os.getenv("GOOGLE_API_KEY", ""),
                type="password", placeholder="AIza...",
                help="Get yours at https://aistudio.google.com/",
            )
            serper_key = st.text_input(
                "Serper.dev API key", value=os.getenv("SERPER_API_KEY", ""),
                type="password", placeholder="abc123...",
                help="Get yours at https://serper.dev/",
            )
            if google_key:
                os.environ["GOOGLE_API_KEY"] = google_key
            if serper_key:
                os.environ["SERPER_API_KEY"] = serper_key

        model_label = st.selectbox("Model", options=list(MODELS.keys()), index=0)
        model_id = MODELS[model_label]
        os.environ["GEMINI_MODEL"] = model_id

        with st.expander("Advanced"):
            max_iter_val = st.slider("Max agent iterations", 3, 15, 8)
            os.environ["MAX_ITER"] = str(max_iter_val)
            temp_val = st.slider("LLM temperature", 0.0, 1.0, 0.2, step=0.05)
            os.environ["LLM_TEMPERATURE"] = str(temp_val)

        st.markdown(
            '<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid var(--border);'
            'color:var(--muted);font-size:0.72rem;line-height:1.6;">'
            'FinSight runs four agents in sequence — Research, Analysis, Validation, Writing — '
            'built on CrewAI and Gemini.<br><br>Not financial advice.</div>',
            unsafe_allow_html=True,
        )

    return model_id


def _init_state() -> None:
    defaults = {"runner": None, "logs": [], "result": None, "running": False, "last_ticker": ""}
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _recommendation_badge(report_md: str) -> str:
    """Best-effort extraction of BUY/HOLD/SELL for a badge near the top."""
    upper = report_md.upper()
    if "RECOMMENDATION: BUY" in upper or "**BUY**" in upper:
        return '<span class="rec-badge rec-buy">BUY</span>'
    if "RECOMMENDATION: SELL" in upper or "**SELL**" in upper:
        return '<span class="rec-badge rec-sell">SELL</span>'
    if "RECOMMENDATION: HOLD" in upper or "**HOLD**" in upper:
        return '<span class="rec-badge rec-hold">HOLD</span>'
    return ""


def main() -> None:
    _init_state()
    _inject_secrets()
    _inject_css()
    model_id = _sidebar()

    _render_ticker_marquee()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-eyebrow">◈ MULTI-AGENT EQUITY RESEARCH</div>'
        '<div class="hero-title">FinSight</div>'
        '<div class="hero-sub">Four AI analysts — research, analysis, validation, writing — '
        'working a single ticker until it becomes a report.</div>',
        unsafe_allow_html=True,
    )

    # ── Command bar ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 3, 1.1])
    with col1:
        st.markdown('<div class="cmdbar-label">Ticker / Company</div>', unsafe_allow_html=True)
        ticker = st.text_input(
            "ticker", placeholder="AAPL, Tesla, Axis Bank…",
            disabled=st.session_state.running, label_visibility="collapsed",
        ).strip().upper()
    with col2:
        st.markdown('<div class="cmdbar-label">Focus question (optional)</div>', unsafe_allow_html=True)
        extra_query = st.text_input(
            "focus", placeholder="e.g. how exposed is this to AI chip export controls?",
            disabled=st.session_state.running, label_visibility="collapsed",
        )
    with col3:
        st.markdown('<div class="cmdbar-label">&nbsp;</div>', unsafe_allow_html=True)
        run_btn = st.button(
            "Run analysis" if not st.session_state.running else "Running…",
            type="primary", disabled=st.session_state.running or not ticker,
            use_container_width=True,
        )

    if run_btn:
        missing = [k for k in ("GOOGLE_API_KEY", "SERPER_API_KEY") if not os.getenv(k)]
        if missing:
            st.error(f"Add your API keys in the sidebar first: {', '.join(missing)}")
        else:
            st.session_state.runner = CrewRunner(ticker, extra_query, model_id)
            st.session_state.logs = []
            st.session_state.result = None
            st.session_state.running = True
            st.session_state.last_ticker = ticker
            st.session_state.runner.start()
            st.rerun()

    # ── Live run view ─────────────────────────────────────────────────────────
    if st.session_state.running and st.session_state.runner:
        runner: CrewRunner = st.session_state.runner
        st.session_state.logs.extend(runner.drain_logs())

        current_agent = _current_agent_from_logs(st.session_state.logs)
        completed = _completed_stages_from_logs(st.session_state.logs, current_agent)
        _render_pipeline(current_agent, completed)

        st.markdown(
            f'<div class="cmdbar-label">Live log · {st.session_state.last_ticker}</div>',
            unsafe_allow_html=True,
        )
        lines_html = "".join(
            f'<div class="log-line{" err" if "error" in l.action.lower() else ""}">'
            f'<span class="ts">{l.timestamp}</span><span class="ag">{l.agent[:28]}</span> › {l.detail[:180]}</div>'
            for l in st.session_state.logs[-25:]
        )
        empty_state = '<i style="color:var(--muted)">warming up…</i>'
        st.markdown(f'<div class="console">{lines_html or empty_state}</div>', unsafe_allow_html=True)

        if runner.done:
            st.session_state.result = runner.result
            st.session_state.running = False
            st.rerun()
        else:
            time.sleep(0.8)
            st.rerun()

    # ── Final result ──────────────────────────────────────────────────────────
    if st.session_state.result is not None:
        result = st.session_state.result

        if not result.success:
            st.error(f"Analysis failed: {result.error}")
            st.info("Check your API keys and try again. If it persists, try gemini-2.5-flash.")
            return

        badge = _recommendation_badge(result.report_md)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.8rem;margin:1rem 0;">'
            f'<span style="color:var(--up);font-family:\'Space Grotesk\',sans-serif;font-weight:600;">'
            f'✓ Report ready for {st.session_state.last_ticker}</span>{badge}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'''<div class="stat-row">
                <div class="stat-chip"><div class="stat-chip-label">Ticker</div><div class="stat-chip-value mono">{st.session_state.last_ticker}</div></div>
                <div class="stat-chip"><div class="stat-chip-label">Model</div><div class="stat-chip-value mono">{os.getenv("GEMINI_MODEL","—")}</div></div>
                <div class="stat-chip"><div class="stat-chip-label">Agent steps</div><div class="stat-chip-value mono">{len(st.session_state.logs)}</div></div>
                <div class="stat-chip"><div class="stat-chip-label">Generated</div><div class="stat-chip-value mono">{datetime.now().strftime("%H:%M")}</div></div>
            </div>''',
            unsafe_allow_html=True,
        )

        tab_report, tab_logs = st.tabs(["Report", "Agent logs"])

        with tab_report:
            st.markdown('<div class="report-shell">', unsafe_allow_html=True)
            st.markdown(result.report_md)
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download report (.md)", data=result.report_md,
                    file_name=f"{st.session_state.last_ticker}_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown", use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "Download logs (.txt)",
                    data="\n".join(f"[{l.timestamp}] {l.agent} › {l.detail}" for l in st.session_state.logs),
                    file_name=f"{st.session_state.last_ticker}_logs_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain", use_container_width=True,
                )

        with tab_logs:
            lines_html = "".join(
                f'<div class="log-line{" err" if "error" in l.action.lower() else ""}">'
                f'<span class="ts">{l.timestamp}</span><span class="ag">{l.agent[:28]}</span> › {l.detail[:300]}</div>'
                for l in st.session_state.logs
            )
            st.markdown(f'<div class="console" style="max-height:600px;">{lines_html or "<i>No logs captured.</i>"}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()