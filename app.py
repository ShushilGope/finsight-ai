"""
app.py  —  Streamlit frontend for the Finance Analyst Agentic AI
────────────────────────────────────────────────────────────────
Design direction: warm paper-and-gold research desk. Soft cream base,
sage green for bullish/confirmed signals, dusty rose for bearish/risk,
warm gold accent for "active" states. Signature element: a live SVG
system-architecture diagram showing the 4-agent pipeline with the
currently-executing node glowing, paired with a detailed log feed below.

Functional architecture (unchanged threading model):
  • crew.kickoff() runs in a daemon thread (CrewRunner) so Streamlit stays responsive.
  • AgentLog objects flow through a queue.Queue → polled by st.rerun() loop.
  • st.session_state holds: runner, logs, result, running flag.
  • Secrets: reads from st.secrets first, falls back to os.environ / .env file.

New in this version:
  • Live SVG architecture diagram (boxes + arrows), active node glows gold.
  • Plotly bar charts for Bull/Base/Bear scenarios + valuation ratio comparison,
    sourced from result.structured_analysis (CrewAI output_pydantic).
  • Data Notice banner — surfaces when the Writer flags unverified figures.
  • Full error + traceback display on failure (previously silently truncated).
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import plotly.graph_objects as go
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
# Color:  base #FAF8F5 (warm paper) · surface #FFFFFF · border #E8E2D6
#         text #2B2823 (warm near-black) · muted #8A8478
#         gold #C9A876 (accent / active) · sage #7FA88F (up/confirmed)
#         rose #C97D7D (down/risk/error)
# Type:   display = Space Grotesk · body = Inter · data = JetBrains Mono

PIPELINE_STAGES = [
    ("Senior Financial Research Analyst", "research", "Research", "Gathers live data"),
    ("Principal Financial Analyst",       "analyst",  "Analysis", "Valuation & scenarios"),
    ("Independent Research Validator",    "validator", "Validate", "Fact-checks figures"),
    ("Senior Financial Writer",           "writer",   "Report",   "Drafts final report"),
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
        --base: #FAF8F5;
        --surface: #FFFFFF;
        --surface-warm: #F4EFE6;
        --border: #E8E2D6;
        --text: #2B2823;
        --muted: #8A8478;
        --gold: #C9A876;
        --gold-soft: #F0E6D2;
        --sage: #7FA88F;
        --sage-soft: #E3EDE7;
        --rose: #C97D7D;
        --rose-soft: #F5E4E2;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--base) !important;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }

    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] label { color: var(--muted) !important; font-size: 0.8rem; }

    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; color: var(--text); }
    .mono { font-family: 'JetBrains Mono', monospace; }

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
        margin-bottom: 1.6rem;
        position: relative;
    }
    .ticker-wrap::before, .ticker-wrap::after {
        content: ""; position: absolute; top: 0; bottom: 0; width: 36px; z-index: 2;
    }
    .ticker-wrap::before { left: 0; background: linear-gradient(90deg, var(--surface), transparent); }
    .ticker-wrap::after  { right: 0; background: linear-gradient(270deg, var(--surface), transparent); }
    .ticker-track { display: inline-flex; animation: scroll-ticker 38s linear infinite; padding: 0.6rem 0; }
    @keyframes scroll-ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .ticker-item {
        font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; padding: 0 1.3rem;
        color: var(--muted); border-right: 1px solid var(--border);
        display: inline-flex; align-items: center; gap: 0.45rem;
    }
    .ticker-item b { color: var(--text); font-weight: 500; }
    .tick-up   { color: #4F7A63; }
    .tick-down { color: #A14F4F; }
    @media (prefers-reduced-motion: reduce) { .ticker-track { animation: none; } }

    /* ── Hero ── */
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--gold);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .hero-title { font-size: 2.4rem; font-weight: 700; margin: 0 0 0.3rem 0; line-height: 1.1; color: var(--text); }
    .hero-sub { color: var(--muted); font-size: 0.95rem; margin-bottom: 1.6rem; max-width: 640px; }

    /* ── Command bar ── */
    .cmdbar-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem; font-weight: 600;
    }
    div[data-testid="stTextInput"] input {
        background: var(--surface) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
        padding: 0.65rem 0.9rem !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 3px var(--gold-soft) !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: var(--muted); opacity: 0.8; }

    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--text) !important;
        color: var(--base) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        height: 100%;
        transition: all 0.15s ease;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover { background: var(--gold) !important; color: var(--text) !important; }
    div[data-testid="stButton"] button[kind="primary"]:disabled { background: var(--border) !important; color: var(--muted) !important; }

    /* ── Cards ── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.5rem 1.7rem;
        margin: 1rem 0;
    }
    .card-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600;
        color: var(--text); margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .card-title .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--gold); display: inline-block; }

    /* ── Console (log feed) ── */
    .console {
        background: var(--surface-warm);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        max-height: 320px;
        overflow-y: auto;
    }
    .log-line {
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        color: var(--text); padding: 0.32rem 0; border-bottom: 1px solid var(--border);
    }
    .log-line:last-child { border-bottom: none; }
    .log-line .ts { color: var(--muted); margin-right: 0.5rem; }
    .log-line .ag { color: var(--gold); font-weight: 600; }
    .log-line.err { color: var(--rose); }

    /* ── Stat chips ── */
    .stat-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
    .stat-chip { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.7rem 1rem; flex: 1; min-width: 120px; }
    .stat-chip-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.25rem; }
    .stat-chip-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: var(--text); }

    /* ── Recommendation badge ── */
    .rec-badge { display: inline-block; padding: 0.4rem 1.1rem; border-radius: 8px; font-weight: 700; font-size: 1rem; font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.02em; }
    .rec-buy  { background: var(--sage-soft); color: #4F7A63; border: 1px solid var(--sage); }
    .rec-hold { background: var(--gold-soft); color: #8A6D2F; border: 1px solid var(--gold); }
    .rec-sell { background: var(--rose-soft); color: #A14F4F; border: 1px solid var(--rose); }

    /* ── Data notice banner ── */
    .data-notice {
        background: var(--gold-soft);
        border: 1px solid var(--gold);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin: 1rem 0;
        color: #6B5524;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .data-notice b { color: #4A3A18; }

    /* ── Error panel ── */
    .error-panel { background: var(--rose-soft); border: 1px solid var(--rose); border-radius: 12px; padding: 1.2rem 1.4rem; margin: 1rem 0; }
    .error-panel summary { cursor: pointer; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #A14F4F; margin-top: 0.6rem; }
    .error-trace { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; white-space: pre-wrap; color: #7A3535; background: #FBEFEE; border-radius: 8px; padding: 0.8rem; margin-top: 0.5rem; max-height: 280px; overflow-y: auto; }

    /* ── Report container ── */
    .report-shell { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 2.2rem 2.4rem; }
    .report-shell h2 { font-size: 1.25rem; color: var(--text); margin-top: 1.6rem; padding-bottom: 0.5rem; border-bottom: 2px solid var(--gold-soft); }
    .report-shell h2:first-child { margin-top: 0; }
    .report-shell h3 { font-size: 1.05rem; color: var(--text); margin-top: 1.2rem; }
    .report-shell p, .report-shell li { color: #4A4640; line-height: 1.75; font-size: 0.95rem; }
    .report-shell table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
    .report-shell th { background: var(--surface-warm); color: var(--text); padding: 0.55rem 0.8rem; text-align: left; border-bottom: 2px solid var(--gold); }
    .report-shell td { padding: 0.5rem 0.8rem; border-bottom: 1px solid var(--border); color: var(--text); }
    .report-shell strong { color: var(--text); }
    .report-shell hr { border-color: var(--border); }

    [data-testid="stTabs"] button { font-family: 'Space Grotesk', sans-serif; color: var(--muted); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--text) !important; border-bottom-color: var(--gold) !important; }
    [data-baseweb="tab-highlight"] { background-color: var(--gold) !important; }

    [data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
    div[data-testid="stDownloadButton"] button {
        background: var(--surface) !important; color: var(--text) !important;
        border: 1.5px solid var(--border) !important; border-radius: 10px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    div[data-testid="stDownloadButton"] button:hover { border-color: var(--gold) !important; background: var(--gold-soft) !important; }
    .stAlert { border-radius: 10px; }

    /* ── Architecture SVG animation (defined once, not per-render) ── */
    .arch-pulse { animation: archPulse 1.6s ease-in-out infinite; }
    @keyframes archPulse {
        0%, 100% { filter: drop-shadow(0 0 0px #C9A876); }
        50% { filter: drop-shadow(0 0 10px #C9A876); }
    }
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


# ── SVG architecture diagram (signature element) ───────────────────────────────

def _render_architecture_svg(active_agent_name: str | None, completed: set[str], timings: dict[str, float]) -> str:
    """
    4-box pipeline diagram with connecting arrows. Active node glows gold and
    pulses; completed nodes turn sage with a checkmark; pending nodes stay
    neutral. Shows elapsed seconds per agent once available.
    """
    box_w, box_h, gap, y = 230, 92, 38, 40
    total_w = box_w * 4 + gap * 3 + 40
    svg_parts = [f'<svg viewBox="0 0 {total_w} 190" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:Inter,sans-serif;">']

    for i, (full_name, key, label, sub) in enumerate(PIPELINE_STAGES):
        x = 20 + i * (box_w + gap)
        is_done = key in completed
        is_active = (active_agent_name or "").lower() in full_name.lower() and not is_done

        if is_done:
            fill, stroke, text_color = "#E3EDE7", "#7FA88F", "#4F7A63"
        elif is_active:
            fill, stroke, text_color = "#F0E6D2", "#C9A876", "#8A6D2F"
        else:
            fill, stroke, text_color = "#FFFFFF", "#E8E2D6", "#8A8478"

        node_cls = "arch-pulse" if is_active else ""
        elapsed = timings.get(full_name)
        time_str = f"{elapsed:.0f}s" if elapsed is not None else ""
        status_icon = "✓" if is_done else ("●" if is_active else "○")

        svg_parts.append(f'''
            <g class="{node_cls}">
                <rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="14"
                      fill="{fill}" stroke="{stroke}" stroke-width="2"/>
                <text x="{x+18}" y="{y+30}" font-size="13" font-weight="700" fill="{text_color}" font-family="Space Grotesk, sans-serif">{status_icon}  {label}</text>
                <text x="{x+18}" y="{y+52}" font-size="11" fill="{text_color}" opacity="0.85">{sub}</text>
                <text x="{x+18}" y="{y+74}" font-size="10" fill="{text_color}" opacity="0.6" font-family="JetBrains Mono, monospace">{time_str}</text>
            </g>
        ''')

        if i < len(PIPELINE_STAGES) - 1:
            arrow_x1 = x + box_w
            arrow_x2 = arrow_x1 + gap
            arrow_color = "#7FA88F" if is_done else "#E8E2D6"
            mid_y = y + box_h / 2
            svg_parts.append(f'''
                <line x1="{arrow_x1+4}" y1="{mid_y}" x2="{arrow_x2-8}" y2="{mid_y}" stroke="{arrow_color}" stroke-width="2.5"/>
                <polygon points="{arrow_x2-8},{mid_y-5} {arrow_x2-8},{mid_y+5} {arrow_x2-2},{mid_y}" fill="{arrow_color}"/>
            ''')

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def _completed_stages_from_logs(logs: list[AgentLog]) -> set[str]:
    """
    Reliable completion tracking: only `action == "task_complete"` logs count
    (emitted by crew.py's task_callback, which has a stable .agent string).
    Fuzzy step-level logs are ignored here on purpose.
    """
    completed: set[str] = set()
    for log in logs:
        if log.action != "task_complete":
            continue
        for full_name, key, _, _ in PIPELINE_STAGES:
            if full_name.lower() in log.agent.lower():
                completed.add(key)
    return completed


def _current_agent_from_logs(logs: list[AgentLog]) -> str | None:
    """
    The active agent is the first stage in the fixed pipeline order that
    hasn't completed yet. This sidesteps unreliable step-object parsing
    entirely — we know the sequence (Research → Analyst → Validator → Writer)
    in advance, so "next incomplete stage" is always correct once at least
    one log has arrived (meaning the run has actually started).
    """
    if not logs:
        return None
    completed = _completed_stages_from_logs(logs)
    for full_name, key, _, _ in PIPELINE_STAGES:
        if key not in completed:
            return full_name
    return None  # all stages done


# ── Charts from structured analysis ────────────────────────────────────────────

def _scenario_chart(structured) -> go.Figure:
    labels = [s.label for s in structured.scenarios]
    prices = [s.price_target for s in structured.scenarios]
    pcts   = [s.pct_change for s in structured.scenarios]
    colors = ["#7FA88F" if p >= 0 else "#C97D7D" for p in pcts]

    fig = go.Figure(go.Bar(
        x=labels, y=prices, marker_color=colors,
        text=[f"{p:+.1f}%" for p in pcts], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=structured.current_price, line_dash="dot", line_color="#8A8478",
                  annotation_text=f"Current: {structured.current_price:.2f}", annotation_position="top left")
    fig.update_layout(
        title="Price targets by scenario", height=320, margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#2B2823"),
        yaxis=dict(gridcolor="#E8E2D6"), xaxis=dict(gridcolor="#E8E2D6"),
    )
    return fig


def _ratio_chart(structured) -> go.Figure | None:
    ratios = [r for r in structured.valuation_ratios if r.peer_avg is not None]
    if not ratios:
        return None
    metrics = [r.metric for r in ratios]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=structured.ticker, x=metrics, y=[r.company_value for r in ratios], marker_color="#C9A876"))
    fig.add_trace(go.Bar(name="Peer avg", x=metrics, y=[r.peer_avg for r in ratios], marker_color="#E8E2D6"))
    fig.update_layout(
        title="Valuation vs peers", height=320, barmode="group", margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#2B2823"),
        yaxis=dict(gridcolor="#E8E2D6"), xaxis=dict(gridcolor="#E8E2D6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Sidebar / state / helpers ───────────────────────────────────────────────────

def _inject_secrets() -> None:
    for key in ("GOOGLE_API_KEY", "SERPER_API_KEY"):
        if key not in os.environ:
            val = st.secrets.get(key, "")
            if val:
                os.environ[key] = val


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.2rem;">◈ FinSight</div>'
            '<div style="color:var(--muted);font-size:0.78rem;margin-bottom:1.4rem;">Configuration</div>',
            unsafe_allow_html=True,
        )

        with st.expander("API keys", expanded=not bool(os.getenv("GOOGLE_API_KEY"))):
            google_key = st.text_input("Google Gemini API key", value=os.getenv("GOOGLE_API_KEY", ""), type="password",
                                        placeholder="AIza...", help="Get yours at https://aistudio.google.com/")
            serper_key = st.text_input("Serper.dev API key", value=os.getenv("SERPER_API_KEY", ""), type="password",
                                        placeholder="abc123...", help="Get yours at https://serper.dev/")
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
    upper = report_md.upper()
    if "RECOMMENDATION: BUY" in upper or "**BUY**" in upper:
        return '<span class="rec-badge rec-buy">BUY</span>'
    if "RECOMMENDATION: SELL" in upper or "**SELL**" in upper:
        return '<span class="rec-badge rec-sell">SELL</span>'
    if "RECOMMENDATION: HOLD" in upper or "**HOLD**" in upper:
        return '<span class="rec-badge rec-hold">HOLD</span>'
    return ""


def _data_notice_banner(report_md: str) -> str | None:
    """Extract the Data Notice line the Writer is instructed to emit, if present."""
    marker = "⚠️ **Data Notice:**"
    if marker not in report_md:
        return None
    start = report_md.index(marker)
    end = report_md.find("\n\n", start)
    snippet = report_md[start:end if end != -1 else start + 400]
    return snippet


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
        ticker = st.text_input("ticker", placeholder="AAPL, Tesla, Axis Bank…",
                                disabled=st.session_state.running, label_visibility="collapsed").strip().upper()
    with col2:
        st.markdown('<div class="cmdbar-label">Focus question (optional)</div>', unsafe_allow_html=True)
        extra_query = st.text_input("focus", placeholder="e.g. how exposed is this to AI chip export controls?",
                                     disabled=st.session_state.running, label_visibility="collapsed")
    with col3:
        st.markdown('<div class="cmdbar-label">&nbsp;</div>', unsafe_allow_html=True)
        run_btn = st.button("Run analysis" if not st.session_state.running else "Running…",
                             type="primary", disabled=st.session_state.running or not ticker, use_container_width=True)

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
        completed = _completed_stages_from_logs(st.session_state.logs)
        timings = runner.live_timings

        st.markdown('<div class="card"><div class="card-title"><span class="dot"></span>System architecture — live</div>', unsafe_allow_html=True)
        st.markdown(_render_architecture_svg(current_agent, completed, timings), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="cmdbar-label">Detail feed · {st.session_state.last_ticker}</div>', unsafe_allow_html=True)
        empty_state = '<i style="color:var(--muted)">warming up…</i>'
        lines_html = "".join(
            f'<div class="log-line{" err" if "error" in l.action.lower() else ""}">'
            f'<span class="ts">{l.timestamp}</span><span class="ag">{l.agent[:28]}</span> › {l.detail[:220]}</div>'
            for l in st.session_state.logs[-25:]
        )
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
            st.markdown(
                f'<div class="error-panel">'
                f'<b style="color:#A14F4F;font-family:\'Space Grotesk\',sans-serif;">Analysis failed</b><br>'
                f'<span style="color:#7A3535;">{result.error}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if result.error_trace:
                with st.expander("Show full traceback"):
                    st.code(result.error_trace, language="text")
            st.info("Check your API keys and try again. If it persists, try gemini-2.5-flash.")
            return

        # Completed architecture snapshot (all nodes done)
        st.markdown('<div class="card"><div class="card-title"><span class="dot" style="background:#7FA88F;"></span>System architecture — complete</div>', unsafe_allow_html=True)
        all_keys = {k for _, k, _, _ in PIPELINE_STAGES}
        st.markdown(_render_architecture_svg(None, all_keys, result.agent_timings), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        badge = _recommendation_badge(result.report_md)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.8rem;margin:1rem 0;">'
            f'<span style="color:#4F7A63;font-family:\'Space Grotesk\',sans-serif;font-weight:600;">'
            f'✓ Report ready for {st.session_state.last_ticker}</span>{badge}</div>',
            unsafe_allow_html=True,
        )

        notice = _data_notice_banner(result.report_md)
        if notice:
            st.markdown(f'<div class="data-notice">{notice}</div>', unsafe_allow_html=True)

        st.markdown(
            f'''<div class="stat-row">
                <div class="stat-chip"><div class="stat-chip-label">Ticker</div><div class="stat-chip-value mono">{st.session_state.last_ticker}</div></div>
                <div class="stat-chip"><div class="stat-chip-label">Model</div><div class="stat-chip-value mono">{os.getenv("GEMINI_MODEL","—")}</div></div>
                <div class="stat-chip"><div class="stat-chip-label">Agent steps</div><div class="stat-chip-value mono">{len(st.session_state.logs)}</div></div>
                <div class="stat-chip"><div class="stat-chip-label">Generated</div><div class="stat-chip-value mono">{datetime.now().strftime("%H:%M")}</div></div>
            </div>''',
            unsafe_allow_html=True,
        )

        tab_report, tab_charts, tab_logs = st.tabs(["Report", "Charts", "Agent logs"])

        with tab_report:
            st.markdown('<div class="report-shell">', unsafe_allow_html=True)
            st.markdown(result.report_md)
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("Download report (.md)", data=result.report_md,
                                    file_name=f"{st.session_state.last_ticker}_report_{datetime.now().strftime('%Y%m%d')}.md",
                                    mime="text/markdown", use_container_width=True)
            with dl2:
                st.download_button("Download logs (.txt)",
                                    data="\n".join(f"[{l.timestamp}] {l.agent} › {l.detail}" for l in st.session_state.logs),
                                    file_name=f"{st.session_state.last_ticker}_logs_{datetime.now().strftime('%Y%m%d')}.txt",
                                    mime="text/plain", use_container_width=True)

        with tab_charts:
            if result.structured_analysis is None:
                st.markdown(
                    '<div class="card" style="text-align:center;color:var(--muted);padding:2.5rem;">'
                    "The analyst's structured data didn't parse cleanly this run, so charts aren't "
                    "available — the written report above is unaffected.</div>",
                    unsafe_allow_html=True,
                )
            else:
                sa = result.structured_analysis
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(_scenario_chart(sa), use_container_width=True, key="scenario_chart")
                with c2:
                    ratio_fig = _ratio_chart(sa)
                    if ratio_fig:
                        st.plotly_chart(ratio_fig, use_container_width=True, key="ratio_chart")
                    else:
                        st.markdown(
                            '<div class="card" style="text-align:center;color:var(--muted);padding:2.5rem;">'
                            "No peer comparison data available for this run.</div>",
                            unsafe_allow_html=True,
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
