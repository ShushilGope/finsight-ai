"""
tasks/finance_tasks.py
──────────────────────
One CrewAI Task per agent, wired together via `context` so outputs flow
downstream: Research → Analyst → Validator → Writer.

Each task has:
  • description   – The "system prompt" for the task; be explicit about format.
  • expected_output – Tells CrewAI what success looks like (used for quality gating).
  • agent          – Assigned agent instance.
  • context        – List of upstream tasks whose outputs are injected as context.
"""

from __future__ import annotations

from crewai import Task

from finance_agents import (
    create_analyst_agent,
    create_research_agent,
    create_validator_agent,
    create_writer_agent,
)
from schemas.analysis_schema import StructuredAnalysis


def build_tasks(ticker: str, extra_query: str = "", model: str | None = None) -> list[Task]:
    """
    Build and return the four tasks for a given ticker/query.

    Args:
        ticker:      Stock ticker or company name (e.g. "AAPL", "Tesla").
        extra_query: Optional user question to layer on top (e.g. "focus on AI division").
        model:       Gemini model string; passed down to all agents.

    Returns:
        Ordered list of Tasks ready to hand to the Crew.
    """

    # ── Instantiate agents ────────────────────────────────────────────────────
    researcher = create_research_agent(model)
    analyst    = create_analyst_agent(model)
    validator  = create_validator_agent(model)
    writer     = create_writer_agent(model)

    focus_note = f"\n\nAdditional user focus: {extra_query}" if extra_query.strip() else ""

    # ── Task 1 — Research ─────────────────────────────────────────────────────
    research_task = Task(
        description=(
            f"Conduct comprehensive research on **{ticker}**.\n\n"
            "Collect and clearly present:\n"
            "1. Current stock price, 52-week high/low, market capitalisation.\n"
            "2. Last two quarterly earnings: EPS (actual vs estimate), revenue (actual vs estimate).\n"
            "3. Key financial ratios: P/E, forward P/E, P/S, EV/EBITDA, debt-to-equity, current ratio.\n"
            "4. Recent significant news (last 30 days): earnings calls, M&A, regulatory, macro.\n"
            "5. Analyst consensus: average target price, Buy/Hold/Sell breakdown.\n"
            "6. Top 2–3 direct competitors and their P/E for benchmarking.\n\n"
            "For EVERY data point include: source URL + date retrieved.\n"
            "Flag any data that could not be confirmed with a ⚠️ symbol."
            + focus_note
        ),
        expected_output=(
            "A structured Markdown document with labelled sections for each of the 6 areas above. "
            "Every figure must have an inline citation (URL + date). "
            "No financial figures without a source."
        ),
        agent=researcher,
    )

    # ── Task 2 — Analysis ─────────────────────────────────────────────────────
    analysis_task = Task(
        description=(
            f"Analyse **{ticker}** using the research data provided in context.\n\n"
            "Produce:\n"
            "1. **Valuation assessment** — Are current multiples stretched, fair, or cheap "
            "   vs 5-year average and sector peers? Quantify.\n"
            "2. **Financial health** — Trend in FCF, margin trajectory, balance sheet risk.\n"
            "3. **Bull case** — 3 key catalysts with realistic upside price target.\n"
            "4. **Bear case** — 3 key risks with downside price target.\n"
            "5. **Base case** — Most probable 12-month price target with assumptions.\n"
            "6. **Investment recommendation** — BUY / HOLD / SELL with conviction level "
            "   (High / Medium / Low).\n\n"
            "Use numbers. Avoid vague language ('could potentially'). "
            "Every claim must reference a figure from the research context.\n\n"
            "IMPORTANT — structured fields: in addition to your written analysis, you must "
            "populate the structured output fields exactly: ticker, current_price, "
            "exactly 3 scenarios in order [Bull, Base, Bear] each with price_target and "
            "pct_change (signed, e.g. -18.7 for bear), valuation_ratios (metric name + "
            "company_value + optional peer_avg), recommendation, and conviction. "
            "These numbers must match what you state in your written analysis exactly."
            + focus_note
        ),
        expected_output=(
            "A structured analytical memo covering all 6 points, AND the structured "
            "data fields populated to match (3 price targets bull/base/bear with "
            "supporting assumptions, valuation ratios, recommendation, conviction)."
        ),
        agent=analyst,
        context=[research_task],    # ← research output injected here
        output_pydantic=StructuredAnalysis,
    )

    # ── Task 3 — Validation ───────────────────────────────────────────────────
    validation_task = Task(
        description=(
            f"Validate the financial analysis of **{ticker}**.\n\n"
            "Steps:\n"
            "1. Cross-check every numerical figure in the analysis against the research data.\n"
            "2. Verify that each price target has supporting assumptions.\n"
            "3. Identify any logical inconsistencies (e.g. bull case lower than current price).\n"
            "4. Flag any claim not traceable to a cited source with a ⚠️ marker.\n"
            "5. Output a **Validation Summary**:\n"
            "   - Status: PASS | PASS WITH CAVEATS | FAIL\n"
            "   - List of confirmed figures\n"
            "   - **Flagged Figures**: a clearly labelled bullet list of every specific figure "
            "     that could not be verified, each starting with the exact figure and its claimed "
            "     value (e.g. '⚠️ Current share price ₹1,352.40 — source did not confirm this exact "
            "     number'). This list must be easy for a downstream writer to lift verbatim.\n\n"
            "Note: Status FAIL does not mean the process stops — it means the report must "
            "visibly mark every flagged figure rather than presenting it as confirmed fact."
        ),
        expected_output=(
            "A Validation Summary with: overall status, list of confirmed data points, "
            "and a distinct 'Flagged Figures' bullet list of unverified figures (empty list "
            "explicitly stated if none). Concise — max 400 words."
        ),
        agent=validator,
        context=[research_task, analysis_task],   # ← both upstream tasks as context
    )

    # ── Task 4 — Write report ─────────────────────────────────────────────────
    report_task = Task(
        description=(
            f"Write a professional equity research report for **{ticker}** "
            "based on the validated analysis.\n\n"
            "**Required sections (use these exact Markdown headings):**\n\n"
            "## Executive Summary\n"
            "## Company Overview\n"
            "## Financial Performance\n"
            "## Valuation Analysis\n"
            "## Bull / Base / Bear Scenarios\n"
            "## Key Risks\n"
            "## Investment Recommendation\n"
            "## Disclaimer\n\n"
            "**Formatting rules:**\n"
            "- Use Markdown tables for financial comparisons.\n"
            "- Use bullet lists for risks and catalysts.\n"
            "- Bold the final recommendation on its own line.\n"
            "- The Disclaimer section must state: data is for informational purposes only, "
            "  not financial advice, and sources may contain errors.\n"
            "- Target length: 800–1 200 words.\n\n"
            "**Mandatory — surfacing unverified data:**\n"
            "If the Validator's output lists ANY 'Flagged Figures', you MUST:\n"
            "1. Open the Executive Summary with a callout line starting exactly with "
            "   '⚠️ **Data Notice:**' summarising how many figures could not be independently "
            "   verified and why the reader should treat them with caution.\n"
            "2. Every time one of those specific flagged figures appears anywhere in the report "
            "   body, prefix it inline with ⚠️ (e.g. '⚠️ ₹1,352.40').\n"
            "3. Never silently present a flagged figure as plain confirmed fact.\n"
            "If the Validator reported zero flagged figures, state this is the case in one short "
            "sentence near the top instead.\n\n"
            "Do NOT invent any figures. Only use data from the validated analysis context."
            + focus_note
        ),
        expected_output=(
            "A complete, well-structured Markdown equity research report with all 8 required "
            "sections, at least one Markdown table, a clearly stated investment recommendation, "
            "and — if applicable — a visible Data Notice plus inline ⚠️ markers on every figure "
            "the Validator could not confirm."
        ),
        agent=writer,
        context=[research_task, analysis_task, validation_task],
        output_file="output/report.md",   # CrewAI writes this automatically
    )

    return [research_task, analysis_task, validation_task, report_task]
