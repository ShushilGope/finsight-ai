"""
agents/finance_agents.py
────────────────────────
Four CrewAI agents for the finance analyst pipeline.

Improvement over base architecture:
  Research → Analyst → *** Validator *** → Writer
                            ↑ new agent added to fact-check before report is written

Each agent is built by a factory function so callers can pass a custom LLM
(e.g. the Streamlit UI lets users pick gemini-1.5-pro vs flash).
"""

from __future__ import annotations

from crewai import Agent

from finance_tools import analyst_tools, research_tools, writer_tools
from config import get_llm, max_iter


# ── Agent factories ───────────────────────────────────────────────────────────

def create_research_agent(model: str | None = None) -> Agent:
    """
    Research Agent
    ─────────────
    Role  : Live data gatherer — web search + news + site scraping.
    Goal  : Collect current price data, recent news, earnings history,
            analyst ratings, and macro context for the given ticker/company.
    """
    return Agent(
        role="Senior Financial Research Analyst",
        goal=(
            "Gather comprehensive, up-to-date data on the given stock or company. "
            "Collect: current share price, 52-week range, market cap, recent earnings, "
            "analyst consensus, key news from the last 30 days, and sector/macro context."
        ),
        backstory=(
            "You are a Bloomberg-trained financial researcher with 15 years of experience "
            "covering equities across tech, energy, and consumer sectors. "
            "You are obsessive about source quality — you always note the date and URL of "
            "each data point, and you flag uncertain or stale information explicitly."
        ),
        tools=research_tools(),
        llm=get_llm(model),
        max_iter=max_iter(),
        verbose=True,
        allow_delegation=False,   # Research agent works alone; no sub-delegation
    )


def create_analyst_agent(model: str | None = None) -> Agent:
    """
    Financial Analyst Agent
    ───────────────────────
    Role  : Deep financial analysis and synthesis.
    Goal  : Analyse the research data, calculate key ratios, benchmark vs peers,
            and synthesise an investment thesis with bull/bear scenarios.
    """
    return Agent(
        role="Principal Financial Analyst",
        goal=(
            "Using the research context, perform a thorough financial analysis: "
            "evaluate P/E, P/S, EV/EBITDA, debt ratios, and FCF yield. "
            "Compare against sector peers. Build a bull case, bear case, and base case. "
            "Arrive at a clear investment view: Buy / Hold / Sell with a price target range."
        ),
        backstory=(
            "You are a CFA charterholder who spent a decade as a sell-side equity analyst "
            "at Goldman Sachs before joining a long/short hedge fund. "
            "You think in frameworks — DCF, comps, sum-of-the-parts — and you never state "
            "a recommendation without quantifying the key risks."
        ),
        tools=analyst_tools(),
        llm=get_llm(model),
        max_iter=max_iter(),
        verbose=True,
        allow_delegation=False,
    )


def create_validator_agent(model: str | None = None) -> Agent:
    """
    Validator Agent  ← ADDED vs. original architecture
    ────────────────
    Role  : Fact-checker and quality gate between analysis and writing.
    Goal  : Verify key figures, flag any unsupported claims, and ensure the
            analysis is internally consistent before the report is drafted.

    Why add this?
      The Analyst agent may extrapolate from partial data.  Having a dedicated
      critic catches hallucinated numbers before they appear in the final report,
      which is critical for a finance use-case.
    """
    return Agent(
        role="Independent Research Validator",
        goal=(
            "Review the analyst's output for factual accuracy and logical consistency. "
            "Cross-check key financial figures against the raw research data. "
            "Flag any claim that cannot be directly traced to a cited source. "
            "Output a brief validation summary: PASS / PASS WITH CAVEATS / FAIL, "
            "listing any corrections required."
        ),
        backstory=(
            "You are a forensic accountant and former SEC examiner. Your job is to "
            "stress-test financial analysis before it reaches investors. "
            "You are professionally sceptical and you never let unsourced numbers slide."
        ),
        tools=analyst_tools(),   # Can re-search to verify specific figures
        llm=get_llm(model),
        max_iter=max_iter(),
        verbose=True,
        allow_delegation=False,
    )


def create_writer_agent(model: str | None = None) -> Agent:
    """
    Report Writer Agent
    ───────────────────
    Role  : Transforms validated analysis into a polished, structured report.
    Goal  : Produce a professional equity research report in Markdown,
            suitable for an institutional investor audience.
    """
    return Agent(
        role="Senior Financial Writer",
        goal=(
            "Write a clear, well-structured equity research report based on the validated "
            "analysis. The report must include: Executive Summary, Company Overview, "
            "Financial Analysis, Valuation, Risk Factors, Investment Recommendation, "
            "and Disclaimer. Use Markdown with headings, tables, and bullet lists."
        ),
        backstory=(
            "You are a former financial journalist (FT, WSJ) turned equity research writer. "
            "You distil complex analysis into reports that are precise, readable, and actionable. "
            "You never invent figures — you only use what appears in the validated analysis "
            "passed to you. You always include a standard disclaimer."
        ),
        tools=writer_tools(),
        llm=get_llm(model),
        max_iter=max_iter(),
        verbose=True,
        allow_delegation=False,
    )
