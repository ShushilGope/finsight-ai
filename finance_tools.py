"""
tools/finance_tools.py
──────────────────────
All CrewAI tools used by the three agents.

Design notes
────────────
• Each tool wraps a single, testable function so unit tests don't need CrewAI.
• Retry logic lives here, not in agents, to keep agent prompts clean.
• SerperFinanceTool and SerperNewsTool are separate so the Research agent can
  call them independently and the Analyst agent can search SEC/earnings filings
  with a tighter query.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from crewai_tools import ScrapeWebsiteTool
from crewai.tools import BaseTool
from pydantic import Field


# ── Internal helpers ──────────────────────────────────────────────────────────

def _serper_request(query: str, search_type: str = "search", n_results: int = 6) -> dict[str, Any]:
    """
    Low-level Serper.dev call with 3-attempt exponential backoff.

    Args:
        query:       Search query string.
        search_type: "search" | "news" | "scholar"
        n_results:   Number of organic results to request.

    Returns:
        Parsed JSON response dict from Serper.
    """
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set.")

    url     = f"https://google.serper.dev/{search_type}"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": n_results}

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == 2:
                raise RuntimeError(f"Serper request failed after 3 attempts: {exc}") from exc
            time.sleep(2 ** attempt)   # 1 s, 2 s back-off

    return {}  # unreachable, keeps type checkers happy


def _format_serper_results(data: dict[str, Any]) -> str:
    """Convert Serper JSON to a compact, LLM-friendly string."""
    lines: list[str] = []

    # Knowledge Graph snippet (if present)
    if kg := data.get("knowledgeGraph"):
        lines.append(f"[Knowledge Graph] {kg.get('title', '')} — {kg.get('description', '')}")

    # Answer box
    if ab := data.get("answerBox"):
        lines.append(f"[Answer] {ab.get('answer') or ab.get('snippet', '')}")

    # Organic results
    for r in data.get("organic", []):
        lines.append(f"• {r.get('title', '')}\n  {r.get('link', '')}\n  {r.get('snippet', '')}")

    # News results
    for r in data.get("news", []):
        lines.append(f"• [News] {r.get('title', '')} ({r.get('date', '')})\n  {r.get('link', '')}\n  {r.get('snippet', '')}")

    return "\n".join(lines) or "No results found."


# ── CrewAI Tool definitions ───────────────────────────────────────────────────

class SerperWebSearchTool(BaseTool):
    """
    General-purpose web search powered by Serper.dev.
    Used by the Research agent to gather live company/market data.
    """
    name: str        = "web_search"
    description: str = (
        "Search the web for current information about a company, stock, or financial topic. "
        "Input should be a concise search query string. "
        "Returns titles, URLs, and snippets from top search results."
    )

    def _run(self, query: str) -> str:
        data = _serper_request(query, search_type="search", n_results=6)
        return _format_serper_results(data)


class SerperNewsSearchTool(BaseTool):
    """
    News-specific search — prioritises recent articles.
    Used by the Research agent for earnings announcements, M&A news, etc.
    """
    name: str        = "news_search"
    description: str = (
        "Search for recent news articles about a company or financial event. "
        "Input should be a search query. Returns news headlines, dates, and snippets."
    )

    def _run(self, query: str) -> str:
        data = _serper_request(query, search_type="news", n_results=8)
        return _format_serper_results(data)


class SerperFilingsSearchTool(BaseTool):
    """
    SEC / regulatory filings-focused search.
    Used by the Financial Analyst agent to locate 10-K, 10-Q, 8-K filings.
    """
    name: str        = "filings_search"
    description: str = (
        "Search for SEC filings, earnings reports, annual reports, or regulatory documents "
        "for a specific company. Prefix the query with the ticker or company name."
    )

    def _run(self, query: str) -> str:
        filings_query = f"SEC filing 10-K 10-Q {query} site:sec.gov OR site:macrotrends.net OR site:wisesheets.io"
        data = _serper_request(filings_query, search_type="search", n_results=5)
        return _format_serper_results(data)


# ── Re-export ScrapeWebsiteTool (from crewai_tools) ───────────────────────────
# We wrap it so future changes (e.g. adding auth headers) stay in one place.

def get_scrape_tool() -> ScrapeWebsiteTool:
    """
    Returns a configured ScrapeWebsiteTool instance.
    The Research agent uses this to pull full article or filing content
    once the URL is known from search results.
    """
    return ScrapeWebsiteTool()


# ── Convenience bundles for each agent ────────────────────────────────────────

def research_tools() -> list:
    return [SerperWebSearchTool(), SerperNewsSearchTool(), get_scrape_tool()]


def analyst_tools() -> list:
    return [SerperFilingsSearchTool(), get_scrape_tool()]


def writer_tools() -> list:
    # Writer only needs the scraper to optionally verify a citation URL.
    return [get_scrape_tool()]
