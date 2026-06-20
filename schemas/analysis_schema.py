"""
schemas/analysis_schema.py
───────────────────────────
Structured output contract for the Analyst agent.

Why this exists
────────────────
The original pipeline only ever produced free-text Markdown between agents,
which is great for the final report but useless for charts — you can't plot
a paragraph. CrewAI's `output_pydantic` lets a Task return BOTH a narrative
string AND a parsed, typed object in the same call, with no second LLM pass.

The Analyst agent is told (via the task description) to emit the scenario
numbers in a fenced ```json block at the end of its answer; CrewAI extracts
and validates it against this schema. If parsing fails, `task.output.pydantic`
is None and the UI simply skips the chart — it never blocks the text report.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioTarget(BaseModel):
    label: str = Field(..., description="Scenario name, e.g. 'Bull', 'Base', 'Bear'")
    price_target: float = Field(..., description="Numeric price target, no currency symbol")
    pct_change: float = Field(..., description="Percent change vs current price, signed (e.g. -18.7)")


class ValuationRatio(BaseModel):
    metric: str = Field(..., description="Ratio name, e.g. 'P/E (TTM)', 'Forward P/E'")
    company_value: float = Field(..., description="The subject company's value for this metric")
    peer_avg: float | None = Field(None, description="Average of peer values if available, else null")


class StructuredAnalysis(BaseModel):
    """
    Full structured payload the Analyst agent must emit as a trailing JSON block.
    All fields are required except peer_avg inside ValuationRatio and recommendation
    conviction, which can genuinely be unknown.
    """
    ticker: str
    current_price: float
    scenarios: list[ScenarioTarget] = Field(..., min_length=3, max_length=3, description="Exactly Bull, Base, Bear in that order")
    valuation_ratios: list[ValuationRatio] = Field(default_factory=list)
    recommendation: str = Field(..., description="BUY | HOLD | SELL")
    conviction: str = Field(..., description="High | Medium | Low")
