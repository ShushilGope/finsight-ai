"""
crew.py
───────
CrewAI orchestrator — assembles agents + tasks into a Crew and exposes
a clean `run()` interface used by both the CLI and the Streamlit app.

Key design choices
──────────────────
• `step_callback` feeds a queue.Queue so Streamlit can stream logs without
  blocking the main thread (threading.Thread wraps crew.kickoff).
• `@st.cache_resource` is applied at the Streamlit layer, not here, so this
  module stays framework-agnostic and unit-testable.
• Process is SEQUENTIAL — tasks run in order, each receiving prior outputs
  as context. A hierarchical process would need a manager LLM; sequential
  is cheaper and sufficient for a 4-step pipeline.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from crewai import Crew, Process

from schemas.analysis_schema import StructuredAnalysis
from finance_tasks import build_tasks
from config import validate_env


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AgentLog:
    """One structured log entry surfaced through the step callback."""
    timestamp: str
    agent: str
    action: str
    detail: str

    @classmethod
    def from_step(cls, step_output: Any) -> "AgentLog":
        """
        CrewAI passes a TaskOutput-like object to step_callback.
        We extract what we can and fall back gracefully — but the fallback
        itself is logged (not silently swallowed) so parsing issues are visible
        in the console rather than vanishing.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            agent  = getattr(step_output, "agent",  "Agent")
            action = getattr(step_output, "type",   "step")
            detail = getattr(step_output, "output", str(step_output))
            if hasattr(detail, "raw"):        # CrewAI TaskOutput
                detail = detail.raw
        except Exception as exc:  # noqa: BLE001
            print(f"[AgentLog.from_step] parse fallback triggered: {exc}")
            agent, action, detail = "Agent", "step", str(step_output)

        return cls(timestamp=ts, agent=str(agent), action=str(action), detail=str(detail)[:600])


@dataclass
class RunResult:
    """Final result returned by `run_crew()`."""
    report_md: str
    logs: list[AgentLog] = field(default_factory=list)
    error: str | None = None
    error_trace: str | None = None
    structured_analysis: StructuredAnalysis | None = None
    agent_timings: dict[str, float] = field(default_factory=dict)  # agent name -> seconds

    @property
    def success(self) -> bool:
        return self.error is None


# ── Core runner ───────────────────────────────────────────────────────────────

def run_crew(
    ticker: str,
    extra_query: str = "",
    model: str | None = None,
    log_callback: Callable[[AgentLog], None] | None = None,
) -> RunResult:
    """
    Synchronous crew runner — call from CLI or from a background thread.

    Args:
        ticker:       Stock ticker / company name.
        extra_query:  Optional user focus question.
        model:        Gemini model override.
        log_callback: Optional function called with each AgentLog as it arrives.
                      Used by Streamlit to push logs into a queue.

    Returns:
        RunResult with report Markdown, log history, structured analysis (if
        the Analyst's output parsed cleanly), per-agent timing, and on failure
        both a short error string and the full traceback for debugging.
    """
    logs: list[AgentLog] = []
    agent_start_times: dict[str, float] = {}
    agent_timings: dict[str, float] = {}

    try:
        validate_env()
    except EnvironmentError as exc:
        return RunResult(report_md="", error=str(exc))

    def _step_callback(step_output: Any) -> None:
        log = AgentLog.from_step(step_output)
        logs.append(log)

        # Track first-seen / last-seen time per agent for the timing chart.
        now = datetime.now().timestamp()
        if log.agent not in agent_start_times:
            agent_start_times[log.agent] = now
        agent_timings[log.agent] = now - agent_start_times[log.agent]

        if log_callback:
            log_callback(log)

    try:
        tasks = build_tasks(ticker=ticker, extra_query=extra_query, model=model)
        agents = [task.agent for task in tasks]

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            step_callback=_step_callback,
            verbose=False,   # We handle our own logging via step_callback
            memory=False,    # Disable embeddings store to reduce latency for this pet project
        )

        result = crew.kickoff()

        # CrewAI ≥0.80 returns a CrewOutput object; extract raw string
        report_md = result.raw if hasattr(result, "raw") else str(result)

        # Pull the Analyst task's structured output specifically (2nd task, index 1)
        # so the UI can chart bull/base/bear without parsing the Markdown report.
        structured: StructuredAnalysis | None = None
        try:
            analysis_task_output = tasks[1].output
            if analysis_task_output is not None and getattr(analysis_task_output, "pydantic", None):
                structured = analysis_task_output.pydantic
        except Exception as exc:  # noqa: BLE001
            print(f"[run_crew] structured analysis extraction failed (non-fatal): {exc}")

        return RunResult(
            report_md=report_md,
            logs=logs,
            structured_analysis=structured,
            agent_timings=agent_timings,
        )

    except Exception as exc:  # noqa: BLE001
        full_trace = traceback.format_exc()
        print(full_trace)  # always land in the terminal even if UI swallows it
        return RunResult(
            report_md="", logs=logs, error=str(exc), error_trace=full_trace,
            agent_timings=agent_timings,
        )


# ── Non-blocking wrapper for Streamlit ────────────────────────────────────────

class CrewRunner:
    """
    Runs `run_crew()` in a background thread and exposes a queue of AgentLog
    entries so Streamlit can poll and render them without blocking the event loop.

    Usage (Streamlit):
        runner = CrewRunner(ticker, extra_query, model)
        runner.start()
        while not runner.done:
            while log := runner.next_log():
                st.write(log.detail)
            time.sleep(0.5)
            st.rerun()
        result = runner.result
    """

    def __init__(self, ticker: str, extra_query: str = "", model: str | None = None):
        self.ticker      = ticker
        self.extra_query = extra_query
        self.model       = model

        self._log_queue:    queue.Queue[AgentLog] = queue.Queue()
        self._result:       RunResult | None      = None
        self._thread:       threading.Thread | None = None
        self._done_event:   threading.Event       = threading.Event()
        self._lock:         threading.Lock        = threading.Lock()
        self._live_timings: dict[str, float]      = {}
        self._live_start:   dict[str, float]      = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Kick off the crew in a daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    def result(self) -> RunResult | None:
        """Available once `done` is True."""
        return self._result

    @property
    def live_timings(self) -> dict[str, float]:
        """Thread-safe snapshot of elapsed seconds per agent so far, mid-run."""
        with self._lock:
            return dict(self._live_timings)

    def next_log(self) -> AgentLog | None:
        """Non-blocking dequeue — returns None if queue is empty."""
        try:
            log = self._log_queue.get_nowait()
        except queue.Empty:
            return None

        with self._lock:
            now = datetime.now().timestamp()
            if log.agent not in self._live_start:
                self._live_start[log.agent] = now
            self._live_timings[log.agent] = now - self._live_start[log.agent]

        return log

    def drain_logs(self) -> list[AgentLog]:
        """Drain all pending logs at once."""
        logs: list[AgentLog] = []
        while log := self.next_log():
            logs.append(log)
        return logs

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._result = run_crew(
            ticker=self.ticker,
            extra_query=self.extra_query,
            model=self.model,
            log_callback=self._log_queue.put,
        )
        self._done_event.set()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    extra  = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"\n🚀  Running finance crew for: {ticker}\n{'─'*60}")

    result = run_crew(ticker, extra)
    if result.success:
        print(result.report_md)
    else:
        print(f"❌  Error: {result.error}", file=sys.stderr)
        if result.error_trace:
            print(f"\nFull traceback:\n{result.error_trace}", file=sys.stderr)
        sys.exit(1)
