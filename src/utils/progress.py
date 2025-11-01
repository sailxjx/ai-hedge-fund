from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text

console = Console()


class AgentProgress:
    """Manages progress tracking for multiple agents with thread-safe updates."""

    def __init__(self):
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.table: Table | None = None
        self.live: Live | None = None
        self.started = False
        self.current_trading_date: str | None = None
        self.update_handlers: List[Callable[[str, Optional[str], str, Optional[str], str], None]] = []
        self._log_path = Path("log") / "backtest_progress.log"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread_id: int | None = None

    def register_handler(self, handler: Callable[[str, Optional[str], str, Optional[str], str], None]):
        """Register a handler invoked as handler(agent, ticker, status, analysis, timestamp)."""
        self.update_handlers.append(handler)
        return handler

    def unregister_handler(self, handler: Callable[[str, Optional[str], str, Optional[str], str], None]):
        """Unregister a previously registered handler."""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)

    def start(self):
        """Start the progress display and capture the active event loop for thread dispatch."""
        if self.started:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._loop_thread_id = threading.get_ident()
        self.agent_status.clear()
        self.table = Table(show_header=False, box=None, padding=(0, 1))
        self.live = Live(self.table, console=console, refresh_per_second=4)
        self.live.start()
        self.started = True

    def stop(self):
        """Stop the progress display and release runtime references."""
        if not self.started:
            return
        if self.live:
            self.live.stop()
        self.live = None
        self.table = None
        self.started = False
        self.current_trading_date = None
        self._loop = None
        self._loop_thread_id = None

    def update_status(self, agent_name: str, ticker: Optional[str] = None, status: str = "", analysis: Optional[str] = None):
        """Update the status of a specific agent."""
        if not self.started:
            self.start()

        current_thread = threading.get_ident()
        if (
            self._loop
            and self._loop_thread_id is not None
            and current_thread != self._loop_thread_id
        ):
            self._loop.call_soon_threadsafe(partial(self._update_status_sync, agent_name, ticker, status, analysis))
            return
        if self._loop is None and self._loop_thread_id is not None and current_thread != self._loop_thread_id:
            console.call_from_thread(self._update_status_sync, agent_name, ticker, status, analysis)
            return

        self._update_status_sync(agent_name, ticker, status, analysis)

    def _update_status_sync(self, agent_name: str, ticker: Optional[str], status: str, analysis: Optional[str]):
        """Synchronous implementation that performs the actual update work."""
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()

        status_entry = self.agent_status.setdefault(
            agent_name,
            {"status": status, "ticker": ticker, "analysis": analysis, "_last_update_dt": now, "timestamp": timestamp},
        )
        if ticker:
            status_entry["ticker"] = ticker
        if status:
            status_entry["status"] = status
        if analysis:
            status_entry["analysis"] = analysis
        previous = status_entry.get("_last_update_dt")
        if isinstance(previous, datetime):
            elapsed = (now - previous).total_seconds()
        else:
            elapsed = None
        status_entry["_last_update_dt"] = now
        status_entry["timestamp"] = timestamp

        log_entry = {
            "timestamp": timestamp,
            "agent": agent_name,
            "ticker": status_entry.get("ticker"),
            "status": status_entry.get("status"),
        }
        if self.current_trading_date:
            log_entry["trading_date"] = self.current_trading_date
        if analysis:
            log_entry["analysis"] = analysis[:500]
        if elapsed is not None:
            log_entry["elapsed_since_last"] = elapsed

        try:
            with self._log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
        except OSError:
            pass

        for handler in list(self.update_handlers):
            handler(agent_name, ticker, status, analysis, timestamp)

        self._refresh_display()

    def _refresh_display(self):
        """Render the progress table."""
        if not self.table or not self.live:
            return

        self.table.columns.clear()
        self.table.add_column(width=100)

        def sort_key(item: tuple[str, dict]):
            agent_name = item[0]
            if "risk_management" in agent_name:
                return (2, agent_name)
            if "portfolio_manager" in agent_name:
                return (3, agent_name)
            return (1, agent_name)

        for agent_name, info in sorted(self.agent_status.items(), key=sort_key):
            status = info.get("status", "")
            ticker = info.get("ticker")
            if status.lower() == "done":
                style = Style(color="green", bold=True)
                symbol = "✓"
            elif status.lower() == "error":
                style = Style(color="red", bold=True)
                symbol = "✗"
            else:
                style = Style(color="yellow")
                symbol = "⋯"

            agent_display = self._get_display_name(agent_name)
            status_text = Text()
            status_text.append(f"{symbol} ", style=style)
            status_text.append(f"{agent_display:<20}", style=Style(bold=True))

            if ticker:
                status_text.append(f"[{ticker}] ", style=Style(color="cyan"))
            status_text.append(status, style=style)

            self.table.add_row(status_text)

    def get_all_status(self):
        """Return a snapshot of agent statuses."""
        snapshot: Dict[str, Dict[str, Optional[str]]] = {}
        for agent_name, info in self.agent_status.items():
            snapshot[agent_name] = {
                "ticker": info.get("ticker"),
                "status": info.get("status"),
                "display_name": self._get_display_name(agent_name),
            }
        return snapshot

    def _get_display_name(self, agent_name: str) -> str:
        return agent_name.replace("_agent", "").replace("_", " ").title()

    async def aupdate_status(self, agent_name: str, ticker: Optional[str] = None, status: str = "", analysis: Optional[str] = None):
        """Async-friendly status update helper."""
        self.update_status(agent_name, ticker, status, analysis)

    def set_trading_date(self, trading_date: Optional[str]) -> None:
        """Annotate subsequent log entries with the active trading date."""
        self.current_trading_date = trading_date


# Create a global instance
progress = AgentProgress()
