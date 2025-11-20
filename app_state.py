"""Encapsulated application state for the optimization GUI.

The GUI previously relied on module-level globals for snapshot data and run
context. AppState keeps these pieces of state together, provides small helper
methods for resolving run contexts and loading data, and maintains a running log
of user-visible messages.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import polars as pl

from RunConfig import RunConfig, RunContext
from data_loaders import load_intraday_state, load_static_snapshot


@dataclass
class AppState:
    """Container for GUI-run state and logging."""

    run_config: RunConfig
    run_context: Optional[RunContext] = None
    snapshot: Optional[Dict[str, pl.DataFrame]] = None
    intraday: Optional[Dict[str, pl.DataFrame]] = None
    log_messages: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Append a timestamped message to the log buffer."""
        ts = dt.datetime.now().strftime("%H:%M:%S")
        self.log_messages.append(f"[{ts}] {message}")

    def resolve_run_context(
        self,
        *,
        portfolio_id: str,
        run_type_name: str,
        as_of: dt.date,
        cash_injection: float,
        custom_risk_aversion: Optional[float] = None,
    ) -> RunContext:
        """Create and store a RunContext using the underlying RunConfig."""
        self.run_context = self.run_config.make_run_context(
            portfolio_id=portfolio_id,
            run_type_name=run_type_name,
            as_of=as_of,
            cash_injection=cash_injection,
            custom_risk_aversion=custom_risk_aversion,
        )
        self.log(
            "Resolved run context with RA="
            f"{self.run_context.effective_risk_aversion:.3f}, TO cap="
            f"{self.run_context.effective_turnover_cap_pct_gross:.1f}%"
        )
        return self.run_context

    def load_data(self, portfolio_id: str, as_of: dt.date) -> tuple[Dict[str, pl.DataFrame], Dict[str, pl.DataFrame]]:
        """Load static snapshot and intraday state for the selected run."""
        self.snapshot = load_static_snapshot(as_of)
        self.intraday = load_intraday_state(portfolio_id, as_of)
        self.log(
            f"Loaded snapshot for {portfolio_id} as of {as_of}; "
            f"universe size={self.snapshot['instruments'].height}"
        )
        return self.snapshot, self.intraday

    def ensure_data_ready(self) -> tuple[RunContext, Dict[str, pl.DataFrame], Dict[str, pl.DataFrame]]:
        """Return cached state or raise if load_data has not run."""
        if self.run_context is None or self.snapshot is None or self.intraday is None:
            raise RuntimeError("Run context and data must be loaded before running the optimizer stub.")
        return self.run_context, self.snapshot, self.intraday

    def clear_logs(self) -> None:
        self.log_messages.clear()

    def flush_logs(self) -> None:
        """Print and clear accumulated log messages."""
        for msg in self.log_messages:
            print(msg)
        self.clear_logs()
