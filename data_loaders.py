"""Data loading helpers for the optimization GUI.

These functions currently return small placeholder Polars DataFrames so the GUI
is interactive out of the box. Replace the internals with real loaders wired to
Parquet files, databases, or services as you integrate the engine.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict

import polars as pl

__all__ = ["load_static_snapshot", "load_intraday_state"]


def load_static_snapshot(as_of: dt.date) -> Dict[str, pl.DataFrame]:
    """Load static start-of-day data for the provided date.

    Returns a dictionary keyed by logical dataset names. Each value is a Polars
    DataFrame ready to be inspected by the GUI or piped into the optimizer. The
    function is intentionally deterministic and does not rely on global state so
    it can be swapped out for a production implementation without changing the
    GUI wiring.
    """
    df_instr = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "SPY", "ESZ5"],
            "ticker": ["AAPL", "SPY", "ESZ5"],
            "type": ["Equity", "ETF", "Future"],
            "universe_flags": [["core_equity"], ["broad_etf"], ["index_future"]],
            "currency": ["USD", "USD", "USD"],
            "sector": ["Technology", "Multi", "Index Future"],
        }
    )

    df_risk = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "SPY", "ESZ5"],
            "has_factor_loadings": [True, True, True],
            "idio_var": [0.15, 0.08, 0.05],
        }
    )

    df_costs = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "SPY", "ESZ5"],
            "spread_bps": [5.0, 1.0, 0.5],
            "borrow_rate": [0.02, 0.0, 0.0],
        }
    )

    df_restr = pl.DataFrame(
        {
            "instrument_id": ["AAPL"],
            "restricted": [False],
            "no_new_shorts": [False],
        }
    )

    return {
        "instruments": df_instr,
        "risk_model_summary": df_risk,
        "cost_params": df_costs,
        "restrictions": df_restr,
    }


def load_intraday_state(portfolio_id: str, as_of: dt.date) -> Dict[str, pl.DataFrame]:
    """Load intraday state for a given portfolio and date.

    The stub returns a minimal set of positions and cash rows that roughly align
    with the static snapshot. Production implementations should mirror this
    interface so the GUI and optimizer continue to work without modification.
    """
    df_pos = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "SPY", "ESZ5"],
            "position": [1_000_000.0, 2_000_000.0, -10_000_000.0],  # notional example
            "price": [190.0, 500.0, 5200.0],
        }
    )

    df_cash = pl.DataFrame(
        {
            "portfolio_id": [portfolio_id],
            "cash": [5_000_000.0],
            "currency": ["USD"],
            "as_of": [as_of.isoformat()],
        }
    )

    return {
        "positions": df_pos,
        "cash": df_cash,
    }
