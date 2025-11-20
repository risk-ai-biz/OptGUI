"""Example data generators and stub optimizer for the OptGUI prototype.

This module centralizes the tiny sample datasets used by the GUI so the
widgets can stay focused on layout rather than data fabrication. The goal is
not to represent production scale inputs but to surface the expected schemas
and a simple trade suggestion flow end-to-end.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict

import polars as pl


def load_static_snapshot(as_of: dt.date) -> Dict[str, pl.DataFrame]:
    """Return a small synthetic SoD snapshot.

    The fields mirror what a production loader would provide but are tiny so
they can be browsed comfortably in the widget grid.
    """

    df_instr = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "MSFT", "SPY", "QQQ", "ESZ5"],
            "ticker": ["AAPL", "MSFT", "SPY", "QQQ", "ESZ5"],
            "type": ["Equity", "Equity", "ETF", "ETF", "Future"],
            "universe_flags": [
                ["core_equity"],
                ["core_equity"],
                ["broad_etf"],
                ["sector_etf"],
                ["index_future"],
            ],
            "currency": ["USD"] * 5,
            "sector": ["Technology", "Technology", "Multi", "Multi", "Index Future"],
            "adv_usd": [2.5e9, 2.0e9, 3.0e9, 1.8e9, 15e9],
        }
    )

    df_risk = pl.DataFrame(
        {
            "instrument_id": df_instr["instrument_id"],
            "has_factor_loadings": [True, True, True, True, True],
            "idio_var": [0.15, 0.12, 0.08, 0.10, 0.05],
        }
    )

    df_costs = pl.DataFrame(
        {
            "instrument_id": df_instr["instrument_id"],
            "spread_bps": [5.0, 4.5, 1.0, 1.2, 0.5],
            "borrow_rate": [0.02, 0.01, 0.0, 0.0, 0.0],
            "impact_coeff": [15.0, 13.0, 8.0, 9.0, 3.0],
        }
    )

    df_restr = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "MSFT"],
            "restricted": [False, False],
            "no_new_shorts": [False, True],
        }
    )

    df_constraints = pl.DataFrame(
        {
            "constraint": ["max_single_name", "sector_cap"],
            "description": [
                "Max 10% NAV per single name including hedges",
                "Cap Technology to 40% of gross exposure",
            ],
            "limit_type": ["position_nav_pct", "sector_gross_pct"],
            "limit_value": [10.0, 40.0],
        }
    )

    return {
        "instruments": df_instr,
        "risk_model_summary": df_risk,
        "cost_params": df_costs,
        "restrictions": df_restr,
        "constraints": df_constraints,
        "as_of": pl.DataFrame({"as_of": [as_of.isoformat()]}),
    }


def load_intraday_state(portfolio_id: str, as_of: dt.date) -> Dict[str, pl.DataFrame]:
    """Return a toy intraday snapshot for a portfolio."""
    df_pos = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "MSFT", "SPY", "QQQ", "ESZ5"],
            "position": [1_000_000.0, 500_000.0, 2_000_000.0, 750_000.0, -5_000_000.0],
            "price": [190.0, 340.0, 500.0, 420.0, 5200.0],
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


def compute_stub_trades(snapshot: Dict[str, pl.DataFrame], intraday: Dict[str, pl.DataFrame]):
    """Generate a deterministic set of suggested trades.

    The rules are intentionally simple: trim large ETF exposures, top-up
    technology singles, and lighten the future short. This keeps the DataGrid
    readable while demonstrating how optimizer output could be rendered.
    """

    df_pos = intraday["positions"]
    join_prices = df_pos.select("instrument_id", "price")
    suggestions = pl.DataFrame(
        {
            "instrument_id": ["AAPL", "MSFT", "SPY", "ESZ5"],
            "target_delta_notional": [250_000.0, 250_000.0, -400_000.0, 800_000.0],
            "rationale": [
                "Add to tech momentum name",
                "Add to quality tech with shorts disabled",
                "Trim broad ETF to fund singles",
                "Reduce short future bias",
            ],
        }
    )

    trades = suggestions.join(join_prices, on="instrument_id", how="left")
    trades = trades.with_columns(
        (pl.col("target_delta_notional") / pl.col("price")).alias("shares"),
    )

    return trades
