"""
RunConfig.py

Configuration and run-context management for the single-period optimization engine.

This module is intentionally independent of the optimizer itself.
It handles:

- Loading YAML configuration (global, run types, portfolios)
- Providing a typed RunConfig object usable by both batch and GUI workflows
- Creating a resolved RunContext for a specific run (portfolio, run type, cash injection, etc.)

Dependencies:
    pip install pyyaml polars

Note:
    This module does not depend on Mosek or ipywidgets. It is pure config/orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path
import datetime as dt

import yaml
import polars as pl


# ---------------------------------------------------------------------------
# Dataclasses for configuration
# ---------------------------------------------------------------------------


@dataclass
class GlobalSettings:
    """Global configuration fields that apply across portfolios and run types."""

    data_root: Path
    log_root: Path
    default_run_type: str
    default_risk_aversion: float
    default_cash_buffer_pct: float
    default_asof_days_lag: int = 0


@dataclass
class RunTypeSettings:
    """
    Settings for a given logical run type.

    Example run types:
        - main_rebalance
        - cash_injection
        - true_up
        - hedge_finder
        - efp_arb
    """

    name: str
    description: str
    apply_alpha: bool
    allow_etf_create_redeem: bool
    allow_efp: bool
    allow_new_shorts: bool
    risk_aversion_multiplier: float
    turnover_cap_pct_gross: float
    enforce_factor_neutrality: bool
    soft_factor_neutrality_penalty: float


@dataclass
class PortfolioSettings:
    """
    Static configuration for a single portfolio.

    These are higher-level constraints and defaults, not day-to-day positions.
    """

    portfolio_id: str
    name: str
    base_universe_flags: list[str]
    hedge_universe_flags: list[str]
    benchmark_id: str
    leverage_max: float
    gross_exposure_max_pct_nav: float
    net_exposure_target_pct_nav: float
    net_exposure_band_pct_nav: float
    turnover_cap_pct_gross: float
    allow_new_shorts: bool
    risk_aversion: float
    cash_buffer_pct: float
    default_run_type: str


@dataclass
class RunContext:
    """
    Concrete resolved context for a single optimization run.

    This is what you pass to the optimizer and GUI: it contains the portfolio,
    run type, as-of date, and all derived numeric settings.
    """

    portfolio: PortfolioSettings
    run_type: RunTypeSettings
    as_of: dt.date

    # dynamic / per-run parameters:
    cash_injection: float = 0.0
    custom_risk_aversion: Optional[float] = None

    # derived fields (pre-computed for convenience)
    effective_risk_aversion: float = field(init=False)
    effective_turnover_cap_pct_gross: float = field(init=False)

    def __post_init__(self) -> None:
        # risk aversion: portfolio base * run_type multiplier, unless overridden
        base = self.portfolio.risk_aversion
        mult = self.run_type.risk_aversion_multiplier
        if self.custom_risk_aversion is not None:
            self.effective_risk_aversion = self.custom_risk_aversion
        else:
            self.effective_risk_aversion = base * mult

        # turnover cap: min(portfolio cap, run_type cap) as a simple default rule
        self.effective_turnover_cap_pct_gross = min(
            self.portfolio.turnover_cap_pct_gross,
            self.run_type.turnover_cap_pct_gross,
        )


@dataclass
class RunConfig:
    """
    Top-level configuration object, typically created from a YAML file.

    Example usage:

        cfg = RunConfig.from_yaml("config/example_config.yaml")
        ctx = cfg.make_run_context(portfolio_id="EQ_CORE", run_type_name="cash_injection")

    The RunContext can then be used by both the GUI and back-end optimizer.
    """

    global_settings: GlobalSettings
    run_types: Dict[str, RunTypeSettings]
    portfolios: Dict[str, PortfolioSettings]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Load configuration from a YAML file and build a RunConfig object."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            raw = yaml.safe_load(f)

        # --- Global settings ---
        g = raw.get("global", {})
        global_settings = GlobalSettings(
            data_root=Path(g.get("data_root", "data")),
            log_root=Path(g.get("log_root", "logs")),
            default_run_type=g.get("default_run_type", "main_rebalance"),
            default_risk_aversion=float(g.get("default_risk_aversion", 1.0)),
            default_cash_buffer_pct=float(g.get("default_cash_buffer_pct", 0.05)),
            default_asof_days_lag=int(g.get("default_asof_days_lag", 0)),
        )

        # --- Run types ---
        run_types_section = raw.get("run_types", {})
        run_types: Dict[str, RunTypeSettings] = {}
        for name, cfg in run_types_section.items():
            run_types[name] = RunTypeSettings(
                name=name,
                description=cfg.get("description", ""),
                apply_alpha=bool(cfg.get("apply_alpha", True)),
                allow_etf_create_redeem=bool(cfg.get("allow_etf_create_redeem", False)),
                allow_efp=bool(cfg.get("allow_efp", False)),
                allow_new_shorts=bool(cfg.get("allow_new_shorts", True)),
                risk_aversion_multiplier=float(cfg.get("risk_aversion_multiplier", 1.0)),
                turnover_cap_pct_gross=float(cfg.get("turnover_cap_pct_gross", 0.25)),
                enforce_factor_neutrality=bool(cfg.get("enforce_factor_neutrality", False)),
                soft_factor_neutrality_penalty=float(cfg.get("soft_factor_neutrality_penalty", 0.0)),
            )

        # --- Portfolios ---
        portfolios_section = raw.get("portfolios", {})
        portfolios: Dict[str, PortfolioSettings] = {}
        for pid, cfg in portfolios_section.items():
            portfolios[pid] = PortfolioSettings(
                portfolio_id=pid,
                name=cfg.get("name", pid),
                base_universe_flags=list(cfg.get("base_universe_flags", [])),
                hedge_universe_flags=list(cfg.get("hedge_universe_flags", [])),
                benchmark_id=cfg.get("benchmark_id", "NONE"),
                leverage_max=float(cfg.get("leverage_max", 1.0)),
                gross_exposure_max_pct_nav=float(cfg.get("gross_exposure_max_pct_nav", 100.0)),
                net_exposure_target_pct_nav=float(cfg.get("net_exposure_target_pct_nav", 0.0)),
                net_exposure_band_pct_nav=float(cfg.get("net_exposure_band_pct_nav", 0.0)),
                turnover_cap_pct_gross=float(cfg.get("turnover_cap_pct_gross", 0.25)),
                allow_new_shorts=bool(cfg.get("allow_new_shorts", True)),
                risk_aversion=float(cfg.get("risk_aversion", global_settings.default_risk_aversion)),
                cash_buffer_pct=float(cfg.get("cash_buffer_pct", global_settings.default_cash_buffer_pct)),
                default_run_type=cfg.get("default_run_type", global_settings.default_run_type),
            )

        rc = cls(global_settings=global_settings, run_types=run_types, portfolios=portfolios)
        rc._validate()  # sanity checks
        return rc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_run_context(
        self,
        portfolio_id: str,
        run_type_name: Optional[str] = None,
        *,
        as_of: Optional[dt.date] = None,
        cash_injection: float = 0.0,
        custom_risk_aversion: Optional[float] = None,
    ) -> RunContext:
        """
        Create a RunContext using configuration and high-level user inputs.

        Args:
            portfolio_id: portfolio key as specified in YAML.
            run_type_name: one of the configured run_types. If omitted, use
                           the portfolio's default_run_type.
            as_of: date of the run. If None, uses today's date minus
                   global.default_asof_days_lag.
            cash_injection: cash amount added to the portfolio in base currency.
            custom_risk_aversion: optional override of risk aversion.

        Returns:
            RunContext
        """
        if portfolio_id not in self.portfolios:
            raise KeyError(f"Unknown portfolio_id: {portfolio_id}")

        portfolio = self.portfolios[portfolio_id]

        if run_type_name is None:
            run_type_name = portfolio.default_run_type

        if run_type_name not in self.run_types:
            raise KeyError(f"Unknown run_type: {run_type_name}")

        run_type = self.run_types[run_type_name]

        if as_of is None:
            today = dt.date.today()
            lag = self.global_settings.default_asof_days_lag
            as_of = today - dt.timedelta(days=lag)

        ctx = RunContext(
            portfolio=portfolio,
            run_type=run_type,
            as_of=as_of,
            cash_injection=float(cash_injection),
            custom_risk_aversion=custom_risk_aversion,
        )
        return ctx

    # ------------------------------------------------------------------
    # Utility helpers (e.g., for GUI)
    # ------------------------------------------------------------------

    @property
    def portfolio_ids(self) -> list[str]:
        """Return portfolio IDs in a stable sorted order."""
        return sorted(self.portfolios.keys())

    @property
    def run_type_names(self) -> list[str]:
        """Return run type names in a stable sorted order."""
        return sorted(self.run_types.keys())

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Simple validation to catch common misconfigurations."""
        if not self.run_types:
            raise ValueError("No run_types defined in configuration.")
        if not self.portfolios:
            raise ValueError("No portfolios defined in configuration.")

        # Check that each portfolio's default run type exists
        for pid, p in self.portfolios.items():
            if p.default_run_type not in self.run_types:
                raise ValueError(
                    f"Portfolio {pid} has default_run_type={p.default_run_type} "
                    f"which is not present in run_types ({list(self.run_types.keys())})."
                )

        # Basic numeric sanity checks (can be extended)
        for name, rt in self.run_types.items():
            if rt.risk_aversion_multiplier <= 0:
                raise ValueError(f"RunType {name}: risk_aversion_multiplier must be > 0.")
            if rt.turnover_cap_pct_gross < 0:
                raise ValueError(f"RunType {name}: turnover_cap_pct_gross must be >= 0.")

        for pid, p in self.portfolios.items():
            if p.leverage_max <= 0:
                raise ValueError(f"Portfolio {pid}: leverage_max must be > 0.")
            if p.gross_exposure_max_pct_nav <= 0:
                raise ValueError(f"Portfolio {pid}: gross_exposure_max_pct_nav must be > 0.")


# ---------------------------------------------------------------------------
# Simple tests (can be moved into a separate test file for pytest)
# ---------------------------------------------------------------------------


def _selftest_example_config() -> None:
    """
    Simple sanity test that config/example_config.yaml can be loaded and
    that make_run_context works.

    This is not a substitute for proper unit tests, but useful during
    early development.
    """
    cfg_path = Path("config/example_config.yaml")
    if not cfg_path.exists():
        print("WARNING: example_config.yaml not found; selftest skipped.")
        return

    cfg = RunConfig.from_yaml(cfg_path)
    print(f"Loaded config with {len(cfg.portfolios)} portfolios, {len(cfg.run_types)} run types.")

    ctx = cfg.make_run_context(portfolio_id="EQ_CORE", run_type_name="cash_injection")
    print("RunContext example:")
    print(f"  portfolio: {ctx.portfolio.portfolio_id} ({ctx.portfolio.name})")
    print(f"  run_type:  {ctx.run_type.name}")
    print(f"  as_of:     {ctx.as_of}")
    print(f"  eff RA:    {ctx.effective_risk_aversion:.3f}")
    print(f"  eff TO cap:{ctx.effective_turnover_cap_pct_gross:.1f}% of gross")


if __name__ == "__main__":
    _selftest_example_config()
