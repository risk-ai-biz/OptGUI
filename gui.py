"""Tabbed ipywidgets GUI for the single-period optimization workflow.

The previous single-column widget layout has been expanded into the
specification's tabbed experience:

Tab 1 – Config
    Load configuration, build run contexts, and surface quick metadata.

Tab 2 – Data & Universe
    Load snapshots, inspect instruments/risk/costs, and view QC messages.

Tab 3 – Constraints
    Summaries of instrument, group, and portfolio-level constraints.

Tab 4 – Optimization Settings
    Risk/cost/synthetic/margin toggles with effective overrides.

Tab 5 – Run & Results
    Run stub optimization, display summaries, and placeholder trade grids.

The code is organized to keep callbacks small and stateless where possible so
you can swap in real loaders or optimizer calls without rewiring the UI.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Iterable, Optional

import ipywidgets as widgets
import polars as pl
from IPython.display import display
from ipydatagrid import DataGrid

from RunConfig import RunConfig, RunContext
from app_state import AppState


# ---------------------------------------------------------------------------
# Load configuration and initialize app state
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("config/example_config.yaml")


def _load_config(path: Path) -> RunConfig:
    cfg = RunConfig.from_yaml(path)
    print(
        f"Loaded config from {path}: {len(cfg.portfolios)} portfolios, "
        f"{len(cfg.run_types)} run types."
    )
    return cfg


run_config = _load_config(DEFAULT_CONFIG_PATH)
app_state = AppState(run_config)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_selected_asof(date_picker: widgets.DatePicker, cfg: RunConfig) -> dt.date:
    """Return selected as-of or fallback to today minus configured lag."""
    if date_picker.value is not None:
        return date_picker.value
    today = dt.date.today()
    lag = cfg.global_settings.default_asof_days_lag
    return today - dt.timedelta(days=lag)


def _make_grid(df: pl.DataFrame, *, height: str = "260px") -> DataGrid:
    """Create a DataGrid from a Polars DataFrame with sensible defaults."""
    return DataGrid(df.to_pandas(), layout={"height": height})


def _dict_to_table(mapping: Dict[str, str], title_key: str = "Field", value_key: str = "Value") -> pl.DataFrame:
    return pl.DataFrame({title_key: list(mapping.keys()), value_key: list(mapping.values())})


def _make_placeholder_df(rows: Iterable[Dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core widgets shared across tabs
# ---------------------------------------------------------------------------


config_path_input = widgets.Text(
    value=str(DEFAULT_CONFIG_PATH), description="Config path:", layout=widgets.Layout(width="420px")
)
reload_config_button = widgets.Button(description="Reload", icon="refresh", button_style="info")

portfolio_dropdown = widgets.Dropdown(
    options=[(p.name, pid) for pid, p in run_config.portfolios.items()],
    description="Portfolio:",
    layout=widgets.Layout(width="300px"),
)

run_type_dropdown = widgets.Dropdown(
    options=[(rt.name, rt.name) for rt in run_config.run_types.values()],
    description="Run type:",
    layout=widgets.Layout(width="300px"),
)

asof_picker = widgets.DatePicker(description="As-of:")
cash_injection_input = widgets.FloatText(value=0.0, description="Cash Δ:", layout=widgets.Layout(width="220px"))
risk_aversion_override = widgets.FloatText(
    value=None,
    description="RA override:",
    placeholder="Optional",
    layout=widgets.Layout(width="220px"),
)
apply_context_button = widgets.Button(description="Build context", button_style="primary", icon="check")

load_snapshot_button = widgets.Button(description="Load snapshot", button_style="info", icon="download")
run_optimizer_button = widgets.Button(description="Run optimizer (stub)", button_style="success", icon="play")

# Output areas shared across tabs
context_summary_output = widgets.Output()
data_log_output = widgets.Output(layout={"border": "1px solid #ddd"})
universe_output = widgets.Output()
risk_output = widgets.Output()
costs_output = widgets.Output()
restrictions_output = widgets.Output()
constraint_output = widgets.Output()
settings_output = widgets.Output()
results_output = widgets.Output()
trade_output = widgets.Output()
mcp_output = widgets.Output()


# ---------------------------------------------------------------------------
# Callback helpers
# ---------------------------------------------------------------------------


def _resolve_context() -> Optional[RunContext]:
    """Resolve and cache run context, updating the summary box."""
    context_summary_output.clear_output()
    try:
        ctx = app_state.resolve_run_context(
            portfolio_id=portfolio_dropdown.value,
            run_type_name=run_type_dropdown.value,
            as_of=_get_selected_asof(asof_picker, app_state.run_config),
            cash_injection=float(cash_injection_input.value or 0.0),
            custom_risk_aversion=
            float(risk_aversion_override.value)
            if risk_aversion_override.value not in (None, "")
            else None,
        )
    except Exception as exc:  # noqa: BLE001 - simple UI error surfacing
        with context_summary_output:
            print(f"Error: {exc}")
        return None

    with context_summary_output:
        summary = {
            "Portfolio": f"{ctx.portfolio.portfolio_id} — {ctx.portfolio.name}",
            "Run type": f"{ctx.run_type.name}: {ctx.run_type.description}",
            "As-of": str(ctx.as_of),
            "Cash Δ": f"{ctx.cash_injection:,.2f}",
            "Effective RA": f"{ctx.effective_risk_aversion:.3f}",
            "Effective TO cap": f"{ctx.effective_turnover_cap_pct_gross:.1f}% of gross",
        }
        display(_make_grid(_dict_to_table(summary)))

    return ctx


def _refresh_dropdowns(cfg: RunConfig) -> None:
    portfolio_dropdown.options = [(p.name, pid) for pid, p in cfg.portfolios.items()]
    run_type_dropdown.options = [(rt.name, rt.name) for rt in cfg.run_types.values()]


def _on_reload_config_clicked(_) -> None:
    global run_config
    cfg_path = Path(config_path_input.value)
    context_summary_output.clear_output()
    if not cfg_path.exists():
        with context_summary_output:
            print(f"Config not found: {cfg_path}")
        return

    run_config = _load_config(cfg_path)
    app_state.run_config = run_config
    _refresh_dropdowns(run_config)
    with context_summary_output:
        print("Config reloaded and dropdowns refreshed.")


reload_config_button.on_click(_on_reload_config_clicked)
apply_context_button.on_click(lambda _: _resolve_context())


# ---------------------------------------------------------------------------
# Data loading and display
# ---------------------------------------------------------------------------


def _on_load_snapshot_clicked(_) -> None:
    data_log_output.clear_output()
    universe_output.clear_output()
    risk_output.clear_output()
    costs_output.clear_output()
    restrictions_output.clear_output()

    if _resolve_context() is None:
        return

    portfolio_id = portfolio_dropdown.value
    as_of = _get_selected_asof(asof_picker, app_state.run_config)

    snapshot, intraday = app_state.load_data(portfolio_id, as_of)
    df_instr = snapshot["instruments"]
    df_risk = snapshot["risk_model_summary"]
    df_costs = snapshot["cost_params"]
    df_restr = snapshot.get("restrictions", pl.DataFrame())

    df_merged = df_instr.join(df_risk, on="instrument_id", how="left")
    missing_risk = df_merged.filter(pl.col("has_factor_loadings") != True)

    with data_log_output:
        app_state.flush_logs()
        print(f"Universe size: {df_instr.height} instruments.")
        print(f"Positions rows: {intraday['positions'].height}")
        if missing_risk.height > 0:
            print(f"WARNING: {missing_risk.height} instruments missing risk model.")
        else:
            print("All instruments have risk coverage.")

    with universe_output:
        display(_make_grid(df_instr))

    with risk_output:
        display(_make_grid(df_risk))

    with costs_output:
        display(_make_grid(df_costs))

    if df_restr.height > 0:
        with restrictions_output:
            display(_make_grid(df_restr))


load_snapshot_button.on_click(_on_load_snapshot_clicked)


# ---------------------------------------------------------------------------
# Constraint summaries
# ---------------------------------------------------------------------------


def _populate_constraints() -> None:
    constraint_output.clear_output()
    try:
        ctx, snapshot, _ = app_state.ensure_data_ready()
    except RuntimeError as exc:
        with constraint_output:
            print(str(exc))
        return

    df_restr = snapshot.get("restrictions", pl.DataFrame())
    instrument_constraints = _make_placeholder_df(
        [
            {
                "instrument_id": row["instrument_id"],
                "restricted": row.get("restricted", False),
                "no_new_shorts": row.get("no_new_shorts", False),
            }
            for row in df_restr.to_dicts()
        ]
        or [
            {
                "instrument_id": "—",
                "restricted": False,
                "no_new_shorts": False,
            }
        ]
    )

    group_constraints = _make_placeholder_df(
        [
            {"group": "Sector", "rule": "±5% vs benchmark", "status": "placeholder"},
            {"group": "Style", "rule": "Target 0 exposure", "status": "placeholder"},
        ]
    )

    portfolio_constraints = _dict_to_table(
        {
            "Leverage max": f"{ctx.portfolio.leverage_max:.1f}x",
            "Gross exposure max": f"{ctx.portfolio.gross_exposure_max_pct_nav:.1f}% NAV",
            "Net exposure target": f"{ctx.portfolio.net_exposure_target_pct_nav:.1f}% ±{ctx.portfolio.net_exposure_band_pct_nav:.1f}%",
            "Turnover cap": f"{ctx.effective_turnover_cap_pct_gross:.1f}% gross",
        },
        title_key="Constraint",
    )

    with constraint_output:
        accordion = widgets.Accordion(
            children=[
                _make_grid(instrument_constraints, height="180px"),
                _make_grid(group_constraints, height="140px"),
                _make_grid(portfolio_constraints, height="150px"),
            ]
        )
        accordion.set_title(0, "Instrument-level")
        accordion.set_title(1, "Group-level")
        accordion.set_title(2, "Portfolio-level")
        display(accordion)


# ---------------------------------------------------------------------------
# Optimization settings tab content
# ---------------------------------------------------------------------------


def _populate_settings() -> None:
    settings_output.clear_output()
    try:
        ctx, _, _ = app_state.ensure_data_ready()
    except RuntimeError as exc:
        with settings_output:
            print(str(exc))
        return

    rt = ctx.run_type
    portfolio = ctx.portfolio

    toggles = widgets.VBox(
        [
            widgets.Checkbox(value=rt.apply_alpha, description="Apply alpha"),
            widgets.Checkbox(value=rt.allow_etf_create_redeem, description="Allow ETF create/redeem"),
            widgets.Checkbox(value=rt.allow_efp, description="Allow EFP"),
            widgets.Checkbox(value=rt.allow_new_shorts and portfolio.allow_new_shorts, description="Allow new shorts"),
        ]
    )

    sliders = widgets.VBox(
        [
            widgets.FloatSlider(
                value=ctx.effective_risk_aversion,
                min=0.1,
                max=max(3.0, ctx.effective_risk_aversion * 2),
                description="Effective RA",
            ),
            widgets.FloatSlider(
                value=ctx.effective_turnover_cap_pct_gross,
                min=0.0,
                max=max(100.0, ctx.effective_turnover_cap_pct_gross * 2),
                description="TO cap (%)",
            ),
        ]
    )

    summary = _dict_to_table(
        {
            "Default run type": app_state.run_config.global_settings.default_run_type,
            "Cost model": "default",
            "Synthetics": "ETF baskets + futures enabled",
            "Margin": "Haircut-aware (stub)",
        }
    )

    with settings_output:
        display(widgets.HBox([toggles, sliders]))
        display(_make_grid(summary, height="180px"))


# ---------------------------------------------------------------------------
# Results tab content
# ---------------------------------------------------------------------------


def _on_run_optimizer_clicked(_) -> None:
    results_output.clear_output()
    trade_output.clear_output()
    mcp_output.clear_output()

    try:
        ctx, snapshot, intraday = app_state.ensure_data_ready()
    except RuntimeError as exc:
        with results_output:
            print(str(exc))
        return

    df_instr = snapshot["instruments"]
    df_pos = intraday["positions"]
    df_cash = intraday["cash"]

    trades = _make_placeholder_df(
        [
            {
                "symbol": row["instrument_id"],
                "side": "buy" if idx % 2 == 0 else "sell",
                "shares": 1_000 + idx * 100,
            }
            for idx, row in enumerate(df_instr.to_dicts())
        ]
    )

    with results_output:
        print("=== Optimization Run (Stub) ===")
        print(f"Portfolio: {ctx.portfolio.portfolio_id} ({ctx.portfolio.name})")
        print(f"Run type:  {ctx.run_type.name} – {ctx.run_type.description}")
        print(f"As-of:     {ctx.as_of}")
        print(f"Cash Δ:    {ctx.cash_injection:,.2f}")
        print("")
        print("Key settings:")
        print(f"  Apply alpha:           {ctx.run_type.apply_alpha}")
        print(f"  Allow ETF create/red:  {ctx.run_type.allow_etf_create_redeem}")
        print(f"  Allow EFP:             {ctx.run_type.allow_efp}")
        print(f"  Allow new shorts:      {ctx.run_type.allow_new_shorts and ctx.portfolio.allow_new_shorts}")
        print(f"  Leverage max:          {ctx.portfolio.leverage_max:.2f}x")
        print(f"  Gross exposure max:    {ctx.portfolio.gross_exposure_max_pct_nav:.1f}% of NAV")
        print(f"  Net exposure target:   {ctx.portfolio.net_exposure_target_pct_nav:.1f}% of NAV")
        print(f"  Net exposure band:     ±{ctx.portfolio.net_exposure_band_pct_nav:.1f}%")
        print(f"  Cash buffer target:    {ctx.portfolio.cash_buffer_pct:.1%}")
        print(f"  Effective RA:          {ctx.effective_risk_aversion:.3f}")
        print(f"  Effective TO cap:      {ctx.effective_turnover_cap_pct_gross:.1f}% of gross")
        print("")
        print("Universe & positions summary:")
        print(f"  Instruments in universe: {df_instr.height}")
        print(f"  Positions rows:          {df_pos.height}")
        print(
            f"  Cash:                    {float(df_cash.select('cash')[0, 0]):,.2f} "
            f"{df_cash.select('currency')[0, 0]}"
        )

    with trade_output:
        display(_make_grid(trades, height="180px"))

    with mcp_output:
        mcp_payload = _dict_to_table(
            {
                "data_qc": "pending",
                "constraint_status": "pending",
                "risk_hotspots": "pending",
                "parameter_suggestions": "pending",
            },
            title_key="MCP signal",
            value_key="Status",
        )
        display(_make_grid(mcp_payload, height="160px"))


run_optimizer_button.on_click(_on_run_optimizer_clicked)


# ---------------------------------------------------------------------------
# Tab assembly
# ---------------------------------------------------------------------------


def _build_config_tab() -> widgets.Widget:
    controls = widgets.VBox(
        [
            widgets.HBox([config_path_input, reload_config_button]),
            portfolio_dropdown,
            run_type_dropdown,
            widgets.HBox([asof_picker, cash_injection_input, risk_aversion_override]),
            apply_context_button,
        ]
    )
    return widgets.VBox([
        widgets.HTML("<b>Run context builder</b>"),
        controls,
        widgets.HTML("<b>Context summary</b>"),
        context_summary_output,
    ])


def _build_data_tab() -> widgets.Widget:
    accordions = widgets.Accordion(children=[universe_output, risk_output, costs_output, restrictions_output])
    accordions.set_title(0, "Universe")
    accordions.set_title(1, "Risk model summary")
    accordions.set_title(2, "Cost parameters")
    accordions.set_title(3, "Restrictions")

    return widgets.VBox(
        [
            widgets.HBox([load_snapshot_button]),
            widgets.HTML("<b>Data quality</b>"),
            data_log_output,
            widgets.HTML("<b>Snapshot views</b>"),
            accordions,
        ]
    )


def _build_constraints_tab() -> widgets.Widget:
    refresh_btn = widgets.Button(description="Refresh constraints", icon="refresh", button_style="info")
    refresh_btn.on_click(lambda _: _populate_constraints())
    return widgets.VBox([
        refresh_btn,
        constraint_output,
    ])


def _build_settings_tab() -> widgets.Widget:
    refresh_btn = widgets.Button(description="Refresh settings", icon="refresh", button_style="info")
    refresh_btn.on_click(lambda _: _populate_settings())
    return widgets.VBox([
        refresh_btn,
        settings_output,
    ])


def _build_results_tab() -> widgets.Widget:
    accordions = widgets.Accordion(children=[results_output, trade_output, mcp_output])
    accordions.set_title(0, "Run summary")
    accordions.set_title(1, "Trades (stub)")
    accordions.set_title(2, "AI logging (MCP)")

    return widgets.VBox(
        [
            run_optimizer_button,
            accordions,
        ]
    )


tabs = widgets.Tab(children=[
    _build_config_tab(),
    _build_data_tab(),
    _build_constraints_tab(),
    _build_settings_tab(),
    _build_results_tab(),
])
tabs.set_title(0, "Config")
tabs.set_title(1, "Data & Universe")
tabs.set_title(2, "Constraints")
tabs.set_title(3, "Opt Settings")
tabs.set_title(4, "Run & Results")

display(tabs)

