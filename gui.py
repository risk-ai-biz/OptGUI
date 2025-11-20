# Cell 1: imports & configuration load

from dataclasses import dataclass, field
from pathlib import Path
import datetime as dt

import polars as pl
from IPython.display import display

import ipywidgets as widgets
from ipydatagrid import DataGrid

from RunConfig import RunConfig, RunContext
from example_data import compute_stub_trades, load_intraday_state, load_static_snapshot

# Load configuration
CONFIG_PATH = Path("config/example_config.yaml")
run_config = RunConfig.from_yaml(CONFIG_PATH)

print(f"Loaded config: {len(run_config.portfolios)} portfolios, {len(run_config.run_types)} run types.")

DATA_ROOT = run_config.global_settings.data_root


@dataclass
class AppState:
    """Hold the GUI's dynamic state for a run."""

    config: RunConfig
    run_context: RunContext | None = None
    snapshot: dict[str, pl.DataFrame] | None = None
    intraday: dict[str, pl.DataFrame] | None = None
    trades: pl.DataFrame | None = None
    messages: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.messages.append(msg)

    def clear(self) -> None:
        self.run_context = None
        self.snapshot = None
        self.intraday = None
        self.trades = None
        self.messages.clear()


app_state = AppState(config=run_config)

# Cell 3: core widgets for run configuration

# Portfolio dropdown
portfolio_dropdown = widgets.Dropdown(
    options=[(p.name, pid) for pid, p in run_config.portfolios.items()],
    description="Portfolio:",
    layout=widgets.Layout(width="300px"),
)

# Run type dropdown
run_type_dropdown = widgets.Dropdown(
    options=[(rt.name, rt.name) for rt in run_config.run_types.values()],
    description="Run type:",
    layout=widgets.Layout(width="300px"),
)

# As-of date picker
asof_picker = widgets.DatePicker(
    description="As-of date:",
    disabled=False,
)

# Cash injection input
cash_injection_input = widgets.FloatText(
    value=0.0,
    description="Cash Δ:",
    layout=widgets.Layout(width="250px"),
)

# Risk aversion override
risk_aversion_override = widgets.FloatText(
    value=None,
    description="RA override:",
    placeholder="Optional",
    layout=widgets.Layout(width="250px"),
)

# Buttons
load_snapshot_button = widgets.Button(
    description="Load snapshot",
    button_style="info",
    icon="download",
)

run_optimizer_button = widgets.Button(
    description="Run optimizer (stub)",
    button_style="success",
    icon="play",
)

# Output areas
log_output = widgets.Output(layout={"border": "1px solid #ccc"})
instr_output = widgets.Output()
results_output = widgets.Output()

controls_box = widgets.VBox(
    [
        portfolio_dropdown,
        run_type_dropdown,
        asof_picker,
        cash_injection_input,
        risk_aversion_override,
        widgets.HBox([load_snapshot_button, run_optimizer_button]),
    ]
)

display(controls_box)
display(widgets.Label("Log / Messages:"))
display(log_output)
display(widgets.Label("Universe snapshot:"))
display(instr_output)
display(widgets.Label("Optimization result (placeholder):"))
display(results_output)

# Cell 4: wiring up the 'Load snapshot' button to show universe & validation

current_snapshot: dict[str, pl.DataFrame] | None = None
current_intraday: dict[str, pl.DataFrame] | None = None
current_run_context = None


def _get_selected_asof() -> dt.date:
    if asof_picker.value is not None:
        return asof_picker.value
    # fallback: use config default lag from today
    today = dt.date.today()
    lag = run_config.global_settings.default_asof_days_lag
    return today - dt.timedelta(days=lag)


@load_snapshot_button.on_click
def on_load_snapshot_clicked(btn):
    global current_snapshot, current_intraday, current_run_context
    instr_output.clear_output()
    log_output.clear_output()

    portfolio_id = portfolio_dropdown.value
    run_type_name = run_type_dropdown.value
    as_of = _get_selected_asof()
    cash_injection = float(cash_injection_input.value or 0.0)

    # risk aversion override (None if empty)
    ra_override_val = risk_aversion_override.value
    ra_override = float(ra_override_val) if ra_override_val not in (None, "") else None

    with log_output:
        app_state.clear()
        print(
            f"Loading snapshot for portfolio={portfolio_id}, run_type={run_type_name}, as_of={as_of}..."
        )
        current_run_context = run_config.make_run_context(
            portfolio_id=portfolio_id,
            run_type_name=run_type_name,
            as_of=as_of,
            cash_injection=cash_injection,
            custom_risk_aversion=ra_override,
        )
        app_state.run_context = current_run_context

        print("Resolved RunContext:")
        print(f"  effective risk aversion: {current_run_context.effective_risk_aversion:.3f}")
        print(
            f"  effective turnover cap:  {current_run_context.effective_turnover_cap_pct_gross:.1f}% of gross"
        )

        # Load SoD + intraday data
        current_snapshot = load_static_snapshot(as_of)
        current_intraday = load_intraday_state(portfolio_id, as_of)
        app_state.snapshot = current_snapshot
        app_state.intraday = current_intraday

        # Basic validation examples
        df_instr = current_snapshot["instruments"]
        df_risk = current_snapshot["risk_model_summary"]

        # Check missing risk entries
        df_merged = df_instr.join(df_risk, on="instrument_id", how="left")
        missing_risk = df_merged.filter(pl.col("has_factor_loadings") != True)

        print(f"Universe size: {df_instr.height} instruments.")
        if missing_risk.height > 0:
            print(
                f"WARNING: {missing_risk.height} instruments missing risk model; they may be dropped or treated conservatively."
            )

    # Show instruments in grid
    with instr_output:
        # For convenience, join a few quick columns
        df_view = (
            df_instr.join(df_risk, on="instrument_id", how="left")
            .join(current_snapshot["cost_params"], on="instrument_id", how="left")
        )
        display(DataGrid(df_view.to_pandas(), layout={"height": "300px"}))
        display(widgets.Label("Constraints (illustrative):"))
        display(
            DataGrid(
                current_snapshot["constraints"].to_pandas(),
                layout={"height": "140px"},
                base_row_size=24,
            )
        )


# Cell 5: 'Run optimizer' stub – inspect run context & data

@run_optimizer_button.on_click
def on_run_optimizer_clicked(btn):
    results_output.clear_output()
    log_output.clear_output(wait=True)

    if current_run_context is None or current_snapshot is None or current_intraday is None:
        with log_output:
            print("Please load snapshot first.")
        return

    ctx = current_run_context
    snapshot = current_snapshot
    intraday = current_intraday

    df_instr = snapshot["instruments"]
    df_pos = intraday["positions"]
    df_cash = intraday["cash"]

    trades = compute_stub_trades(snapshot, intraday)
    app_state.trades = trades

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
        print(
            f"  Allow new shorts:      {ctx.run_type.allow_new_shorts and ctx.portfolio.allow_new_shorts}"
        )
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
            f"  Cash:                    {float(df_cash.select('cash')[0, 0]):,.2f} {df_cash.select('currency')[0, 0]}"
        )

        print("")
        print("Suggested trades (stubbed):")
        display(
            DataGrid(
                trades.to_pandas(),
                layout={"height": "250px"},
                base_row_size=24,
            )
        )
        print("""Note: trades are rule-based placeholders. Replace with optimizer output.""")

  
