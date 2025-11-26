
"""
High-level API and diagnostics for the notional-based portfolio optimizer.

This module defines:
  - Dataclasses for inputs, configs, results, diagnostics
  - A convenience solve_portfolio(...) wrapper that:
      * builds the MOSEK Fusion model via portfolio_model.build_notional_portfolio_auto_scale
      * solves it
      * extracts key diagnostics
      * returns a serialisable OptimizationRun object

The idea is to have a compact JSON-ready snapshot that you can hand to an LLM
(or log to disk) to understand failures and suggest relaxations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

import numpy as np
from mosek.fusion import Model, SolutionStatus
import mosek.fusion.pythonic  # noqa: F401

from portfolio_model import build_notional_portfolio_auto_scale


# ---------------------------------------------------------------------------
# Dataclasses for inputs & configs
# ---------------------------------------------------------------------------

@dataclass
class RiskModelSpec:
    n_assets: int
    n_factors: int
    factor_cov_shape: List[int]
    resid_var_stats: Dict[str, float]


@dataclass
class InstrumentUniverse:
    asset_ids: List[str]              # length N
    synthetic_ids: List[str]          # length S
    underlying_for_synth: Dict[str, List[str]]  # map synthetic_id -> list of underlying ids


@dataclass
class ConstraintConfig:
    net_target: Optional[float] = None
    net_lb: Optional[float] = None
    net_ub: Optional[float] = None

    gross_max: Optional[float] = None
    sigma_max: Optional[float] = None

    integer_contracts: bool = True


@dataclass
class BoundConfig:
    pos_lb_stock: Optional[List[float]] = None
    pos_ub_stock: Optional[List[float]] = None
    contract_lb: Optional[List[float]] = None
    contract_ub: Optional[List[float]] = None
    trade_abs_max_stock: Optional[List[float]] = None
    trade_abs_max_syn: Optional[List[float]] = None


@dataclass
class GroupConstraintConfig:
    group_names: List[str]                  # length G
    group_matrix: List[List[float]]         # shape (G, N) in row-major form
    net_lb: Optional[List[float]] = None    # length G or None
    net_ub: Optional[List[float]] = None    # length G or None


@dataclass
class FactorConstraintConfig:
    factor_names: List[str]                 # length K
    exposure_lb: Optional[List[float]] = None  # length K or None
    exposure_ub: Optional[List[float]] = None  # length K or None


@dataclass
class TurnoverConstraintConfig:
    turnover_max_stock: Optional[float] = None
    turnover_max_syn: Optional[float] = None


@dataclass
class CostConfig:
    has_linear_exec: bool
    has_impact: bool
    has_borrow: bool
    has_funding: bool
    has_carry: bool

    max_exec_cost_per_unit: float = 0.0
    max_borrow_rate: float = 0.0
    max_funding_rate: float = 0.0


@dataclass
class ScalingInfo:
    scale_factor: float
    max_notional_before: float
    target_scale: float


@dataclass
class SolverConfig:
    solver_name: str = "mosek"
    time_limit_sec: Optional[float] = None
    mip_gap: Optional[float] = None
    num_threads: Optional[int] = None
    log_to_stdout: bool = False


@dataclass
class OptimizationInputs:
    universe: InstrumentUniverse
    risk_model: RiskModelSpec
    constraints: ConstraintConfig
    bounds: BoundConfig
    costs: CostConfig
    scaling: ScalingInfo
    group_constraints: Optional[GroupConstraintConfig] = None
    factor_constraints: Optional[FactorConstraintConfig] = None
    turnover_constraints: Optional[TurnoverConstraintConfig] = None


# ---------------------------------------------------------------------------
# Dataclasses for results & diagnostics
# ---------------------------------------------------------------------------

@dataclass
class SolutionSummary:
    status: str
    objective_value: Optional[float]
    risk_sigma: Optional[float]
    net_exposure: Optional[float]
    gross_exposure: Optional[float]
    long_notional: Optional[float] = None
    short_notional: Optional[float] = None


@dataclass
class PositionRecord:
    asset_id: str
    notional: float
    at_lower_bound: bool
    at_upper_bound: bool


@dataclass
class PositionSummary:
    top_long_positions: List[PositionRecord] = field(default_factory=list)
    top_short_positions: List[PositionRecord] = field(default_factory=list)
    num_at_upper_bound: int = 0
    num_at_lower_bound: int = 0


@dataclass
class ConstraintDiagnostics:
    named_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    group_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    factor_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    turnover_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    suspected_conflicting_constraints: List[str] = field(default_factory=list)


@dataclass
class SolverDiagnostics:
    problem_status: str
    primal_status: str
    dual_status: str
    objective_value: Optional[float] = None


@dataclass
class OptimizationResult:
    solution: SolutionSummary
    positions: Optional[PositionSummary] = None
    constraints: Optional[ConstraintDiagnostics] = None
    solver: Optional[SolverDiagnostics] = None


@dataclass
class OptimizationRun:
    inputs: OptimizationInputs
    solver_config: SolverConfig
    result: OptimizationResult

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# High-level solve wrapper
# ---------------------------------------------------------------------------

def solve_portfolio(
    # Identification / metadata
    asset_ids: List[str],
    synthetic_ids: List[str],
    underlying_for_synth: Dict[str, List[str]],

    # Risk-model data
    B_stock: np.ndarray,
    Sigma_F: np.ndarray,
    theta_stock: np.ndarray,
    S_contract: np.ndarray,

    # Alpha & initial positions
    alpha_stock: np.ndarray,
    alpha_syn: np.ndarray,
    x0_stock: np.ndarray,
    q0_syn: np.ndarray,

    # Constraints & bounds
    constraint_cfg: ConstraintConfig,
    bound_cfg: BoundConfig,

    # Structural constraints (optional)
    group_cfg: Optional[GroupConstraintConfig] = None,
    factor_cfg: Optional[FactorConstraintConfig] = None,
    turnover_cfg: Optional[TurnoverConstraintConfig] = None,

    # Costs
    lin_exec_stock: Optional[np.ndarray] = None,
    lin_exec_syn: Optional[np.ndarray] = None,
    impact_stock: Optional[np.ndarray] = None,
    impact_syn: Optional[np.ndarray] = None,
    borrow_stock: Optional[np.ndarray] = None,
    funding_stock: Optional[np.ndarray] = None,
    carry_syn: Optional[np.ndarray] = None,

    # Solver config
    solver_cfg: Optional[SolverConfig] = None,
    target_scale: float = 1e3,
) -> OptimizationRun:
    """Build, solve, and summarise a portfolio optimization run.

    Returns an OptimizationRun object that is easy to serialise (e.g. via JSON).
    """
    if solver_cfg is None:
        solver_cfg = SolverConfig()

    N, K = B_stock.shape
    S = len(synthetic_ids)
    assert len(asset_ids) == N, "asset_ids length must match B_stock rows"
    assert S_contract.shape == (N, S), "S_contract shape mismatch"
    assert len(alpha_stock) == N and len(alpha_syn) == S

    # Scale detection input
    max_notional_before = float(np.max(np.abs(x0_stock))) if x0_stock.size > 0 else 0.0

    # Prepare group / factor / turnover data for builder
    group_matrix = None
    group_lb = None
    group_ub = None
    if group_cfg is not None:
        group_matrix = np.asarray(group_cfg.group_matrix, dtype=float)
        group_lb = group_cfg.net_lb
        group_ub = group_cfg.net_ub

    factor_lb = None
    factor_ub = None
    if factor_cfg is not None:
        factor_lb = factor_cfg.exposure_lb
        factor_ub = factor_cfg.exposure_ub

    turnover_max_stock = None
    turnover_max_syn = None
    if turnover_cfg is not None:
        turnover_max_stock = turnover_cfg.turnover_max_stock
        turnover_max_syn = turnover_cfg.turnover_max_syn

    # Build and scale model
    M, x_var, q_var, sigma_var, scale = build_notional_portfolio_auto_scale(
        alpha_stock=alpha_stock,
        alpha_syn=alpha_syn,
        x0_stock=x0_stock,
        q0_syn=q0_syn,
        B_stock=B_stock,
        Sigma_F=Sigma_F,
        theta_stock=theta_stock,
        S_contract=S_contract,
        pos_lb_stock=bound_cfg.pos_lb_stock,
        pos_ub_stock=bound_cfg.pos_ub_stock,
        contract_lb=bound_cfg.contract_lb,
        contract_ub=bound_cfg.contract_ub,
        net_target=constraint_cfg.net_target,
        net_lb=constraint_cfg.net_lb,
        net_ub=constraint_cfg.net_ub,
        gross_max=constraint_cfg.gross_max,
        risk_aversion=1.0,  # risk_aversion is inside alpha / config in this wrapper
        sigma_max=constraint_cfg.sigma_max,
        lin_exec_stock=lin_exec_stock,
        lin_exec_syn=lin_exec_syn,
        impact_stock=impact_stock,
        impact_syn=impact_syn,
        borrow_stock=borrow_stock,
        funding_stock=funding_stock,
        carry_syn=carry_syn,
        trade_abs_max_stock=bound_cfg.trade_abs_max_stock,
        trade_abs_max_syn=bound_cfg.trade_abs_max_syn,
        turnover_max_stock=turnover_max_stock,
        turnover_max_syn=turnover_max_syn,
        group_matrix_h=group_matrix,
        group_net_lb=group_lb,
        group_net_ub=group_ub,
        factor_exposure_lb=factor_lb,
        factor_exposure_ub=factor_ub,
        integer_contracts=constraint_cfg.integer_contracts,
        target_scale=target_scale,
        model_name="notional_portfolio_scaled",
    )

    # Basic solver params
    if solver_cfg.time_limit_sec is not None:
        M.setSolverParam("mioMaxTime", float(solver_cfg.time_limit_sec))
    if solver_cfg.mip_gap is not None:
        M.setSolverParam("mioTolRelGap", float(solver_cfg.mip_gap))
    if solver_cfg.num_threads is not None:
        M.setSolverParam("numThreads", int(solver_cfg.num_threads))

    # Optional logging
    if solver_cfg.log_to_stdout:
        import sys
        M.setLogHandler(sys.stdout)

    # Solve
    try:
        M.solve()
    except Exception as e:  # noqa
        # Hard failure before solution; build minimal diagnostics
        rm_spec = RiskModelSpec(
            n_assets=N,
            n_factors=K,
            factor_cov_shape=list(Sigma_F.shape),
            resid_var_stats={
                "min": float(np.min(theta_stock)),
                "max": float(np.max(theta_stock)),
                "median": float(np.median(theta_stock)),
            },
        )
        universe = InstrumentUniverse(
            asset_ids=asset_ids,
            synthetic_ids=synthetic_ids,
            underlying_for_synth=underlying_for_synth,
        )
        scaling = ScalingInfo(
            scale_factor=scale,
            max_notional_before=max_notional_before,
            target_scale=target_scale,
        )
        costs_cfg = CostConfig(
            has_linear_exec=lin_exec_stock is not None or lin_exec_syn is not None,
            has_impact=impact_stock is not None or impact_syn is not None,
            has_borrow=borrow_stock is not None,
            has_funding=funding_stock is not None,
            has_carry=carry_syn is not None,
        )
        inputs = OptimizationInputs(
            universe=universe,
            risk_model=rm_spec,
            constraints=constraint_cfg,
            bounds=bound_cfg,
            costs=costs_cfg,
            scaling=scaling,
            group_constraints=group_cfg,
            factor_constraints=factor_cfg,
            turnover_constraints=turnover_cfg,
        )
        sol_summary = SolutionSummary(
            status="ERROR_DURING_SOLVE",
            objective_value=None,
            risk_sigma=None,
            net_exposure=None,
            gross_exposure=None,
        )
        solver_diag = SolverDiagnostics(
            problem_status="EXCEPTION",
            primal_status=str(e),
            dual_status="UNKNOWN",
            objective_value=None,
        )
        result = OptimizationResult(
            solution=sol_summary,
            positions=None,
            constraints=None,
            solver=solver_diag,
        )
        return OptimizationRun(
            inputs=inputs,
            solver_config=solver_cfg,
            result=result,
        )

    # Extract solver statuses
    prob_status = M.getProblemStatus()
    prim_status = M.getPrimalSolutionStatus()
    dual_status = M.getDualSolutionStatus()

    prob_status_str = str(prob_status)
    prim_status_str = str(prim_status)
    dual_status_str = str(dual_status)

    # Default solution summary
    objective_value = None
    risk_sigma = None
    net_exposure = None
    gross_exposure = None
    long_notional = None
    short_notional = None

    x_real = None
    q_val = None
    h_real = None

    # Only try to read solution if primal status is not Unknown
    if prim_status != SolutionStatus.Unknown:
        objective_value = float(M.primalObjValue())

        x_scaled = np.array(x_var.level())
        q_val    = np.array(q_var.level())
        sigma_int = float(sigma_var.level()[0])

        x_real = x_scaled * scale
        sigma_real = sigma_int * scale

        # Effective underlyings in real units
        h_real = x_real + S_contract @ q_val

        risk_sigma = sigma_real
        net_exposure = float(h_real.sum())
        gross_exposure = float(np.abs(h_real).sum())
        long_notional = float(np.clip(h_real, 0, np.inf).sum())
        short_notional = float(np.clip(h_real, -np.inf, 0).sum())

    # Build solution summary
    sol_status = "UNKNOWN"
    if prim_status == SolutionStatus.Optimal:
        sol_status = "OPTIMAL"
    elif prim_status == SolutionStatus.PrimFeas:
        sol_status = "PRIM_FEASIBLE"
    elif prim_status == SolutionStatus.DualFeas:
        sol_status = "DUAL_FEASIBLE"
    elif prim_status == SolutionStatus.PrimInfeas:
        sol_status = "PRIM_INFEASIBLE"
    elif prim_status == SolutionStatus.DualInfeas:
        sol_status = "DUAL_INFEASIBLE"

    sol_summary = SolutionSummary(
        status=sol_status,
        objective_value=objective_value,
        risk_sigma=risk_sigma,
        net_exposure=net_exposure,
        gross_exposure=gross_exposure,
        long_notional=long_notional,
        short_notional=short_notional,
    )

    # Position summary (top longs/shorts)
    pos_summary: Optional[PositionSummary] = None
    if prim_status != SolutionStatus.Unknown and x_real is not None:
        pos_lb_stock = np.array(bound_cfg.pos_lb_stock) if bound_cfg.pos_lb_stock is not None else None
        pos_ub_stock = np.array(bound_cfg.pos_ub_stock) if bound_cfg.pos_ub_stock is not None else None

        records: List[PositionRecord] = []
        num_at_lb = 0
        num_at_ub = 0

        for j in range(N):
            at_lb = False
            at_ub = False
            if pos_lb_stock is not None and np.isfinite(pos_lb_stock[j]):
                if abs(x_real[j] - pos_lb_stock[j]) <= 1e-6 * (1.0 + abs(pos_lb_stock[j])):
                    at_lb = True
                    num_at_lb += 1
            if pos_ub_stock is not None and np.isfinite(pos_ub_stock[j]):
                if abs(x_real[j] - pos_ub_stock[j]) <= 1e-6 * (1.0 + abs(pos_ub_stock[j])):
                    at_ub = True
                    num_at_ub += 1

            records.append(
                PositionRecord(
                    asset_id=asset_ids[j],
                    notional=float(x_real[j]),
                    at_lower_bound=at_lb,
                    at_upper_bound=at_ub,
                )
            )

        # Sort by notional
        longs = sorted([r for r in records if r.notional > 0], key=lambda r: -r.notional)[:10]
        shorts = sorted([r for r in records if r.notional < 0], key=lambda r: r.notional)[:10]

        pos_summary = PositionSummary(
            top_long_positions=longs,
            top_short_positions=shorts,
            num_at_upper_bound=num_at_ub,
            num_at_lower_bound=num_at_lb,
        )

    # Constraint diagnostics
    cons_diag = ConstraintDiagnostics(named_slacks={})

    # Net exposure slack
    if net_exposure is not None:
        if constraint_cfg.net_target is not None:
            net_slack = float(net_exposure - constraint_cfg.net_target)
            cons_diag.named_slacks["net_eq"] = {
                "lhs": net_exposure,
                "rhs": constraint_cfg.net_target,
                "slack": net_slack,
            }
        else:
            if constraint_cfg.net_lb is not None or constraint_cfg.net_ub is not None:
                lb = -np.inf if constraint_cfg.net_lb is None else constraint_cfg.net_lb
                ub =  np.inf if constraint_cfg.net_ub is None else constraint_cfg.net_ub
                slack_lb = net_exposure - lb
                slack_ub = ub - net_exposure
                cons_diag.named_slacks["net_range"] = {
                    "lhs": net_exposure,
                    "lb": lb,
                    "ub": ub,
                    "slack_lb": slack_lb,
                    "slack_ub": slack_ub,
                }

    # Gross slack
    if gross_exposure is not None and constraint_cfg.gross_max is not None:
        slack = constraint_cfg.gross_max - gross_exposure
        cons_diag.named_slacks["gross_lim"] = {
            "lhs": gross_exposure,
            "rhs": constraint_cfg.gross_max,
            "slack": slack,
        }

    # Risk slack
    if risk_sigma is not None and constraint_cfg.sigma_max is not None:
        slack = constraint_cfg.sigma_max - risk_sigma
        cons_diag.named_slacks["sigma_max"] = {
            "lhs": risk_sigma,
            "rhs": constraint_cfg.sigma_max,
            "slack": slack,
        }

    # Group exposure slacks
    if h_real is not None and group_cfg is not None:
        G = np.asarray(group_cfg.group_matrix, dtype=float)  # (G, N)
        g_names = group_cfg.group_names
        g_exp = G @ h_real
        lb = np.asarray(group_cfg.net_lb, dtype=float) if group_cfg.net_lb is not None else None
        ub = np.asarray(group_cfg.net_ub, dtype=float) if group_cfg.net_ub is not None else None
        for i, name in enumerate(g_names):
            info: Dict[str, float] = {"exposure": float(g_exp[i])}
            if lb is not None:
                info["lb"] = float(lb[i])
                info["slack_lb"] = float(g_exp[i] - lb[i])
            if ub is not None:
                info["ub"] = float(ub[i])
                info["slack_ub"] = float(ub[i] - g_exp[i])
            cons_diag.group_slacks[name] = info

    # Factor exposure slacks
    if h_real is not None and factor_cfg is not None:
        f_names = factor_cfg.factor_names
        exposures = B_stock.T @ h_real  # (K,)
        lb = np.asarray(factor_cfg.exposure_lb, dtype=float) if factor_cfg.exposure_lb is not None else None
        ub = np.asarray(factor_cfg.exposure_ub, dtype=float) if factor_cfg.exposure_ub is not None else None
        for k, name in enumerate(f_names):
            info: Dict[str, float] = {"exposure": float(exposures[k])}
            if lb is not None:
                info["lb"] = float(lb[k])
                info["slack_lb"] = float(exposures[k] - lb[k])
            if ub is not None:
                info["ub"] = float(ub[k])
                info["slack_ub"] = float(ub[k] - exposures[k])
            cons_diag.factor_slacks[name] = info

    # Turnover slacks
    if x_real is not None and q_val is not None and turnover_cfg is not None:
        dx_real = x_real - x0_stock
        dq_real = q_val - q0_syn

        total_turnover_x = float(np.abs(dx_real).sum())
        total_turnover_q = float(np.abs(dq_real).sum())

        if turnover_cfg.turnover_max_stock is not None:
            slack = turnover_cfg.turnover_max_stock - total_turnover_x
            cons_diag.turnover_slacks["stock"] = {
                "total_turnover": total_turnover_x,
                "limit": turnover_cfg.turnover_max_stock,
                "slack": slack,
            }
        if turnover_cfg.turnover_max_syn is not None:
            slack = turnover_cfg.turnover_max_syn - total_turnover_q
            cons_diag.turnover_slacks["synthetic"] = {
                "total_turnover": total_turnover_q,
                "limit": turnover_cfg.turnover_max_syn,
                "slack": slack,
            }

    # Simple heuristic: mark constraints with negative slack as suspected conflicting
    for cname, info in cons_diag.named_slacks.items():
        if "slack" in info and info["slack"] < 0:
            cons_diag.suspected_conflicting_constraints.append(cname)
        if "slack_lb" in info and info["slack_lb"] < 0:
            cons_diag.suspected_conflicting_constraints.append(cname + ":lb")
        if "slack_ub" in info and info["slack_ub"] < 0:
            cons_diag.suspected_conflicting_constraints.append(cname + ":ub")

    for gname, info in cons_diag.group_slacks.items():
        if "slack_lb" in info and info["slack_lb"] < 0:
            cons_diag.suspected_conflicting_constraints.append(f"group:{gname}:lb")
        if "slack_ub" in info and info["slack_ub"] < 0:
            cons_diag.suspected_conflicting_constraints.append(f"group:{gname}:ub")

    for fname, info in cons_diag.factor_slacks.items():
        if "slack_lb" in info and info["slack_lb"] < 0:
            cons_diag.suspected_conflicting_constraints.append(f"factor:{fname}:lb")
        if "slack_ub" in info and info["slack_ub"] < 0:
            cons_diag.suspected_conflicting_constraints.append(f"factor:{fname}:ub")

    for tname, info in cons_diag.turnover_slacks.items():
        if "slack" in info and info["slack"] < 0:
            cons_diag.suspected_conflicting_constraints.append(f"turnover:{tname}")

    # Solver diagnostics
    solver_diag = SolverDiagnostics(
        problem_status=prob_status_str,
        primal_status=prim_status_str,
        dual_status=dual_status_str,
        objective_value=objective_value,
    )

    # Build inputs snapshot for serialisation
    resid_stats = {
        "min": float(np.min(theta_stock)),
        "max": float(np.max(theta_stock)),
        "median": float(np.median(theta_stock)),
    }
    rm_spec = RiskModelSpec(
        n_assets=N,
        n_factors=K,
        factor_cov_shape=list(Sigma_F.shape),
        resid_var_stats=resid_stats,
    )
    universe = InstrumentUniverse(
        asset_ids=asset_ids,
        synthetic_ids=synthetic_ids,
        underlying_for_synth=underlying_for_synth,
    )
    scaling = ScalingInfo(
        scale_factor=scale,
        max_notional_before=max_notional_before,
        target_scale=target_scale,
    )
    costs_cfg = CostConfig(
        has_linear_exec=lin_exec_stock is not None or lin_exec_syn is not None,
        has_impact=impact_stock is not None or impact_syn is not None,
        has_borrow=borrow_stock is not None,
        has_funding=funding_stock is not None,
        has_carry=carry_syn is not None,
        max_exec_cost_per_unit=float(np.max(lin_exec_stock)) if lin_exec_stock is not None else 0.0,
        max_borrow_rate=float(np.max(borrow_stock)) if borrow_stock is not None else 0.0,
        max_funding_rate=float(np.max(funding_stock)) if funding_stock is not None else 0.0,
    )
    inputs = OptimizationInputs(
        universe=universe,
        risk_model=rm_spec,
        constraints=constraint_cfg,
        bounds=bound_cfg,
        costs=costs_cfg,
        scaling=scaling,
        group_constraints=group_cfg,
        factor_constraints=factor_cfg,
        turnover_constraints=turnover_cfg,
    )

    result = OptimizationResult(
        solution=sol_summary,
        positions=pos_summary,
        constraints=cons_diag,
        solver=solver_diag,
    )

    return OptimizationRun(
        inputs=inputs,
        solver_config=solver_cfg,
        result=result,
    )
