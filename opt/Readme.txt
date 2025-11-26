Alright, this is now a pretty complete little optimization “stack”.

Here’s what you’ve got, with the extra structure constraints and richer diagnostics wired in, and ready to serialize and hand to an LLM.


---

1. Downloadable modules

Core model + scaling

portfolio_model.py


High-level API + diagnostics + serialization

portfolio_api.py



Drop both into your project, and pip install mosek / have Fusion on the path.


---

2. What the core model now supports (portfolio_model.py)

Main builders

build_notional_portfolio_core(...)

build_notional_portfolio_auto_scale(...)  -> (M, x, q, sigma, scale)


Key features (all in notional + contract space):

Underlyings: x (continuous notionals)

Synthetics: q (contract counts, integer if you want)

Risk: factor model on h = x + S_contract @ q

Costs: linear exec, impact (3/2-power), borrow, funding, carry

Constraints:

Net exposure (on sum(h))

Gross exposure (sum |h|)

Per-name bounds on x, q

Per-name trade caps on |Δx|, |Δq|

Risk limit on σ



New structure constraints

All applied directly inside the MOSEK model:

1. Group exposure constraints on effective underlyings h
E.g. sectors, regions, styles.

Arguments:

group_matrix_h: shape (G, N) — each row is a group exposure vector over underlyings.

group_net_lb, group_net_ub: length G, optional.


Constraint:



g = G_{\text{groups}} h \in [\text{group\_net\_lb},\ \text{group\_net\_ub}]

Implemented via:

Gmat = Matrix.dense(G_groups)
g_exp = Gmat @ h
M.constraint("group_net", g_exp, Domain.inRange(lb_vec, ub_vec))

2. Factor exposure bounds

Arguments:

factor_exposure_lb, factor_exposure_ub: length K, optional.


Constraint:



f = B_{\text{stock}}^T h \in [\text{factor\_lb},\ \text{factor\_ub}]

So you can cap specific factor bets even while using the same B in the risk model.

3. Turnover limits

Besides per-name trade caps, you now also have aggregate turnover caps:

turnover_max_stock: bound on 

turnover_max_syn: bound on 


Uses the existing zx = |Δx|, zq = |Δq|:

M.constraint("turnover_stock", Expr.sum(zx), Domain.lessThan(turnover_max_stock))
M.constraint("turnover_syn",   Expr.sum(zq), Domain.lessThan(turnover_max_syn))



Auto-scaling integrates these

build_notional_portfolio_auto_scale(...) now also:

Scales group and factor bounds by 1/scale so they apply to scaled h.

Scales turnover limits by 1/scale.

Leaves group_matrix_h and B_stock unchanged.


So you pass everything in real units; the wrapper adjusts bounds for the internal scale.


---

3. High-level API & diagnostics (portfolio_api.py)

Config & input dataclasses

On top of what you had before, there are now:

GroupConstraintConfig:

@dataclass
class GroupConstraintConfig:
    group_names: List[str]          # length G
    group_matrix: List[List[float]] # (G, N)
    net_lb: Optional[List[float]] = None
    net_ub: Optional[List[float]] = None

FactorConstraintConfig:

@dataclass
class FactorConstraintConfig:
    factor_names: List[str]              # length K
    exposure_lb: Optional[List[float]] = None
    exposure_ub: Optional[List[float]] = None

TurnoverConstraintConfig:

@dataclass
class TurnoverConstraintConfig:
    turnover_max_stock: Optional[float] = None
    turnover_max_syn: Optional[float] = None


OptimizationInputs now bundles everything, including structural constraints and scaling:

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

Diagnostics dataclass enhancements

ConstraintDiagnostics now has richer structure:

@dataclass
class ConstraintDiagnostics:
    named_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    group_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    factor_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    turnover_slacks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    suspected_conflicting_constraints: List[str] = field(default_factory=list)

named_slacks: existing net/gross/sigma slacks

group_slacks: per group, with exposures and lb/ub slacks

factor_slacks: per factor, with exposures and lb/ub slacks

turnover_slacks: stock & synthetic totals vs limits

suspected_conflicting_constraints: any constraints with negative slack get tagged, using keys like:

"net_eq", "gross_lim", "sigma_max"

"group:Sector:Tech:lb", "factor:Value:ub"

"turnover:stock"



These strings are exactly what you’ll want to let an LLM latch onto when suggesting “relax this by X%”.

solve_portfolio(...) signature

Now supports structural constraints and diagnostics in one call:

run = solve_portfolio(
    asset_ids=asset_ids,
    synthetic_ids=synthetic_ids,
    underlying_for_synth=underlying_for_synth,
    B_stock=B_stock,
    Sigma_F=Sigma_F,
    theta_stock=theta_stock,
    S_contract=S_contract,
    alpha_stock=alpha_stock,
    alpha_syn=alpha_syn,
    x0_stock=x0_stock,
    q0_syn=q0_syn,
    constraint_cfg=constraint_cfg,
    bound_cfg=bound_cfg,
    group_cfg=group_cfg,          # optional
    factor_cfg=factor_cfg,        # optional
    turnover_cfg=turnover_cfg,    # optional
    lin_exec_stock=lin_exec_stock,
    lin_exec_syn=lin_exec_syn,
    impact_stock=impact_stock,
    impact_syn=impact_syn,
    borrow_stock=borrow_stock,
    funding_stock=funding_stock,
    carry_syn=carry_syn,
    solver_cfg=solver_cfg,
    target_scale=1e3,
)

What solve_portfolio returns

An OptimizationRun:

@dataclass
class OptimizationRun:
    inputs: OptimizationInputs
    solver_config: SolverConfig
    result: OptimizationResult

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

So you can do:

import json
payload = run.to_dict()
print(json.dumps(payload, indent=2))

and hand that JSON straight to an LLM.

Inside result you get:

solution:

status (OPTIMAL / INFEASIBLE / etc.)

objective

σ, net, gross, long, short


positions:

top longs / shorts with bound flags

counts of names at lower/upper bound


constraints:

net/gross/sigma slacks

group_slacks: per group exposures & slacks

factor_slacks: per factor exposures & slacks

turnover_slacks: trade totals & slacks

suspected_conflicting_constraints: list of “likely troublemakers”


solver:

problem/primal/dual statuses

objective




---

4. Example with structural constraints

Very rough sketch:

from portfolio_api import (
    ConstraintConfig, BoundConfig, GroupConstraintConfig,
    FactorConstraintConfig, TurnoverConstraintConfig,
    SolverConfig, solve_portfolio
)

# Suppose you have sectors per asset as a one-hot matrix sector_matrix (G x N)
group_cfg = GroupConstraintConfig(
    group_names=["Tech", "Financials", "Energy"],
    group_matrix=sector_matrix.tolist(),
    net_lb=[-5e6, -3e6, -2e6],
    net_ub=[+5e6, +3e6, +2e6],
)

# Cap factor exposures (e.g., Value, Momentum, Size)
factor_cfg = FactorConstraintConfig(
    factor_names=["Value", "Momentum", "Size"],
    exposure_lb=[-1e6, -5e5, -2e5],
    exposure_ub=[+1e6, +5e5, +2e5],
)

turnover_cfg = TurnoverConstraintConfig(
    turnover_max_stock=2e7,
    turnover_max_syn=5e3,
)

run = solve_portfolio(
    asset_ids=asset_ids,
    synthetic_ids=synthetic_ids,
    underlying_for_synth=underlying_for_synth,
    B_stock=B_stock,
    Sigma_F=Sigma_F,
    theta_stock=theta_stock,
    S_contract=S_contract,
    alpha_stock=alpha_stock,
    alpha_syn=alpha_syn,
    x0_stock=x0_stock,
    q0_syn=q0_syn,
    constraint_cfg=constraint_cfg,
    bound_cfg=bound_cfg,
    group_cfg=group_cfg,
    factor_cfg=factor_cfg,
    turnover_cfg=turnover_cfg,
    solver_cfg=SolverConfig(time_limit_sec=60, log_to_stdout=True),
)

payload = run.to_dict()
# -> hand payload to an LLM, log it, etc.


---

If you’re happy with this shape, you’re now set up to:

Run serious notional-based L/S optimizations with synthetics,

Get structured diagnostics when things go weird, and

Feed those diagnostics + config into an LLM to get “here’s what’s likely conflicting and what you might relax.”


If you later want to plug in lot sizes or per-synthetic integrality options, they’ll slot naturally into the same structure.
