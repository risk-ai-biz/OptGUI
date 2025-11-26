Let’s lock this down properly: canonical notional-based model + auto-scaling wrapper, both using a factor-model risk, synthetics via contracts, and all the costs we discussed.

I’ll give you:

1. build_notional_portfolio_core(...) – raw notionals in, raw notionals out.


2. build_notional_portfolio_auto_scale(...) – same interface but automatically rescales notionals internally for better numerics, and tells you the scale factor.



Both are written for mosek.fusion and compatible with mosek.fusion.pythonic (I import it just for nice @ and .T behavior).


---

1. Canonical notional-based model (core)

Key modeling choices:

Variables

x ∈ ℝᴺ – notional exposures in underlyings (can be ±).

q ∈ ℤᴿ – contract counts in synthetics (integer if requested).


Mapping

S_contract (N×S): column k = notional in underlyings per 1 contract of synthetic k.

Effective underlyings: h = x + S_contract @ q (notionals, what risk sees).


Risk (factor model)

Factor loadings: B_stock (N×K), covariance Sigma_F (K×K), residual variances theta_stock (N,).

Risk cone:



(\sigma,\; P^\top B^\top h,\; \sqrt{\theta}\odot h) \in \mathcal{Q},
    \quad \Sigma_F = P P^\top

Execution: linear + 3/2 impact on |Δx| and |Δq| via power cones.

Borrow: cost on shorts max(-x, 0).

Funding: cost on longs max(x, 0).

Carry: cost on |q| = long + short contracts.

Constraints

Position bounds for x, q (notionals / contracts).

Net exposure (either equality or bounds) on sum(h).

Gross exposure sum |h| ≤ gross_max.

Optional risk limit σ ≤ sigma_max.

Optional trade-size caps on |Δx|, |Δq|.



Here’s the core builder:

import numpy as np
from mosek.fusion import Model, Domain, Expr, Matrix
import mosek.fusion.pythonic  # for @, .T, etc.


def build_notional_portfolio_core(
    # Alpha / expected returns
    alpha_stock,     # (N,)  per unit notional in underlyings
    alpha_syn,       # (S,)  per contract of synthetic

    # Initial holdings
    x0_stock,        # (N,)  initial notional in underlyings
    q0_syn,          # (S,)  initial contract counts (can be fractional if relaxed)

    # Risk model on UNDERLYINGS ONLY
    B_stock,         # (N, K) factor loadings
    Sigma_F,         # (K, K) factor covariance
    theta_stock,     # (N,)   residual variances
    S_contract,      # (N, S) notional in underlyings per 1 contract of each synthetic

    # Position bounds (notionals / contracts)
    pos_lb_stock=None, pos_ub_stock=None,   # (N,) or None
    contract_lb=None,  contract_ub=None,    # (S,) or None

    # Net exposure: on effective underlyings h = x + S_contract @ q
    net_target=None,    # scalar, == target  (if not None)
    net_lb=None,        # scalar, lower bound (if net_target is None)
    net_ub=None,        # scalar, upper bound (if net_target is None)

    # Gross exposure: sum_j |h_j| <= gross_max (notionals)
    gross_max=None,     # scalar or None

    # Risk controls
    risk_aversion=1.0,  # λ in objective (currency-vol penalty)
    sigma_max=None,     # optional upper bound on σ (same units as h)

    # Execution costs (optional, linear + 3/2 impact)
    lin_exec_stock=None,  # (N,) cost per unit |Δx|
    lin_exec_syn=None,    # (S,) cost per contract |Δq|
    impact_stock=None,    # (N,) m_j in m_j |Δx_j|^{3/2}
    impact_syn=None,      # (S,) m_k in m_k |Δq_k|^{3/2}

    # Borrow / funding / carry (optional)
    borrow_stock=None,    # (N,) cost * max(-x_j, 0)
    funding_stock=None,   # (N,) cost * max(x_j, 0)
    carry_syn=None,       # (S,) cost * (|q_k| = long+short)

    # Trade-size caps (optional)
    trade_abs_max_stock=None,  # (N,) or scalar, cap on |Δx|
    trade_abs_max_syn=None,    # (S,) or scalar, cap on |Δq|

    # Integrality of contracts
    integer_contracts=True,

    model_name="notional_longshort_core",
):
    # ---- Convert inputs to numpy ----
    alpha_stock  = np.asarray(alpha_stock,  float)
    alpha_syn    = np.asarray(alpha_syn,    float)
    x0_stock     = np.asarray(x0_stock,     float)
    q0_syn       = np.asarray(q0_syn,       float)
    B_stock      = np.asarray(B_stock,      float)
    Sigma_F      = np.asarray(Sigma_F,      float)
    theta_stock  = np.asarray(theta_stock,  float)
    S_contract   = np.asarray(S_contract,   float)

    N, K = B_stock.shape
    S    = S_contract.shape[1]

    # Sanity checks
    assert alpha_stock.shape == (N,)
    assert alpha_syn.shape   == (S,)
    assert x0_stock.shape    == (N,)
    assert q0_syn.shape      == (S,)
    assert theta_stock.shape == (N,)
    assert Sigma_F.shape     == (K, K)

    # Factor transformation: Sigma_F = P P^T
    P = np.linalg.cholesky(Sigma_F)
    G_factor_T = (B_stock @ P).T              # (K, N)
    sqrt_theta = np.sqrt(theta_stock)         # (N,)

    def to_vec_or_none(v, dim):
        if v is None:
            return None
        v = np.asarray(v, float)
        if v.shape != (dim,):
            raise ValueError(f"Expected shape ({dim},), got {v.shape}")
        return v

    lin_exec_stock = to_vec_or_none(lin_exec_stock, N)
    lin_exec_syn   = to_vec_or_none(lin_exec_syn,   S)
    impact_stock   = to_vec_or_none(impact_stock,   N)
    impact_syn     = to_vec_or_none(impact_syn,     S)
    borrow_stock   = to_vec_or_none(borrow_stock,   N)
    funding_stock  = to_vec_or_none(funding_stock,  N)
    carry_syn      = to_vec_or_none(carry_syn,      S)

    # Trade caps: allow scalar → vector
    if trade_abs_max_stock is not None:
        if np.isscalar(trade_abs_max_stock):
            trade_abs_max_stock = float(trade_abs_max_stock) * np.ones(N)
        trade_abs_max_stock = to_vec_or_none(trade_abs_max_stock, N)

    if trade_abs_max_syn is not None:
        if np.isscalar(trade_abs_max_syn):
            trade_abs_max_syn = float(trade_abs_max_syn) * np.ones(S)
        trade_abs_max_syn = to_vec_or_none(trade_abs_max_syn, S)

    # Position bounds as numpy (or None)
    def to_bounds(lb, ub, dim):
        lb_vec = None if lb is None else np.asarray(lb, float)
        ub_vec = None if ub is None else np.asarray(ub, float)
        if lb_vec is not None and lb_vec.shape != (dim,):
            raise ValueError("lb has wrong shape")
        if ub_vec is not None and ub_vec.shape != (dim,):
            raise ValueError("ub has wrong shape")
        return lb_vec, ub_vec

    pos_lb_stock, pos_ub_stock = to_bounds(pos_lb_stock, pos_ub_stock, N)
    contract_lb,  contract_ub  = to_bounds(contract_lb,  contract_ub,  S)

    # ---- Build model ----
    M = Model(model_name)

    # Underlying notionals x (continuous)
    x = M.variable("x_stock", N, Domain.unbounded())

    # Contract counts q (integer or continuous)
    q = M.variable("q_syn", S, Domain.unbounded())
    if integer_contracts:
        q.makeInteger()

    # Position bounds as explicit constraints
    if pos_lb_stock is not None:
        M.constraint("x_lb", Expr.sub(x, pos_lb_stock), Domain.greaterThan(0.0))  # x >= lb
    if pos_ub_stock is not None:
        M.constraint("x_ub", Expr.sub(pos_ub_stock, x), Domain.greaterThan(0.0))  # x <= ub

    if contract_lb is not None:
        M.constraint("q_lb", Expr.sub(q, contract_lb), Domain.greaterThan(0.0))   # q >= lb
    if contract_ub is not None:
        M.constraint("q_ub", Expr.sub(contract_ub, q), Domain.greaterThan(0.0))   # q <= ub

    # Effective underlying notionals: h = x + S_contract @ q
    SC = Matrix.dense(S_contract)   # (N x S)
    h  = Expr.add(x, SC @ q)        # affine expression, not a variable

    # Risk variable σ (currency-vol in same units as h)
    sigma = M.variable("sigma", 1, Domain.greaterThan(0.0))

    # Factor-model risk cone: (σ, G^T h, sqrt(θ) ⊙ h) ∈ Q
    M.constraint(
        "risk_cone",
        Expr.vstack(
            sigma,
            G_factor_T @ h,                # factor part
            Expr.mulElm(sqrt_theta, h)     # specific part
        ),
        Domain.inQCone()
    )

    # Optional risk limit σ ≤ sigma_max
    if sigma_max is not None:
        M.constraint("sigma_max", sigma, Domain.lessThan(float(sigma_max)))

    # ---- Trades and execution costs ----
    dx = Expr.sub(x, x0_stock)
    dq = Expr.sub(q, q0_syn)

    zx = M.variable("abs_dx", N, Domain.greaterThan(0.0))
    zq = M.variable("abs_dq", S, Domain.greaterThan(0.0))

    # zx >= |dx|
    M.constraint("abs_dx_pos", Expr.sub(zx, dx),  Domain.greaterThan(0.0))
    M.constraint("abs_dx_neg", Expr.add(zx, dx),  Domain.greaterThan(0.0))
    # zq >= |dq|
    M.constraint("abs_dq_pos", Expr.sub(zq, dq),  Domain.greaterThan(0.0))
    M.constraint("abs_dq_neg", Expr.add(zq, dq),  Domain.greaterThan(0.0))

    # Trade caps
    if trade_abs_max_stock is not None:
        M.constraint("trade_cap_x", Expr.sub(trade_abs_max_stock, zx),
                     Domain.greaterThan(0.0))   # zx <= cap
    if trade_abs_max_syn is not None:
        M.constraint("trade_cap_q", Expr.sub(trade_abs_max_syn, zq),
                     Domain.greaterThan(0.0))   # zq <= cap

    # Market impact costs via power cones m_j |Δx_j|^{3/2}, etc.
    # (t_j, 1, Δx_j) ∈ POW_3^{(2/3,1/3)}  ⇒ t_j ≥ |Δx_j|^{3/2}
    if impact_stock is not None:
        t_x = M.variable("impact_x", N, Domain.greaterThan(0.0))
        M.constraint(
            "impact_pow_x",
            Expr.hstack(t_x, Expr.constTerm(N, 1.0), dx),
            Domain.inPPowerCone(2.0/3.0)
        )
    else:
        t_x = None

    if impact_syn is not None:
        t_q = M.variable("impact_q", S, Domain.greaterThan(0.0))
        M.constraint(
            "impact_pow_q",
            Expr.hstack(t_q, Expr.constTerm(S, 1.0), dq),
            Domain.inPPowerCone(2.0/3.0)
        )
    else:
        t_q = None

    # ---- Borrow / funding / carry on positions ----
    # x_long = max(x,0), x_short = max(-x,0)
    x_long  = M.variable("x_long",  N, Domain.greaterThan(0.0))
    x_short = M.variable("x_short", N, Domain.greaterThan(0.0))

    M.constraint("x_long_ge_x",   Expr.sub(x_long, x),  Domain.greaterThan(0.0))  # x_long >= x
    M.constraint("x_short_ge_mx", Expr.add(x_short, x), Domain.greaterThan(0.0))  # x_short >= -x

    # q_long, q_short for carry on synthetics
    q_long  = M.variable("q_long",  S, Domain.greaterThan(0.0))
    q_short = M.variable("q_short", S, Domain.greaterThan(0.0))

    M.constraint("q_long_ge_q",   Expr.sub(q_long, q),  Domain.greaterThan(0.0))  # q_long >= q
    M.constraint("q_short_ge_mq", Expr.add(q_short, q), Domain.greaterThan(0.0))  # q_short >= -q

    # ---- Net & gross exposure constraints (on h) ----
    # Net exposure
    if net_target is not None:
        M.constraint("net_eq", Expr.sum(h), Domain.equalsTo(float(net_target)))
    elif (net_lb is not None) or (net_ub is not None):
        lb = -np.inf if net_lb is None else float(net_lb)
        ub =  np.inf if net_ub is None else float(net_ub)
        M.constraint("net_range", Expr.sum(h), Domain.inRange(lb, ub))

    # Gross exposure: sum |h_j| <= gross_max
    if gross_max is not None:
        abs_h = M.variable("abs_h", N, Domain.greaterThan(0.0))
        M.constraint("abs_h_pos", Expr.sub(abs_h, h),  Domain.greaterThan(0.0))   # abs_h >= h
        M.constraint("abs_h_neg", Expr.add(abs_h, h),  Domain.greaterThan(0.0))   # abs_h >= -h
        M.constraint("gross_lim", Expr.sum(abs_h),     Domain.lessThan(float(gross_max)))

    # ---- Objective: maximize alpha^T x + alpha^T q - λσ - costs ----
    ret_term  = Expr.add(Expr.dot(alpha_stock, x), Expr.dot(alpha_syn, q))
    risk_term = Expr.mul(risk_aversion, sigma)

    obj = Expr.sub(ret_term, risk_term)

    # Linear execution costs
    if lin_exec_stock is not None:
        obj = Expr.sub(obj, Expr.dot(lin_exec_stock, zx))
    if lin_exec_syn is not None:
        obj = Expr.sub(obj, Expr.dot(lin_exec_syn, zq))

    # Impact costs
    if t_x is not None:
        obj = Expr.sub(obj, Expr.dot(impact_stock, t_x))
    if t_q is not None:
        obj = Expr.sub(obj, Expr.dot(impact_syn, t_q))

    # Borrow / funding
    if borrow_stock is not None:
        obj = Expr.sub(obj, Expr.dot(borrow_stock, x_short))
    if funding_stock is not None:
        obj = Expr.sub(obj, Expr.dot(funding_stock, x_long))

    # Carry on synthetic contracts
    if carry_syn is not None:
        obj = Expr.sub(obj, Expr.dot(carry_syn, Expr.add(q_long, q_short)))

    M.objective("obj", Model.ObjectiveSense.Maximize, obj)

    return M, x, q, sigma

Usage (core version):

M, x, q, sigma = build_notional_portfolio_core(
    alpha_stock=alpha_stock,
    alpha_syn=alpha_syn,
    x0_stock=x0,
    q0_syn=q0,
    B_stock=B,
    Sigma_F=Sigma_F,
    theta_stock=theta,
    S_contract=S_contract,
    net_target=0.0,         # e.g. dollar-neutral
    gross_max=1e8,          # e.g. 100mm gross limit
    risk_aversion=5.0,
    sigma_max=None,
    lin_exec_stock=lin_exec_stock,
    lin_exec_syn=lin_exec_syn,
    impact_stock=impact_stock,
    impact_syn=impact_syn,
    borrow_stock=borrow,
    funding_stock=funding,
    carry_syn=carry,
    integer_contracts=True,
)

M.solve()

x_star = np.array(x.level())         # notional in underlyings
q_star = np.array(q.level())         # contract counts (integers if requested)
sigma_star = float(sigma.level()[0]) # currency-vol in same units as notional
# Effective underlyings: h_star = x_star + S_contract @ q_star


---

2. Auto-scaling wrapper (to tame big notionals)

This wrapper:

Looks at your notional magnitudes (x₀, bounds, S_contract, gross/net),

Chooses a scale factor s (e.g. if max notionals are around 1e9 and target_scale=1e3, it takes s ≈ 1e6),

Internally solves the scaled problem in “model units” = notional / s,

Returns the model, variables, and the scale factor so you can convert solutions back.


Scaling logic:

Internal variables: x_int = x_real / s, h_int = h_real / s, S_int = S_real / s.

Risk: σ_int = σ_real / s.

We keep α, λ, costs unchanged; objective just gets multiplied by s, which doesn’t change the argmax (per the reparameterization argument).


Wrapper:

def build_notional_portfolio_auto_scale(
    alpha_stock,
    alpha_syn,
    x0_stock,
    q0_syn,
    B_stock,
    Sigma_F,
    theta_stock,
    S_contract,

    pos_lb_stock=None, pos_ub_stock=None,
    contract_lb=None,  contract_ub=None,

    net_target=None,
    net_lb=None,
    net_ub=None,
    gross_max=None,

    risk_aversion=1.0,
    sigma_max=None,

    lin_exec_stock=None,
    lin_exec_syn=None,
    impact_stock=None,
    impact_syn=None,
    borrow_stock=None,
    funding_stock=None,
    carry_syn=None,
    trade_abs_max_stock=None,
    trade_abs_max_syn=None,

    integer_contracts=True,
    target_scale=1e3,           # "nice" internal notional scale
    model_name="notional_longshort_scaled",
):
    # Convert to numpy for scale detection
    x0_stock   = np.asarray(x0_stock,   float)
    S_contract = np.asarray(S_contract, float)
    N, S = S_contract.shape

    # Helper to collect magnitudes from vectors that may be None
    def max_abs(arr):
        if arr is None:
            return 0.0
        a = np.asarray(arr, float)
        if a.size == 0:
            return 0.0
        return float(np.max(np.abs(a)))

    # Candidate magnitudes to determine scale
    mags = [
        max_abs(x0_stock),
        max_abs(pos_lb_stock),
        max_abs(pos_ub_stock),
        max_abs(trade_abs_max_stock),
        max_abs(net_target),
        max_abs(net_lb),
        max_abs(net_ub),
        max_abs(gross_max),
        float(np.max(np.abs(S_contract))) if S_contract.size > 0 else 0.0,
    ]
    max_mag = max(mags) if mags else 0.0

    if max_mag > target_scale and max_mag > 0:
        scale = max_mag / target_scale
    else:
        scale = 1.0

    # If scale == 1, just build core model
    if scale == 1.0:
        M, x, q, sigma = build_notional_portfolio_core(
            alpha_stock=alpha_stock,
            alpha_syn=alpha_syn,
            x0_stock=x0_stock,
            q0_syn=q0_syn,
            B_stock=B_stock,
            Sigma_F=Sigma_F,
            theta_stock=theta_stock,
            S_contract=S_contract,
            pos_lb_stock=pos_lb_stock,
            pos_ub_stock=pos_ub_stock,
            contract_lb=contract_lb,
            contract_ub=contract_ub,
            net_target=net_target,
            net_lb=net_lb,
            net_ub=net_ub,
            gross_max=gross_max,
            risk_aversion=risk_aversion,
            sigma_max=sigma_max,
            lin_exec_stock=lin_exec_stock,
            lin_exec_syn=lin_exec_syn,
            impact_stock=impact_stock,
            impact_syn=impact_syn,
            borrow_stock=borrow_stock,
            funding_stock=funding_stock,
            carry_syn=carry_syn,
            trade_abs_max_stock=trade_abs_max_stock,
            trade_abs_max_syn=trade_abs_max_syn,
            integer_contracts=integer_contracts,
            model_name=model_name,
        )
        return M, x, q, sigma, scale

    # --- Scale notional-like things by 1/scale ---
    def scale_vec(v):
        if v is None:
            return None
        return np.asarray(v, float) / scale

    x0_scaled        = x0_stock / scale
    pos_lb_scaled    = scale_vec(pos_lb_stock)
    pos_ub_scaled    = scale_vec(pos_ub_stock)
    S_scaled         = S_contract / scale
    net_target_scaled= None if net_target is None else float(net_target) / scale
    net_lb_scaled    = None if net_lb is None else float(net_lb) / scale
    net_ub_scaled    = None if net_ub is None else float(net_ub) / scale
    gross_scaled     = None if gross_max is None else float(gross_max) / scale
    trade_x_scaled   = scale_vec(trade_abs_max_stock)
    # sigma_max must be scaled because σ_int = σ_real / scale
    sigma_max_scaled = None if sigma_max is None else float(sigma_max) / scale

    # Note: q0_syn, contract bounds, alphas, lambdas, costs remain in their original units.
    # Only exposures and exposure-like limits are scaled.

    M, x, q, sigma = build_notional_portfolio_core(
        alpha_stock=alpha_stock,
        alpha_syn=alpha_syn,
        x0_stock=x0_scaled,
        q0_syn=q0_syn,
        B_stock=B_stock,
        Sigma_F=Sigma_F,
        theta_stock=theta_stock,
        S_contract=S_scaled,
        pos_lb_stock=pos_lb_scaled,
        pos_ub_stock=pos_ub_scaled,
        contract_lb=contract_lb,
        contract_ub=contract_ub,
        net_target=net_target_scaled,
        net_lb=net_lb_scaled,
        net_ub=net_ub_scaled,
        gross_max=gross_scaled,
        risk_aversion=risk_aversion,
        sigma_max=sigma_max_scaled,
        lin_exec_stock=lin_exec_stock,
        lin_exec_syn=lin_exec_syn,
        impact_stock=impact_stock,
        impact_syn=impact_syn,
        borrow_stock=borrow_stock,
        funding_stock=funding_stock,
        carry_syn=carry_syn,
        trade_abs_max_stock=trade_x_scaled,
        trade_abs_max_syn=trade_abs_max_syn,
        integer_contracts=integer_contracts,
        model_name=model_name,
    )

    return M, x, q, sigma, scale

Reading the scaled solution back to real notionals

Example:

M, x_var, q_var, sigma_var, scale = build_notional_portfolio_auto_scale(
    alpha_stock=alpha_stock,
    alpha_syn=alpha_syn,
    x0_stock=x0_notional,
    q0_syn=q0_contracts,
    B_stock=B,
    Sigma_F=Sigma_F,
    theta_stock=theta,
    S_contract=S_contract,   # notional per contract
    net_target=0.0,
    gross_max=2e8,
    risk_aversion=3.0,
    # ...costs etc...
    integer_contracts=True,
    target_scale=1e3,
)

M.solve()

x_scaled   = np.array(x_var.level())          # internal units
q_star     = np.array(q_var.level())          # contracts (not scaled)
sigma_int  = float(sigma_var.level()[0])

# Convert back to real notionals
x_star     = x_scaled * scale                # underlyings (real notionals)
sigma_star = sigma_int * scale               # real currency-vol
h_star     = x_star + S_contract @ q_star    # effective underlying notionals



This gives you:

A canonical notional-based model that behaves correctly for:

long/short underlyings,

synthetics as exact linear combos (so long-basket + short-future can be ~zero risk),

borrow/funding/execution/impact/carry,

net & gross & risk constraints,

integer contract lots
