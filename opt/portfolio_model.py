
"""
Notional-based long/short portfolio optimization with synthetics using MOSEK Fusion.

This module provides:
  - build_notional_portfolio_core:    build a MOSEK Fusion model from notional data
  - build_notional_portfolio_auto_scale: optional internal scaling for large notionals

The model:
  * Underlyings: continuous notional variables x (can be long/short)
  * Synthetics:  contract-count variables q (integer by default, configurable)
  * Risk:        factor model on effective underlyings h = x + S_contract @ q
  * Costs:       linear + impact execution, borrow, funding, carry
  * Constraints: net, gross, risk, position bounds, trade caps
  * Extra structure: group exposure constraints, factor exposure bounds,
                    turnover limits on total trading.
"""

from typing import Optional, Tuple, Union

import numpy as np
from mosek.fusion import Model, Domain, Expr, Matrix, ObjectiveSense
import mosek.fusion.pythonic  # noqa: F401  - enables @, .T, etc.


ArrayLike = Union[np.ndarray, list, tuple]


def _to_vec_or_none(v: Optional[ArrayLike], dim: int) -> Optional[np.ndarray]:
    """Convert v to a 1D numpy array of length dim, or None.

    Raises:
        ValueError if the shape is incompatible.
    """
    if v is None:
        return None
    a = np.asarray(v, dtype=float)
    if a.shape != (dim,):
        raise ValueError(f"Expected shape ({dim},), got {a.shape}")
    return a


def _to_bounds(lb: Optional[ArrayLike],
               ub: Optional[ArrayLike],
               dim: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert lower/upper bound containers to numpy arrays or None."""
    lb_vec = None if lb is None else np.asarray(lb, dtype=float)
    ub_vec = None if ub is None else np.asarray(ub, dtype=float)
    if lb_vec is not None and lb_vec.shape != (dim,):
        raise ValueError(f"lb has wrong shape {lb_vec.shape}, expected ({dim},)")
    if ub_vec is not None and ub_vec.shape != (dim,):
        raise ValueError(f"ub has wrong shape {ub_vec.shape}, expected ({dim},)")
    return lb_vec, ub_vec


def build_notional_portfolio_core(
    # Alpha / expected returns
    alpha_stock: ArrayLike,     # (N,)  per unit notional in underlyings
    alpha_syn: ArrayLike,       # (S,)  per contract of synthetic

    # Initial holdings
    x0_stock: ArrayLike,        # (N,)  initial notional in underlyings
    q0_syn: ArrayLike,          # (S,)  initial contract counts (can be fractional if relaxed)

    # Risk model on UNDERLYINGS ONLY
    B_stock: ArrayLike,         # (N, K) factor loadings
    Sigma_F: ArrayLike,         # (K, K) factor covariance
    theta_stock: ArrayLike,     # (N,)   residual variances
    S_contract: ArrayLike,      # (N, S) notional in underlyings per 1 contract of each synthetic

    # Position bounds (notionals / contracts)
    pos_lb_stock: Optional[ArrayLike] = None,
    pos_ub_stock: Optional[ArrayLike] = None,
    contract_lb: Optional[ArrayLike] = None,
    contract_ub: Optional[ArrayLike] = None,

    # Net exposure: on effective underlyings h = x + S_contract @ q
    net_target: Optional[float] = None,    # scalar, equality target (if not None)
    net_lb: Optional[float] = None,        # scalar, lower bound (if net_target is None)
    net_ub: Optional[float] = None,        # scalar, upper bound (if net_target is None)

    # Gross exposure: sum_j |h_j| <= gross_max (notionals)
    gross_max: Optional[float] = None,     # scalar or None

    # Risk controls
    risk_aversion: float = 1.0,            # λ in objective (currency-vol penalty)
    sigma_max: Optional[float] = None,     # optional upper bound on σ (same units as h)

    # Execution costs (optional, linear + 3/2 impact)
    lin_exec_stock: Optional[ArrayLike] = None,  # (N,) cost per unit |Δx|
    lin_exec_syn: Optional[ArrayLike] = None,    # (S,) cost per contract |Δq|
    impact_stock: Optional[ArrayLike] = None,    # (N,) m_j in m_j |Δx_j|^{3/2}
    impact_syn: Optional[ArrayLike] = None,      # (S,) m_k in m_k |Δq_k|^{3/2}

    # Borrow / funding / carry (optional)
    borrow_stock: Optional[ArrayLike] = None,    # (N,) cost * max(-x_j, 0)
    funding_stock: Optional[ArrayLike] = None,   # (N,) cost * max(x_j, 0)
    carry_syn: Optional[ArrayLike] = None,       # (S,) cost * (|q_k| = long+short)

    # Trade-size caps (optional, per-name)
    trade_abs_max_stock: Optional[ArrayLike] = None,  # (N,) or scalar, cap on |Δx|
    trade_abs_max_syn: Optional[ArrayLike] = None,    # (S,) or scalar, cap on |Δq|

    # Turnover limits (optional, aggregate)
    turnover_max_stock: Optional[float] = None,  # cap on sum_j |Δx_j|
    turnover_max_syn: Optional[float] = None,    # cap on sum_k |Δq_k|

    # Group exposure constraints on h (e.g. sectors, regions)
    # group_matrix_h: (G, N), group_net_lb/ub: (G,)
    group_matrix_h: Optional[ArrayLike] = None,
    group_net_lb: Optional[ArrayLike] = None,
    group_net_ub: Optional[ArrayLike] = None,

    # Factor exposure bounds: B_stock^T h in [lb, ub]
    factor_exposure_lb: Optional[ArrayLike] = None,  # (K,)
    factor_exposure_ub: Optional[ArrayLike] = None,  # (K,)

    # Integrality of contracts
    integer_contracts: bool = True,

    model_name: str = "notional_longshort_core",
) -> Tuple[Model, 'mosek.fusion.Variable', 'mosek.fusion.Variable', 'mosek.fusion.Variable']:
    """Build a MOSEK Fusion model for a notional-based long/short portfolio.

    Returns:
        (M, x, q, sigma) where:
          - M:     mosek.fusion.Model
          - x:     variable of length N (underlying notionals)
          - q:     variable of length S (contract counts)
          - sigma: scalar risk variable (portfolio currency-vol)
    """
    # ---- Convert inputs to numpy ----
    alpha_stock = np.asarray(alpha_stock, dtype=float)
    alpha_syn   = np.asarray(alpha_syn,   dtype=float)
    x0_stock    = np.asarray(x0_stock,    dtype=float)
    q0_syn      = np.asarray(q0_syn,      dtype=float)
    B_stock     = np.asarray(B_stock,     dtype=float)
    Sigma_F     = np.asarray(Sigma_F,     dtype=float)
    theta_stock = np.asarray(theta_stock, dtype=float)
    S_contract  = np.asarray(S_contract,  dtype=float)

    N, K = B_stock.shape
    S    = S_contract.shape[1]

    # Sanity checks
    if alpha_stock.shape != (N,):
        raise ValueError("alpha_stock shape mismatch")
    if alpha_syn.shape != (S,):
        raise ValueError("alpha_syn shape mismatch")
    if x0_stock.shape != (N,):
        raise ValueError("x0_stock shape mismatch")
    if q0_syn.shape != (S,):
        raise ValueError("q0_syn shape mismatch")
    if theta_stock.shape != (N,):
        raise ValueError("theta_stock shape mismatch")
    if Sigma_F.shape != (K, K):
        raise ValueError("Sigma_F must be K x K")  # noqa

    # Factor transformation: Sigma_F = P P^T
    P = np.linalg.cholesky(Sigma_F)
    G_factor_T = (B_stock @ P).T              # (K, N)
    sqrt_theta = np.sqrt(theta_stock)         # (N,)

    lin_exec_stock = _to_vec_or_none(lin_exec_stock, N)
    lin_exec_syn   = _to_vec_or_none(lin_exec_syn,   S)
    impact_stock   = _to_vec_or_none(impact_stock,   N)
    impact_syn     = _to_vec_or_none(impact_syn,     S)
    borrow_stock   = _to_vec_or_none(borrow_stock,   N)
    funding_stock  = _to_vec_or_none(funding_stock,  N)
    carry_syn      = _to_vec_or_none(carry_syn,      S)

    # Trade caps: allow scalar -> vector
    if trade_abs_max_stock is not None:
        if np.isscalar(trade_abs_max_stock):
            trade_abs_max_stock = float(trade_abs_max_stock) * np.ones(N)
        trade_abs_max_stock = _to_vec_or_none(trade_abs_max_stock, N)

    if trade_abs_max_syn is not None:
        if np.isscalar(trade_abs_max_syn):
            trade_abs_max_syn = float(trade_abs_max_syn) * np.ones(S)
        trade_abs_max_syn = _to_vec_or_none(trade_abs_max_syn, S)

    # Position bounds as numpy (or None)
    pos_lb_stock, pos_ub_stock = _to_bounds(pos_lb_stock, pos_ub_stock, N)
    contract_lb,  contract_ub  = _to_bounds(contract_lb,  contract_ub,  S)

    # Group exposure matrix & bounds
    if group_matrix_h is not None:
        group_matrix_h = np.asarray(group_matrix_h, dtype=float)
        if group_matrix_h.ndim != 2 or group_matrix_h.shape[1] != N:
            raise ValueError("group_matrix_h must have shape (G, N)")
        G_groups = group_matrix_h
        G = G_groups.shape[0]
        group_net_lb_vec = _to_vec_or_none(group_net_lb, G) if group_net_lb is not None else None
        group_net_ub_vec = _to_vec_or_none(group_net_ub, G) if group_net_ub is not None else None
    else:
        G_groups = None
        group_net_lb_vec = None
        group_net_ub_vec = None

    # Factor exposure bounds
    factor_lb_vec = _to_vec_or_none(factor_exposure_lb, K) if factor_exposure_lb is not None else None
    factor_ub_vec = _to_vec_or_none(factor_exposure_ub, K) if factor_exposure_ub is not None else None

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
        # x >= lb  => x - lb >= 0
        M.constraint("x_lb", Expr.sub(x, pos_lb_stock), Domain.greaterThan(0.0))
    if pos_ub_stock is not None:
        # x <= ub  => ub - x >= 0
        M.constraint("x_ub", Expr.sub(pos_ub_stock, x), Domain.greaterThan(0.0))

    if contract_lb is not None:
        M.constraint("q_lb", Expr.sub(q, contract_lb), Domain.greaterThan(0.0))
    if contract_ub is not None:
        M.constraint("q_ub", Expr.sub(contract_ub, q), Domain.greaterThan(0.0))

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

    # Trade caps (per-name)
    if trade_abs_max_stock is not None:
        # zx <= cap  => cap - zx >= 0
        M.constraint(
            "trade_cap_x",
            Expr.sub(trade_abs_max_stock, zx),
            Domain.greaterThan(0.0)
        )
    if trade_abs_max_syn is not None:
        M.constraint(
            "trade_cap_q",
            Expr.sub(trade_abs_max_syn, zq),
            Domain.greaterThan(0.0)
        )

    # Turnover limits (aggregate)
    if turnover_max_stock is not None:
        M.constraint(
            "turnover_stock",
            Expr.sum(zx),
            Domain.lessThan(float(turnover_max_stock))
        )
    if turnover_max_syn is not None:
        M.constraint(
            "turnover_syn",
            Expr.sum(zq),
            Domain.lessThan(float(turnover_max_syn))
        )

    # Market impact costs via power cones m_j |Δx_j|^{3/2}, etc.
    # (t_j, 1, Δx_j) ∈ POW_3^{(2/3,1/3)}  ⇒ t_j ≥ |Δx_j|^{3/2}
    if impact_stock is not None:
        t_x = M.variable("impact_x", N, Domain.greaterThan(0.0))
        M.constraint(
            "impact_pow_x",
            Expr.hstack(t_x, Expr.constTerm(N, 1.0), dx),
            Domain.inPPowerCone(2.0 / 3.0)
        )
    else:
        t_x = None

    if impact_syn is not None:
        t_q = M.variable("impact_q", S, Domain.greaterThan(0.0))
        M.constraint(
            "impact_pow_q",
            Expr.hstack(t_q, Expr.constTerm(S, 1.0), dq),
            Domain.inPPowerCone(2.0 / 3.0)
        )
    else:
        t_q = None

    # ---- Borrow / funding / carry on positions ----
    # x_long = max(x,0), x_short = max(-x,0)
    x_long  = M.variable("x_long",  N, Domain.greaterThan(0.0))
    x_short = M.variable("x_short", N, Domain.greaterThan(0.0))

    M.constraint("x_long_ge_x",   Expr.sub(x_long, x),  Domain.greaterThan(0.0))
    M.constraint("x_short_ge_mx", Expr.add(x_short, x), Domain.greaterThan(0.0))

    # q_long, q_short for carry on synthetics
    q_long  = M.variable("q_long",  S, Domain.greaterThan(0.0))
    q_short = M.variable("q_short", S, Domain.greaterThan(0.0))

    M.constraint("q_long_ge_q",   Expr.sub(q_long, q),  Domain.greaterThan(0.0))
    M.constraint("q_short_ge_mq", Expr.add(q_short, q), Domain.greaterThan(0.0))

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
        M.constraint("abs_h_pos", Expr.sub(abs_h, h),  Domain.greaterThan(0.0))
        M.constraint("abs_h_neg", Expr.add(abs_h, h),  Domain.greaterThan(0.0))
        M.constraint("gross_lim", Expr.sum(abs_h),     Domain.lessThan(float(gross_max)))

    # ---- Group exposure constraints ----
    if G_groups is not None and (group_net_lb_vec is not None or group_net_ub_vec is not None):
        Gmat = Matrix.dense(G_groups)  # (G, N)
        g_exp = Gmat @ h               # length G expression
        if group_net_lb_vec is None:
            lb_vec = -np.inf * np.ones(G_groups.shape[0])
        else:
            lb_vec = group_net_lb_vec
        if group_net_ub_vec is None:
            ub_vec = np.inf * np.ones(G_groups.shape[0])
        else:
            ub_vec = group_net_ub_vec
        M.constraint("group_net", g_exp, Domain.inRange(lb_vec, ub_vec))

    # ---- Factor exposure constraints ----
    if factor_lb_vec is not None or factor_ub_vec is not None:
        B_T = Matrix.dense(B_stock.T)   # (K, N)
        f_exp = B_T @ h                # factor exposures
        if factor_lb_vec is None:
            lb_f = -np.inf * np.ones(K)
        else:
            lb_f = factor_lb_vec
        if factor_ub_vec is None:
            ub_f = np.inf * np.ones(K)
        else:
            ub_f = factor_ub_vec
        M.constraint("factor_exposure", f_exp, Domain.inRange(lb_f, ub_f))

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

    M.objective("obj", ObjectiveSense.Maximize, obj)

    return M, x, q, sigma


def build_notional_portfolio_auto_scale(
    alpha_stock: ArrayLike,
    alpha_syn: ArrayLike,
    x0_stock: ArrayLike,
    q0_syn: ArrayLike,
    B_stock: ArrayLike,
    Sigma_F: ArrayLike,
    theta_stock: ArrayLike,
    S_contract: ArrayLike,

    pos_lb_stock: Optional[ArrayLike] = None,
    pos_ub_stock: Optional[ArrayLike] = None,
    contract_lb: Optional[ArrayLike] = None,
    contract_ub: Optional[ArrayLike] = None,

    net_target: Optional[float] = None,
    net_lb: Optional[float] = None,
    net_ub: Optional[float] = None,
    gross_max: Optional[float] = None,

    risk_aversion: float = 1.0,
    sigma_max: Optional[float] = None,

    lin_exec_stock: Optional[ArrayLike] = None,
    lin_exec_syn: Optional[ArrayLike] = None,
    impact_stock: Optional[ArrayLike] = None,
    impact_syn: Optional[ArrayLike] = None,
    borrow_stock: Optional[ArrayLike] = None,
    funding_stock: Optional[ArrayLike] = None,
    carry_syn: Optional[ArrayLike] = None,
    trade_abs_max_stock: Optional[ArrayLike] = None,
    trade_abs_max_syn: Optional[ArrayLike] = None,

    turnover_max_stock: Optional[float] = None,
    turnover_max_syn: Optional[float] = None,

    group_matrix_h: Optional[ArrayLike] = None,
    group_net_lb: Optional[ArrayLike] = None,
    group_net_ub: Optional[ArrayLike] = None,

    factor_exposure_lb: Optional[ArrayLike] = None,
    factor_exposure_ub: Optional[ArrayLike] = None,

    integer_contracts: bool = True,
    target_scale: float = 1e3,           # "nice" internal notional scale
    model_name: str = "notional_longshort_scaled",
) -> Tuple[Model, 'mosek.fusion.Variable', 'mosek.fusion.Variable', 'mosek.fusion.Variable', float]:
    """Build a scaled notional-based portfolio model.

    Chooses an internal scale factor so that notionals are O(target_scale),
    which can improve numerical conditioning. Returns:

        (M, x_var, q_var, sigma_var, scale)

    where x_var is in *scaled* units:

        x_real = x_var.level() * scale
    """
    x0_stock   = np.asarray(x0_stock,   dtype=float)
    S_contract = np.asarray(S_contract, dtype=float)
    N, S = S_contract.shape

    def _max_abs(arr) -> float:
        if arr is None:
            return 0.0
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return 0.0
        return float(np.max(np.abs(a)))

    # Candidate magnitudes to determine scale
    mags = [
        _max_abs(x0_stock),
        _max_abs(pos_lb_stock),
        _max_abs(pos_ub_stock),
        _max_abs(trade_abs_max_stock),
        float(abs(net_target)) if net_target is not None else 0.0,
        float(abs(net_lb)) if net_lb is not None else 0.0,
        float(abs(net_ub)) if net_ub is not None else 0.0,
        float(abs(gross_max)) if gross_max is not None else 0.0,
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
            turnover_max_stock=turnover_max_stock,
            turnover_max_syn=turnover_max_syn,
            group_matrix_h=group_matrix_h,
            group_net_lb=group_net_lb,
            group_net_ub=group_net_ub,
            factor_exposure_lb=factor_exposure_lb,
            factor_exposure_ub=factor_exposure_ub,
            integer_contracts=integer_contracts,
            model_name=model_name,
        )
        return M, x, q, sigma, scale

    # --- Scale notional-like things by 1/scale ---
    def _scale_vec(v):
        if v is None:
            return None
        return np.asarray(v, dtype=float) / scale

    x0_scaled         = x0_stock / scale
    pos_lb_scaled     = _scale_vec(pos_lb_stock)
    pos_ub_scaled     = _scale_vec(pos_ub_stock)
    S_scaled          = S_contract / scale
    net_target_scaled = None if net_target is None else float(net_target) / scale
    net_lb_scaled     = None if net_lb is None else float(net_lb) / scale
    net_ub_scaled     = None if net_ub is None else float(net_ub) / scale
    gross_scaled      = None if gross_max is None else float(gross_max) / scale
    trade_x_scaled    = _scale_vec(trade_abs_max_stock)
    sigma_max_scaled  = None if sigma_max is None else float(sigma_max) / scale

    # Group bounds scaling: matrix itself unchanged, bounds scale with h
    group_net_lb_scaled = _scale_vec(group_net_lb) if group_net_lb is not None else None
    group_net_ub_scaled = _scale_vec(group_net_ub) if group_net_ub is not None else None

    # Factor exposure bounds scale with h
    factor_lb_scaled = _scale_vec(factor_exposure_lb) if factor_exposure_lb is not None else None
    factor_ub_scaled = _scale_vec(factor_exposure_ub) if factor_exposure_ub is not None else None

    # Turnover limits scale with |Δx|, |Δq|
    turnover_stock_scaled = None if turnover_max_stock is None else float(turnover_max_stock) / scale
    turnover_syn_scaled   = None if turnover_max_syn is None else float(turnover_max_syn) / scale

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
        turnover_max_stock=turnover_stock_scaled,
        turnover_max_syn=turnover_syn_scaled,
        group_matrix_h=group_matrix_h,
        group_net_lb=group_net_lb_scaled,
        group_net_ub=group_net_ub_scaled,
        factor_exposure_lb=factor_lb_scaled,
        factor_exposure_ub=factor_ub_scaled,
        integer_contracts=integer_contracts,
        model_name=model_name,
    )

    return M, x, q, sigma, scale
