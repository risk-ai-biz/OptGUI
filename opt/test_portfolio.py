
"""
Simple end-to-end test for the notional-based portfolio optimizer.

INTENTIONS / DESIGN NOTES
-------------------------
This test is deliberately small and heavily commented so you can:
  - See exactly how the data is constructed.
  - Debug / modify pieces if anything looks off.
  - Use it as a template for your own tests.

What we test:
  * N = 4 underlying assets, S = 2 synthetics (1 index future, 1 sector future).
  * K = 2 factors (e.g. Market, Value) with a simple, artificial factor covariance.
  * A consistent factor + specific risk model.
  * Synthetics defined as exact linear combos of underlyings.
  * Net, gross, group, factor, turnover constraints.
  * Integer contracts for synthetics.
  * Basic execution, borrow, and funding costs.

IMPORTANT:
  - This is a functional smoke test, not a formal unit test. We don't assert
    specific numerical optimum values (those depend on MOSEK, tolerances, etc.).
  - If you see any infeasibility or strange solutions, check the data generation
    sections below; all assumptions are spelled out in comments.
"""

import json
import numpy as np

from portfolio_api import (
    ConstraintConfig,
    BoundConfig,
    GroupConstraintConfig,
    FactorConstraintConfig,
    TurnoverConstraintConfig,
    SolverConfig,
    solve_portfolio,
)


def build_test_data():
    """Construct a small, self-contained test instance.

    Returns a dictionary with all fields needed by solve_portfolio(...).

    High-level scenario:
      - 4 underlyings: A, B, C, D
      - 2 synthetics:
          * F_INDEX: equal-weight future on all 4 assets
          * F_TECH:  future on the 'Tech' sector only (A and B)
      - 2 factors: 'MKT', 'VAL'
    """
    # ------------------------
    # 1. Universe definitions
    # ------------------------
    asset_ids = ["A", "B", "C", "D"]
    synthetic_ids = ["F_INDEX", "F_TECH"]

    # Map each synthetic to underlying IDs purely for metadata;
    # the actual risk mapping uses S_contract below.
    underlying_for_synth = {
        "F_INDEX": asset_ids,        # depends on all 4
        "F_TECH": ["A", "B"],       # depends on A and B only
    }

    N = len(asset_ids)
    S = len(synthetic_ids)
    K = 2  # two factors: MKT, VAL

    # ------------------------
    # 2. Factor model
    # ------------------------
    # Factor loadings B_stock: (N x K)
    # Intention:
    #   - A, B are 'Tech', with higher market beta and mixed value tilt.
    #   - C, D are 'Other', with lower market beta.
    #
    # Col 0: Market (MKT), Col 1: Value (VAL)
    B_stock = np.array([
        [1.2,  0.3],   # A
        [1.1, -0.1],   # B
        [0.8,  0.2],   # C
        [0.7, -0.2],   # D
    ], dtype=float)

    # Factor covariance Sigma_F: (K x K)
    # Intention:
    #   - Market factor has higher variance.
    #   - Value factor moderate variance.
    #   - Modest positive correlation.
    Sigma_F = np.array([
        [0.04, 0.01],  # Var(MKT)=0.04, Cov(MKT,VAL)=0.01
        [0.01, 0.02],  # Cov(MKT,VAL)=0.01, Var(VAL)=0.02
    ], dtype=float)

    # Residual variances theta_stock: (N,)
    # Intention:
    #   - A, B have slightly higher idiosyncratic risk (Tech).
    #   - C, D slightly lower.
    theta_stock = np.array([0.03, 0.028, 0.02, 0.018], dtype=float)

    # ------------------------
    # 3. Synthetics mapping S_contract
    # ------------------------
    # S_contract: (N x S) = notional in each underlying per 1 contract of each synthetic.
    #
    # For F_INDEX:
    #   - Intention: 1 contract corresponds to +1 unit notional in EACH stock (simple toy).
    #     So column 0 is [1, 1, 1, 1]^T.
    #
    # For F_TECH:
    #   - Intention: 1 contract is +1 in A and +1 in B only.
    #     So column 1 is [1, 1, 0, 0]^T.
    S_contract = np.array([
        [1.0, 1.0],   # A
        [1.0, 1.0],   # B
        [1.0, 0.0],   # C
        [1.0, 0.0],   # D
    ], dtype=float)

    # ------------------------
    # 4. Initial positions and alphas
    # ------------------------
    # Initial underlyings: flat book
    x0_stock = np.zeros(N, dtype=float)

    # Initial contracts: also flat
    q0_syn = np.zeros(S, dtype=float)

    # Alphas (per unit notional / per contract)
    # Intention:
    #   - Mildly positive alpha in Tech (A, B), neutral/low in others.
    #   - Small positive alpha for index future, negative for Tech future
    #     to force the optimizer to use underlyings more than the Tech future.
    alpha_stock = np.array([0.05, 0.04, 0.01, 0.0], dtype=float)
    alpha_syn = np.array([0.005, -0.002], dtype=float)

    # ------------------------
    # 5. Constraints / bounds
    # ------------------------
    # Notional bounds on stocks:
    #   - Allow up to +/- 2 units of notional in each
    pos_lb_stock = -2.0 * np.ones(N, dtype=float)
    pos_ub_stock =  2.0 * np.ones(N, dtype=float)

    # Contract bounds on synthetics:
    #   - Index future: between -5 and +5
    #   - Tech future:  between -3 and +3
    contract_lb = np.array([-5.0, -3.0], dtype=float)
    contract_ub = np.array([+5.0, +3.0], dtype=float)

    # Net exposure: for this test, enforce dollar-neutral (net_target = 0)
    constraint_cfg = ConstraintConfig(
        net_target=0.0,
        net_lb=None,
        net_ub=None,
        gross_max=8.0,      # limit sum |h_j| <= 8 notional units
        sigma_max=1.0,      # simple cap on σ (currency-vol)
        integer_contracts=True,
    )

    # Per-name trade caps:
    #   - Limit stock trades to at most 1.5 notional per name.
    #   - Limit synthetic trades to at most 2 contracts per instrument.
    trade_abs_max_stock = 1.5 * np.ones(N, dtype=float)
    trade_abs_max_syn = 2.0 * np.ones(S, dtype=float)

    bound_cfg = BoundConfig(
        pos_lb_stock=pos_lb_stock.tolist(),
        pos_ub_stock=pos_ub_stock.tolist(),
        contract_lb=contract_lb.tolist(),
        contract_ub=contract_ub.tolist(),
        trade_abs_max_stock=trade_abs_max_stock.tolist(),
        trade_abs_max_syn=trade_abs_max_syn.tolist(),
    )

    # ------------------------
    # 6. Group constraints
    # ------------------------
    # Two groups:
    #   G0: Tech (A,B)
    #   G1: Other (C,D)
    #
    # group_matrix (G x N):
    #   row 0: [1, 1, 0, 0] (Tech)
    #   row 1: [0, 0, 1, 1] (Other)
    group_matrix = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ], dtype=float)

    # Group net bounds on h (effective underlyings):
    #   - Tech exposure between -3 and +3
    #   - Other exposure between -3 and +3
    group_net_lb = [-3.0, -3.0]
    group_net_ub = [ 3.0,  3.0]

    group_cfg = GroupConstraintConfig(
        group_names=["Tech", "Other"],
        group_matrix=group_matrix.tolist(),
        net_lb=group_net_lb,
        net_ub=group_net_ub,
    )

    # ------------------------
    # 7. Factor exposure constraints
    # ------------------------
    # For demonstration, lightly constrain factor exposures:
    #
    #   Factor 0 (MKT): in [-4, +4]
    #   Factor 1 (VAL): in [-2, +2]
    factor_cfg = FactorConstraintConfig(
        factor_names=["MKT", "VAL"],
        exposure_lb=[-4.0, -2.0],
        exposure_ub=[+4.0, +2.0],
    )

    # ------------------------
    # 8. Turnover constraints
    # ------------------------
    # Intention:
    #   - Limit total stock turnover (sum |Δx|) to 4 units.
    #   - Limit total synthetic turnover (sum |Δq|) to 4 contracts.
    turnover_cfg = TurnoverConstraintConfig(
        turnover_max_stock=4.0,
        turnover_max_syn=4.0,
    )

    # ------------------------
    # 9. Costs
    # ------------------------
    # Linear execution costs: small, symmetric.
    lin_exec_stock = 0.01 * np.ones(N, dtype=float)
    lin_exec_syn = 0.005 * np.ones(S, dtype=float)

    # Impact costs: small 3/2-power coefficients.
    impact_stock = 0.001 * np.ones(N, dtype=float)
    impact_syn = 0.0005 * np.ones(S, dtype=float)

    # Borrow rates: higher for Tech (A,B), lower for Others (C,D).
    borrow_stock = np.array([0.02, 0.02, 0.01, 0.01], dtype=float)

    # Funding rates (for longs): uniform small carry.
    funding_stock = 0.005 * np.ones(N, dtype=float)

    # Carry on synthetics per |contract|: small uniform cost.
    carry_syn = 0.002 * np.ones(S, dtype=float)

    # ------------------------
    # 10. Solver config
    # ------------------------
    solver_cfg = SolverConfig(
        solver_name="mosek",
        time_limit_sec=60.0,
        mip_gap=1e-3,
        num_threads=0,        # 0 = let MOSEK decide
        log_to_stdout=True,   # set False to silence
    )

    return dict(
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
        lin_exec_stock=lin_exec_stock,
        lin_exec_syn=lin_exec_syn,
        impact_stock=impact_stock,
        impact_syn=impact_syn,
        borrow_stock=borrow_stock,
        funding_stock=funding_stock,
        carry_syn=carry_syn,
        solver_cfg=solver_cfg,
    )


def main():
    data = build_test_data()

    run = solve_portfolio(
        asset_ids=data["asset_ids"],
        synthetic_ids=data["synthetic_ids"],
        underlying_for_synth=data["underlying_for_synth"],
        B_stock=data["B_stock"],
        Sigma_F=data["Sigma_F"],
        theta_stock=data["theta_stock"],
        S_contract=data["S_contract"],
        alpha_stock=data["alpha_stock"],
        alpha_syn=data["alpha_syn"],
        x0_stock=data["x0_stock"],
        q0_syn=data["q0_syn"],
        constraint_cfg=data["constraint_cfg"],
        bound_cfg=data["bound_cfg"],
        group_cfg=data["group_cfg"],
        factor_cfg=data["factor_cfg"],
        turnover_cfg=data["turnover_cfg"],
        lin_exec_stock=data["lin_exec_stock"],
        lin_exec_syn=data["lin_exec_syn"],
        impact_stock=data["impact_stock"],
        impact_syn=data["impact_syn"],
        borrow_stock=data["borrow_stock"],
        funding_stock=data["funding_stock"],
        carry_syn=data["carry_syn"],
        solver_cfg=data["solver_cfg"],
        target_scale=10.0,  # keep internals around O(10)
    )

    # Pretty-print key results and diagnostics
    print("\n=== SOLUTION SUMMARY ===")
    print(json.dumps(run.result.solution.__dict__, indent=2))

    if run.result.positions is not None:
        print("\n=== TOP POSITIONS ===")
        print("Longs:")
        for rec in run.result.positions.top_long_positions:
            print(f"  {rec.asset_id}: notional={rec.notional:.4f}, "
                  f"lb={rec.at_lower_bound}, ub={rec.at_upper_bound}")
        print("Shorts:")
        for rec in run.result.positions.top_short_positions:
            print(f"  {rec.asset_id}: notional={rec.notional:.4f}, "
                  f"lb={rec.at_lower_bound}, ub={rec.at_upper_bound}")
        print(f"Num at lower bound: {run.result.positions.num_at_lower_bound}")
        print(f"Num at upper bound: {run.result.positions.num_at_upper_bound}")

    if run.result.constraints is not None:
        print("\n=== CONSTRAINT SLACKS (SIMPLE) ===")
        print(json.dumps(run.result.constraints.named_slacks, indent=2))

        print("\n=== GROUP SLACKS ===")
        print(json.dumps(run.result.constraints.group_slacks, indent=2))

        print("\n=== FACTOR SLACKS ===")
        print(json.dumps(run.result.constraints.factor_slacks, indent=2))

        print("\n=== TURNOVER SLACKS ===")
        print(json.dumps(run.result.constraints.turnover_slacks, indent=2))

        print("\n=== SUSPECTED CONFLICTS ===")
        print(run.result.constraints.suspected_conflicting_constraints)

    print("\n=== SOLVER DIAGNOSTICS ===")
    print(json.dumps(run.result.solver.__dict__, indent=2))


if __name__ == "__main__":
    main()
