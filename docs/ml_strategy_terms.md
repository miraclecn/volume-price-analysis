# ML Strategy Terms

This document fixes the Phase 9 vocabulary used by the ML strategy layer.

## Term Hierarchy

`model`
: One trained artifact, for example an absolute-return ranker or a risk model.

`model_bundle`
: A registered group of model artifacts used together for inference. A bundle may
contain absolute, active, and risk models, but a strategy does not have to use
every role.

`score_mode`
: The rule that converts model outputs into ranking and risk-filtering fields.
Examples: `v2_three_model`, `v2_absolute_risk_filter`,
`v2_absolute_risk_sort`.

`portfolio_strategy`
: The holding and execution rule applied after scoring. Examples:
`holding_aware_v2`, `abs_ranker_fixed_5d_risk_filter_v1`.

`sleeve`
: A capital bucket that can receive an allocation. Phase 9 sleeves are `core`,
`aggressive`, `fixed_horizon`, and `cash`.

`strategy_allocation`
: The final capital weights after applying market regime, model health, and
account drawdown controls.

## Current Best Strategy Placement

The current best live-simulation candidate is not a single model. It is the
initial `core` sleeve:

```text
sleeve: core
experiment_family: expanding_gap
gap_type: one_year_gap
model_roles: absolute + risk
score_mode: v2_absolute_risk_filter
portfolio_strategy: holding_aware_v2
```

This sleeve remains runnable through the full year. Phase 9 does not replace it
with another single-model search. Phase 9 allocates capital across this core
sleeve, an optional aggressive sleeve, fixed-horizon sleeves, and cash.

## Phase 9 Allocation Rule

The allocation layer uses this order:

```text
regime profile -> model health gate -> account drawdown multiplier -> cash residual
```

Non-cash sleeve budget reduced by health or drawdown controls is moved to cash.
This makes risk reduction explicit and keeps daily final weights summing to 1.
