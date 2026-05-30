# ML Stock Selector Split Readiness

Evaluate a separate repository only after these gates are true:

1. alpha-data normalized bar contract is stable.
2. `vpa_*` schema is stable.
3. `ml_feature_mart_daily` and `ml_predictions_daily` are stable across at least two walk-forward runs.
4. Daily signal reads only normalized bars, `vpa_*` tables, and model artifacts.
5. ML code does not import private VPA implementation modules.
6. Backtest and daily inference use the same prediction and target contracts.
7. Artifact naming and registry activation are stable.

