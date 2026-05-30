# VPA Pipeline Contract

## Alpha-Data Boundary

The VPA pipeline consumes alpha-data's normalized stock-bar table,
`stock_bar_normalized_daily`. Alpha-data owns PIT joins, adjusted prices,
tradeability fields, and industry normalization. VPA does not read raw
industry classification tables and does not infer or repair industry codes.

## Required Stock-Bar Columns

Input rows must provide valid core market fields: `open`, `high`, `low`,
`close`, `volume`, and `amount`. The stock-bar contract also includes
`trade_date`, `code`, `prev_close`, `turnover_rate`, `is_st`, `is_paused`,
`limit_up`, `limit_down`, `industry_code`, and `industry_name`.

## UNKNOWN Industry Handling

`industry_code = "UNKNOWN"` and `industry_name = "UNKNOWN"` are legal
alpha-data outputs. VPA keeps these rows for stock-scope analysis, so UNKNOWN
industry stocks can produce:

- `vpa_features`
- `vpa_bar_context_labels`
- `vpa_sequence_stats`
- `vpa_structure_state`

UNKNOWN industry membership only affects industry-level context. If sector
aggregates are built, `UNKNOWN` is kept as a separate observational group.
Top-down ranking does not treat `UNKNOWN` as a real industry peer group for
relative strength. Stocks mapped to UNKNOWN receive neutral sector context and
an `industry_unknown` risk flag instead of being dropped or repaired.
