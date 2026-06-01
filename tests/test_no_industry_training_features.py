from __future__ import annotations

import pandas as pd

from ml_stock_selector.feature_matrix import build_feature_matrix


def test_no_industry_features_when_deny_enabled():
    frame = pd.DataFrame(
        {
            "features_json": [
                "{\"industry_code\":\"I1\",\"ret_5\":0.1}",
                "{\"industry_code\":\"I2\",\"ret_5\":0.2}",
            ]
        }
    )
    matrix, _ = build_feature_matrix(frame, "vpa_d_sequence", fit=True, deny_industry=True)
    assert "ret_5" in matrix.columns
    assert not any("industry" in col for col in matrix.columns)

