from pathlib import Path

from vpa_structure_recognizer.config import load_config


def test_default_config_matches_spec_windows_and_parent_mapping():
    config = load_config(Path("config/default.toml"))

    assert config.windows == [10, 20, 30, 60, 120, 240]
    assert config.parent_windows == {
        10: [30, 60],
        20: [60],
        30: [120],
        60: [240],
        120: [240],
        240: [240],
    }


def test_default_scoring_weights_sum_to_one():
    config = load_config(Path("config/default.toml"))

    assert sum(config.scoring_weights.values()) == 1.0


def test_default_config_points_at_read_only_external_sources():
    config = load_config(Path("config/default.toml"))

    assert config.sources["research_source"] == "/home/nan/alpha-data-local/output/research_source.duckdb"
    assert config.sources["audited_stock"] == "/home/nan/alpha-data-local/output/research_source.duckdb"
