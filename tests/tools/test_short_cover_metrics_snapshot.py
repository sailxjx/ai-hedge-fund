from src.tools.short_cover_metrics_snapshot import Snapshot, _to_dataframe


def test_snapshot_to_flat_dict_and_dataframe(tmp_path):
    snapshots = [
        Snapshot(
            date="2025-07-01",
            signal="squeeze_cover",
            confidence=80,
            reasoning="Test reasoning",
            metrics={
                "probability": 0.25,
                "threshold": 0.23,
                "raw_probability": 0.22,
                "improvement": 0.004,
                "improvement_threshold_effective": 0.002,
            },
        ),
        Snapshot(
            date="2025-07-02",
            signal="trim_short",
            confidence=60,
            reasoning="Second",
            metrics={
                "probability": 0.18,
                "threshold": 0.22,
                "raw_probability": 0.19,
                "improvement": 0.001,
                "improvement_threshold_effective": 0.003,
            },
        ),
    ]

    flat = [snap.to_flat_dict() for snap in snapshots]
    assert flat[0]["probability"] == 0.25
    assert flat[0]["signal"] == "squeeze_cover"

    df = _to_dataframe(snapshots)
    assert list(df.index) == ["2025-07-01", "2025-07-02"]
    assert abs(df.loc["2025-07-01", "probability_gap"] - 0.02) < 1e-9
    assert abs(df.loc["2025-07-01", "boost_delta_check"] - 0.03) < 1e-9
    assert abs(df.loc["2025-07-01", "improvement_margin"] - 0.002) < 1e-9
    assert abs(df.loc["2025-07-02", "improvement_margin"] + 0.002) < 1e-9
    assert abs(df.loc["2025-07-02", "boost_delta_check"] + 0.01) < 1e-9
    assert abs(df.loc["2025-07-02", "probability_gap"] + 0.04) < 1e-9
