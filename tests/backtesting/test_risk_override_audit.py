from __future__ import annotations

import json
from pathlib import Path

from src.backtesting.risk_override_audit import (
    _load_overrides,
    audit_overrides,
    main,
)
from src.backtesting.short_exposure_audit import parse_summary_records

LOG_SAMPLE = """PORTFOLIO SUMMARY:
Cash Balance: $100,000.00
Total Position Value: $0.00
Total Value: $100,000.00
Portfolio Return: +0.00%

+------------+----------+----------+------------+---------+---------------+----------------+------------------+
| Date       | Ticker   |  Action  |   Quantity |   Price |   Long Shares |   Short Shares |   Position Value |
+============+==========+==========+============+=========+===============+================+==================+
| 2019-08-01 | TSLA     |   BUY    |        388 |   15.59 |           388 |              0 |         6,048.92 |
+------------+----------+----------+------------+---------+---------------+----------------+------------------+

PORTFOLIO SUMMARY:
Cash Balance: $93,999.00
Total Position Value: $6,107.42
Total Value: $100,106.42
Portfolio Return: +0.11%

+------------+----------+----------+------------+---------+---------------+----------------+------------------+
| Date       | Ticker   |  Action  |   Quantity |   Price |   Long Shares |   Short Shares |   Position Value |
+============+==========+==========+============+=========+===============+================+==================+
| 2019-08-02 | TSLA     |   BUY    |          3 |   15.62 |           391 |              0 |         6,107.42 |
+------------+----------+----------+------------+---------+---------------+----------------+------------------+

PORTFOLIO SUMMARY:
Cash Balance: $88,000.00
Total Position Value: $4,537.86
Total Value: $92,537.86
Portfolio Return: -0.07%

+------------+----------+----------+------------+---------+---------------+----------------+------------------+
| Date       | Ticker   |  Action  |   Quantity |   Price |   Long Shares |   Short Shares |   Position Value |
+============+==========+==========+============+=========+===============+================+==================+
| 2019-08-27 | TSLA     |   COVER  |        956 |   14.27 |           318 |              0 |         4,537.86 |
+------------+----------+----------+------------+---------+---------------+----------------+------------------+
"""


def _write_sample_files(tmp_path: Path) -> tuple[Path, Path]:
    log_path = tmp_path / "backtest_ml_crash.log"
    log_path.write_text(LOG_SAMPLE, encoding="utf-8")

    overrides = [
        {
            "end_date": "2019-08-01",
            "risk_snapshot": {
                "overrides": {
                    "target_long_shares": 388,
                    "preferred_direction": "long",
                }
            },
        },
        {
            "end_date": "2019-08-02",
            "risk_snapshot": {
                "overrides": {
                    "target_long_shares": 392,
                    "max_additional_short_shares": 1,
                }
            },
        },
        {
            "end_date": "2019-08-27",
            "risk_snapshot": {
                "overrides": {
                    "target_short_shares": 956,
                    "force_cover_qty": 956,
                    "block_new_shorts": True,
                    "preferred_direction": "long",
                }
            },
        },
    ]
    override_path = tmp_path / "risk_overrides.jsonl"
    override_path.write_text(
        "\n".join(json.dumps(entry) for entry in overrides) + "\n",
        encoding="utf-8",
    )
    return log_path, override_path


def test_audit_overrides_detects_mismatches(tmp_path: Path) -> None:
    log_path, override_path = _write_sample_files(tmp_path)

    records = parse_summary_records("audit", log_path)
    trades = {record.date: record for record in records}
    overrides = _load_overrides(override_path)

    checks = audit_overrides(overrides, trades)
    assert len(checks) == 3

    assert checks[0].compliant
    assert not checks[1].compliant
    assert checks[2].compliant
    assert any("long_shares" in issue for issue in checks[1].issues)


def test_main_reports_failures(tmp_path: Path) -> None:
    log_path, override_path = _write_sample_files(tmp_path)
    exit_code = main([
        "--log",
        str(log_path),
        "--overrides",
        str(override_path),
    ])
    assert exit_code == 1
