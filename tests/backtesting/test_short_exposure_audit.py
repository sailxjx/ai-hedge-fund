import textwrap

import pytest

from src.backtesting.short_exposure_audit import (
    detect_short_failures,
    ExposureFailure,
    parse_args,
    parse_summary_records,
)


def _write_log(tmp_path, content: str):
    path = tmp_path / "sample.log"
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_summary_records_extracts_daily_snapshots(tmp_path):
    log_text = textwrap.dedent(
        """
        PORTFOLIO SUMMARY:
        Cash Balance: $100,000.00
        Total Position Value: $0.00
        Total Value: $100,000.00
        Portfolio Return: +0.00%
        Information Ratio: 0.00
        Benchmark Return: +0.10%

        | 2024-01-02 | TSLA     |   HOLD   |          0 |  200.00 |             0 |              0 |           0.00 |

        PORTFOLIO SUMMARY:
        Cash Balance: $112,060.00
        Total Position Value: $-12,060.00
        Total Value: $100,000.00
        Portfolio Return: +0.00%
        Information Ratio: 1.25
        Benchmark Return: +0.75%

        | 2024-01-03 | TSLA     |  SHORT   |         60 |  201.00 |             0 |             60 |     -12,060.00 |
        | 2024-01-02 | TSLA     |   HOLD   |          0 |  200.00 |             0 |              0 |           0.00 |
        """
    )
    path = _write_log(tmp_path, log_text)

    records = parse_summary_records("sample", path)

    assert [record.date for record in records] == ["2024-01-02", "2024-01-03"]
    assert records[0].price == 200.0
    assert records[1].short_shares == 60
    assert records[1].position_value == -12060.0
    assert records[1].information_ratio == 1.25
    assert records[1].benchmark_return_pct == 0.0075


def test_detect_short_failures_flags_rising_shorts(tmp_path):
    log_text = textwrap.dedent(
        """
        PORTFOLIO SUMMARY:
        Cash Balance: $100,000.00
        Total Position Value: $0.00
        Total Value: $100,000.00
        Portfolio Return: +0.00%
        Information Ratio: 0.00
        Benchmark Return: +0.10%

        | 2024-01-02 | TSLA     |   HOLD   |          0 |  200.00 |             0 |              0 |           0.00 |

        PORTFOLIO SUMMARY:
        Cash Balance: $112,060.00
        Total Position Value: $-12,060.00
        Total Value: $100,000.00
        Portfolio Return: +0.00%
        Information Ratio: 1.25
        Benchmark Return: +0.75%

        | 2024-01-03 | TSLA     |  SHORT   |         60 |  201.00 |             0 |             60 |     -12,060.00 |
        | 2024-01-02 | TSLA     |   HOLD   |          0 |  200.00 |             0 |              0 |           0.00 |
        """
    )
    path = _write_log(tmp_path, log_text)
    records = parse_summary_records("sample", path)

    failures = detect_short_failures(records)

    assert len(failures) == 1
    failure: ExposureFailure = failures[0]
    assert failure.date == "2024-01-03"
    assert failure.price_change == 1.0
    assert round(failure.exposure_pct or 0, 3) == 0.121
    assert failure.information_ratio == 1.25
    assert failure.benchmark_return_pct == 0.0075

    # High threshold should suppress the flag
    none_found = detect_short_failures(records, min_exposure_pct=0.5)
    assert none_found == []


def test_help_flag_does_not_raise(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"])

    captured = capsys.readouterr().out
    assert "2%" in captured
