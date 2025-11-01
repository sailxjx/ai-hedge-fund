# Regime Backtest Windows

Canonical date ranges supporting ML diagnostics and sequential backtests:

| Label | Start Date | End Date | Rationale |
| --- | --- | --- | --- |
| crash | 2019-08-01 | 2020-04-30 | Captures the late-2019 drawdown through the Q1 2020 pandemic crash and the volatile rebound inflection. |
| rally | 2020-05-01 | 2021-01-14 | Encompasses the post-crash melt-up through Tesla's mega-cap breakout into early 2021. |
| consolidation | 2021-01-15 | 2021-09-30 | Covers the extended sideways regime with mean-reverting structure following the parabolic rally. |

All diagnostics, dataset exports, and regression backtests should reference these dates unless a new canonical set is agreed upon and documented here.
