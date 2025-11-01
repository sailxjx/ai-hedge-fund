# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

This system employs several agents working together:

1. Aswath Damodaran Agent - The Dean of Valuation, focuses on story, numbers, and disciplined valuation
2. Ben Graham Agent - The godfather of value investing, only buys hidden gems with a margin of safety
3. Bill Ackman Agent - An activist investor, takes bold positions and pushes for change
4. Cathie Wood Agent - The queen of growth investing, believes in the power of innovation and disruption
5. Charlie Munger Agent - Warren Buffett's partner, only buys wonderful businesses at fair prices
6. Michael Burry Agent - The Big Short contrarian who hunts for deep value
7. Mohnish Pabrai Agent - The Dhandho investor, who looks for doubles at low risk
8. Peter Lynch Agent - Practical investor who seeks "ten-baggers" in everyday businesses
9. Phil Fisher Agent - Meticulous growth investor who uses deep "scuttlebutt" research 
10. Rakesh Jhunjhunwala Agent - The Big Bull of India
11. Stanley Druckenmiller Agent - Macro legend who hunts for asymmetric opportunities with growth potential
12. Warren Buffett Agent - The oracle of Omaha, seeks wonderful companies at a fair price
13. Valuation Agent - Calculates the intrinsic value of a stock and generates trading signals
14. Sentiment Agent - Analyzes market sentiment and generates trading signals
15. Fundamentals Agent - Analyzes fundamental data and generates trading signals
16. Technicals Agent - Analyzes technical indicators and generates trading signals
17. Risk Manager - Calculates risk metrics and sets position limits
18. Portfolio Manager - Makes final trading decisions and generates orders

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application](#️-web-application)
- [How to Contribute](#how-to-contribute)
- [Feature Requests](#feature-requests)
- [License](#license)

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.

### 1. Clone the Repository

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Set up API keys

Create a `.env` file for your API keys and optional tuning knobs:
```bash
# Create .env file for your API keys (in the root directory)
cp .env.example .env
```

Open and edit the `.env` file to add your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=your-openai-api-key

# For getting financial data to power the hedge fund
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key

# Optional: constrain company-news fetches to avoid long hangs
# Seconds spent gathering paginated news per ticker (default 60)
COMPANY_NEWS_FETCH_TIMEOUT_SECONDS=60
# Maximum news pages per ticker (default 8)
COMPANY_NEWS_MAX_PAGES=8
```

**Important**: You must set at least one LLM API key (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`) for the hedge fund to work. 

**Financial Data**: Data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key. For any other ticker, you will need to set the `FINANCIAL_DATASETS_API_KEY` in the .env file.

## How to Run

### Observation-First Personas

All analyst agents now emit raw observations and delegate trade intent to their anthropomorphic personas. When extending the platform, focus on enriching the observation payloads instead of encoding rule-based signals. See `src/agents/persona_utils.py::persona_from_observations` for the shared helper that every analyst uses to package metrics for the LLM personas. Backtest summaries now include turnover rate so you can quickly sanity-check how much gross notional the persona ensemble traded over the window.

### Async Persona Parallelism

Async persona execution is **enabled by default**. The CLI (`src/main.py`), the LangGraph backend, and the Python backtester automatically dispatch analyst, risk, and portfolio personas through the async graph. Set `ASYNC_PERSONAS=0` (or `false`, `off`) to fall back to the legacy synchronous execution when bisecting regressions. The async engine fans out LLM work in parallel while guarding shared state with `src/utils/async_state.py` and concurrency semaphores in `src/utils/llm.py`.

See `docs/async_parallelism.md` for a deeper walkthrough covering scheduler flags, telemetry payloads, and the validation loop.

- Async telemetry: pass `--include-async-telemetry` to `poetry run python -m src.backtesting.experiment_scheduler` (or set `include_async_telemetry` in your schedule JSON) to append timing metadata columns to `log/backtest.csv`. The scheduler will automatically locate summaries under `log/backtest_timings/` and record concurrency ratio, semaphore utilization, and the raw summary path alongside the usual performance metrics.

- CLI smoke (async is default):
  ```bash
  poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA
  ```
- Backtester async run:
  ```bash
  poetry run python src/backtester.py --tickers TSLA,NVDA --start-date 2024-10-01 --end-date 2024-10-03
  ```
- Legacy sync fallback (optional):
  ```bash
  ASYNC_PERSONAS=0 poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA
  ```
- Backend + web app inherit the default; export `ASYNC_PERSONAS=0` before starting `uvicorn` only when you need to force sync dispatch.

Configurability:
- `LLM_ASYNC_MAX_CONCURRENCY` caps simultaneous persona LLM calls (default `8`).
- `LLM_CALL_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` cover async retries just like the sync helpers.
- `ASYNC_PERSONAS=0` explicitly forces the legacy synchronous flow (set `1`/`true` to re-enable async if you previously disabled it).

Regression coverage lives in `tests/agents/test_async_persona.py`, `tests/agents/test_async_wrappers.py`, `tests/backtesting/test_async_engine.py`, and `tests/backend/test_graph_service.py`. Run them locally with:
```bash
poetry run pytest tests/agents/test_async_persona.py \
  tests/agents/test_async_wrappers.py \
  tests/backtesting/test_async_engine.py \
  tests/backend/test_graph_service.py
```

### ⌨️ Command Line Interface

You can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### Quick Start

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

#### Run the AI Hedge Fund
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

You can optionally specify the start and end dates to make decisions over a specific time period.

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

#### Run the Backtester
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


Note: The `--ollama`, `--start-date`, and `--end-date` flags work for the backtester, as well!
Add `--portfolio-seed path/to/snapshot.json` to seed the run with a preexisting portfolio state (e.g., short hangover replays).

#### Summarize Backtest Logs
```bash
poetry run python -m src.backtesting.evaluate_logs \
  --logs baseline=log/backtest_baseline_crash.log candidate=log/backtest_ml_crash.log \
  --overrides baseline=log/risk_overrides/backtest_baseline_crash.jsonl candidate=log/risk_overrides/backtest_ml_crash.jsonl \
  --output log/backtest_metrics/summary.csv
```
Use `--json` to emit machine-readable summaries capturing Sharpe/Sortino/return plus override counts for regression tracking.


### 🖥️ Web Application

The new way to run the AI Hedge Fund is through our web application that provides a user-friendly interface. This is recommended for users who prefer visual interfaces over command line tools.

Please see detailed instructions on how to install and run the web application [here](https://github.com/virattt/ai-hedge-fund/tree/main/app).

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />


## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
