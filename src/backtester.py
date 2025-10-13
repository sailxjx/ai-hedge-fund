import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from colorama import Fore, Style

from src.backtesting.data_types import PerformanceMetrics
from src.backtesting.engine import BacktestEngine
from src.cli.input import parse_cli_inputs
from src.main import run_hedge_fund, run_hedge_fund_async, strip_ansi
from src.utils.runtime import async_personas_enabled


class _StdoutTee(io.TextIOBase):
    """Mirror stdout writes while retaining a buffer for log capture."""

    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams
        self._buffer = io.StringIO()

    def write(self, s: str) -> int:  # type: ignore[override]
        for stream in self._streams:
            stream.write(s)
        self._buffer.write(s)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        for stream in self._streams:
            stream.flush()
        self._buffer.flush()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def run_backtest(backtester: BacktestEngine) -> PerformanceMetrics | None:
    """Run the backtest with graceful KeyboardInterrupt handling."""
    try:
        performance_metrics = backtester.run_backtest()
        print(f"\n{Fore.GREEN}Backtest completed successfully!{Style.RESET_ALL}")
        return performance_metrics
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Backtest interrupted by user.{Style.RESET_ALL}")

        # Try to show any partial results that were computed
        try:
            portfolio_values = backtester.get_portfolio_values()
            if len(portfolio_values) > 1:
                print(f"{Fore.GREEN}Partial results available.{Style.RESET_ALL}")

                # Show basic summary from the available portfolio values
                first_value = portfolio_values[0]["Portfolio Value"]
                last_value = portfolio_values[-1]["Portfolio Value"]
                total_return = ((last_value - first_value) / first_value) * 100

                print(f"{Fore.CYAN}Initial Portfolio Value: ${first_value:,.2f}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Final Portfolio Value: ${last_value:,.2f}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Total Return: {total_return:+.2f}%{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Could not generate partial results: {str(e)}{Style.RESET_ALL}")

        sys.exit(0)


### Run the Backtest #####
if __name__ == "__main__":
    inputs = parse_cli_inputs(
        description="Run backtesting simulation",
        require_tickers=False,
        default_months_back=1,
        include_graph_flag=False,
        include_reasoning_flag=False,
    )

    # Create and run the backtester
    agent_callable = run_hedge_fund_async if async_personas_enabled() else run_hedge_fund
    backtester = BacktestEngine(
        agent=agent_callable,
        tickers=inputs.tickers,
        start_date=inputs.start_date,
        end_date=inputs.end_date,
        initial_capital=inputs.initial_cash,
        model_name=inputs.model_name,
        model_provider=inputs.model_provider,
        selected_analysts=inputs.selected_analysts,
        initial_margin_requirement=inputs.margin_requirement,
        portfolio_seed=inputs.portfolio_seed,
        initial_long_pct=inputs.initial_long_pct,
    )

    performance_metrics: PerformanceMetrics | None = None
    if inputs.log_file:
        original_stdout = sys.stdout
        tee = _StdoutTee(original_stdout)
        try:
            with redirect_stdout(tee):
                performance_metrics = run_backtest(backtester)
        finally:
            log_path = Path(inputs.log_file).expanduser()
            try:
                if log_path.parent and not log_path.parent.exists():
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(strip_ansi(tee.getvalue()))
            except OSError as error:
                print(f"Failed to write log file '{log_path}': {error}", file=sys.stderr)
    else:
        performance_metrics = run_backtest(backtester)
