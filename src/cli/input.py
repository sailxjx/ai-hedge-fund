import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import questionary
from colorama import Fore, Style
from dateutil.relativedelta import relativedelta

from src.llm.models import get_model_info, LLM_ORDER, ModelProvider, OLLAMA_LLM_ORDER
from src.utils.analysts import ANALYST_ORDER
from src.utils.ollama import ensure_ollama_and_model


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    require_tickers: bool = False,
    include_analyst_flags: bool = True,
    include_ollama: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tickers",
        type=str,
        required=require_tickers,
        help="Comma-separated list of stock ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    if include_analyst_flags:
        parser.add_argument(
            "--analysts",
            type=str,
            required=False,
            help="Comma-separated list of analysts to use (e.g., michael_burry,other_analyst)",
        )
        parser.add_argument(
            "--analysts-all",
            action="store_true",
            help="Use all available analysts (overrides --analysts)",
        )
    if include_ollama:
        parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")
    parser.add_argument(
        "--model",
        type=str,
        help="Specify model selection; use provider:model format or pair with --model-provider.",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        help="Model provider to use when specifying --model without provider prefix.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Optional path to write run output (ANSI codes stripped).",
    )
    return parser


def add_date_args(parser: argparse.ArgumentParser, *, default_months_back: int | None = None) -> argparse.ArgumentParser:
    if default_months_back is None:
        parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
        parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    else:
        parser.add_argument(
            "--end-date",
            type=str,
            default=datetime.now().strftime("%Y-%m-%d"),
            help="End date in YYYY-MM-DD format",
        )
        parser.add_argument(
            "--start-date",
            type=str,
            default=(datetime.now() - relativedelta(months=default_months_back)).strftime("%Y-%m-%d"),
            help="Start date in YYYY-MM-DD format",
        )
    return parser


def parse_tickers(tickers_arg: str | None) -> list[str]:
    if not tickers_arg:
        return []
    return [ticker.strip() for ticker in tickers_arg.split(",") if ticker.strip()]


def select_analysts(flags: dict | None = None) -> list[str]:
    if flags and flags.get("analysts_all"):
        return [a[1] for a in ANALYST_ORDER]

    if flags and flags.get("analysts"):
        return [a.strip() for a in flags["analysts"].split(",") if a.strip()]

    # Default to all analysts when running non-interactively so automation doesn't hang
    if not sys.stdin.isatty():
        return [a[1] for a in ANALYST_ORDER]

    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done.",
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)

    print(f"\nSelected analysts: {', '.join(Fore.GREEN + c.title().replace('_', ' ') + Style.RESET_ALL for c in choices)}\n")
    return choices


def select_model(use_ollama: bool) -> tuple[str, str]:
    model_name: str = ""
    model_provider: str | None = None

    if use_ollama:
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")
        model_name = questionary.select(
            "Select your Ollama model:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_name:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        if model_name == "-":
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            sys.exit(1)

        model_provider = ModelProvider.OLLAMA.value
        print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    else:
        model_choice = questionary.select(
            "Select your LLM model:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
        ).ask()

        if not model_choice:
            print("\n\nInterrupt received. Exiting...")
            sys.exit(0)

        model_name, model_provider = model_choice

        model_info = get_model_info(model_name, model_provider)
        if model_info and model_info.is_custom():
            model_name = questionary.text("Enter the custom model name:").ask()
            if not model_name:
                print("\n\nInterrupt received. Exiting...")
                sys.exit(0)

        if model_info:
            print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        else:
            model_provider = "Unknown"
            print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    return model_name, model_provider or ""


def _normalize_provider(provider: str) -> str:
    """Resolve a provider string to a ModelProvider value."""
    normalized = provider.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    provider_aliases = {
        "azure": ModelProvider.AZURE_OPENAI.value,
        "azureopenai": ModelProvider.AZURE_OPENAI.value,
    }
    if normalized in provider_aliases:
        return provider_aliases[normalized]
    for candidate in ModelProvider:
        candidate_value = candidate.value.lower().replace(" ", "")
        candidate_name = candidate.name.lower().replace("_", "")
        if normalized == candidate_value or normalized == candidate_name:
            return candidate.value
    raise ValueError(f"Unknown model provider '{provider}'.")


def _find_model_entry(identifier: str) -> tuple[str, str] | None:
    """Search known model lists by display name or model name."""
    ident = identifier.strip().lower()
    for display, name, provider in [*LLM_ORDER, *OLLAMA_LLM_ORDER]:
        if display.lower() == ident or name.lower() == ident:
            return name, provider
    return None


def _parse_model_option(model_option: str, provider_option: str | None) -> tuple[str, str]:
    """Parse the --model/--model-provider flags into a concrete selection."""
    model_token = model_option.strip()
    provider_token = provider_option.strip() if provider_option else None

    if ":" in model_token:
        prefix, suffix = model_token.split(":", 1)
        if not provider_token:
            provider_token = prefix
        model_token = suffix

    if provider_token:
        provider_value = _normalize_provider(provider_token)
    else:
        match = _find_model_entry(model_token)
        if not match:
            raise ValueError("Unable to determine provider for the chosen model. Use provider:model format or supply --model-provider.")
        resolved_name, resolved_provider = match
        provider_value = resolved_provider
        # Prefer catalog model name unless empty (e.g., Azure deployment entry)
        if resolved_name:
            model_token = resolved_name

    if not model_token:
        if provider_value == ModelProvider.AZURE_OPENAI.value:
            model_token = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "").strip()
            if not model_token:
                raise ValueError("Azure OpenAI selection requires a deployment name. Provide one via provider:model or set AZURE_OPENAI_DEPLOYMENT_NAME.")
        else:
            raise ValueError("Model name cannot be empty. Provide one via provider:model format.")

    return model_token, provider_value


def resolve_dates(start_date: str | None, end_date: str | None, *, default_months_back: int | None = None) -> tuple[str, str]:
    if start_date:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")
    if end_date:
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")

    final_end = end_date or datetime.now().strftime("%Y-%m-%d")
    if start_date:
        final_start = start_date
    else:
        months = default_months_back if default_months_back is not None else 3
        end_date_obj = datetime.strptime(final_end, "%Y-%m-%d")
        final_start = (end_date_obj - relativedelta(months=months)).strftime("%Y-%m-%d")
    return final_start, final_end


@dataclass
class CLIInputs:
    tickers: list[str]
    selected_analysts: list[str]
    model_name: str
    model_provider: str
    start_date: str
    end_date: str
    initial_cash: float
    margin_requirement: float
    initial_long_pct: float = 0.0
    show_reasoning: bool = False
    show_agent_graph: bool = False
    log_file: Optional[str] = None
    raw_args: Optional[argparse.Namespace] = None
    portfolio_seed: Optional[dict[str, Any]] = None


def parse_cli_inputs(
    *,
    description: str,
    require_tickers: bool,
    default_months_back: int | None,
    include_graph_flag: bool = False,
    include_reasoning_flag: bool = False,
) -> CLIInputs:
    parser = argparse.ArgumentParser(description=description)

    # Common/interactive flags
    add_common_args(parser, require_tickers=require_tickers, include_analyst_flags=True, include_ollama=True)
    add_date_args(parser, default_months_back=default_months_back)

    # Funding flags (standardized, with alias)
    parser.add_argument(
        "--initial-cash",
        "--initial-capital",
        dest="initial_cash",
        type=float,
        default=100000.0,
        help="Initial cash position (alias: --initial-capital). Defaults to 100000.0",
    )
    parser.add_argument(
        "--margin-requirement",
        dest="margin_requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement ratio for shorts (e.g., 0.5 for 50%%). Defaults to 0.0",
    )

    parser.add_argument(
        "--portfolio-seed",
        dest="portfolio_seed",
        type=str,
        help="Path to a JSON file containing an initial portfolio snapshot.",
    )

    parser.add_argument(
        "--initial-long-pct",
        dest="initial_long_pct",
        type=float,
        default=0.0,
        help="Percentage of initial cash to allocate to long positions at the start of the run (0-100).",
    )

    if include_reasoning_flag:
        parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    if include_graph_flag:
        parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")

    args = parser.parse_args()

    # Normalize parsed values
    tickers = parse_tickers(getattr(args, "tickers", None))
    selected_analysts = select_analysts(
        {
            "analysts_all": getattr(args, "analysts_all", False),
            "analysts": getattr(args, "analysts", None),
        }
    )

    use_ollama = getattr(args, "ollama", False)
    if getattr(args, "model", None) or getattr(args, "model_provider", None):
        try:
            model_name, model_provider = _parse_model_option(getattr(args, "model", "") or "", getattr(args, "model_provider", None))
        except ValueError as exc:
            parser.error(str(exc))
        if model_provider == ModelProvider.OLLAMA.value:
            use_ollama = True
    else:
        model_name, model_provider = select_model(use_ollama)
    start_date, end_date = resolve_dates(getattr(args, "start_date", None), getattr(args, "end_date", None), default_months_back=default_months_back)

    seed_snapshot = None
    seed_path = getattr(args, "portfolio_seed", None)
    if seed_path:
        try:
            with open(seed_path, "r", encoding="utf-8") as handle:
                seed_snapshot = json.load(handle)
        except FileNotFoundError as exc:
            parser.error(f"Portfolio seed file not found: {seed_path}")
        except json.JSONDecodeError as exc:
            parser.error(f"Portfolio seed file '{seed_path}' is not valid JSON: {exc}")
        except OSError as exc:
            parser.error(f"Failed to read portfolio seed '{seed_path}': {exc}")

    return CLIInputs(
        tickers=tickers,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
        start_date=start_date,
        end_date=end_date,
        initial_cash=getattr(args, "initial_cash", 100000.0),
        margin_requirement=getattr(args, "margin_requirement", 0.0),
        initial_long_pct=min(100.0, max(0.0, getattr(args, "initial_long_pct", 0.0))),
        show_reasoning=getattr(args, "show_reasoning", False),
        show_agent_graph=getattr(args, "show_agent_graph", False),
        log_file=getattr(args, "log_file", None),
        raw_args=args,
        portfolio_seed=seed_snapshot,
    )
