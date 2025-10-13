"""Guarded iteration loop for updating agent prompts/code surfaces."""

from __future__ import annotations

import argparse
import inspect
import json
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.utils.analysts import ANALYST_CONFIG

try:  # pragma: no cover - optional imports for supplemental agents
    from src.agents.portfolio_manager import portfolio_management_agent  # type: ignore
    from src.agents.risk_manager import risk_management_agent  # type: ignore
except Exception:  # pragma: no cover - keep CLI usable without optional deps
    risk_management_agent = None  # type: ignore
    portfolio_management_agent = None  # type: ignore


DEFAULT_SMOKE_COMMAND = "poetry run python src/main.py --model-provider azure --analysts-all --tickers TSLA"
DEFAULT_LOG_PATH = Path("log/analysis/agent_iteration_log.jsonl")
DEFAULT_DIAGNOSTICS = Path("log/analysis/latest_llm_combo_diagnostics.json")


def _normalise_agent_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _truncate(text: str, *, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    suffix = "\n... <truncated>"
    return text[: limit - len(suffix)] + suffix


@dataclass
class IterationHypothesis:
    issue: str
    agents: list[str]
    description: str
    recommendation: str


@dataclass
class CommandResult:
    command: str
    status: str
    returncode: int | None
    duration_seconds: float
    stdout: str
    stderr: str


@dataclass
class IterationRecord:
    label: str
    timestamp: str
    hypotheses: list[IterationHypothesis]
    commands: list[CommandResult]
    status: str


class GuardedFiles:
    """Snapshot files and revert them if the iteration fails."""

    def __init__(self, paths: Sequence[Path]):
        self.paths = sorted({path.resolve() for path in paths})
        self._snapshot: dict[Path, str | None] = {}

    def snapshot(self) -> None:
        for path in self.paths:
            if path.is_file():
                self._snapshot[path] = path.read_text(encoding="utf-8", errors="ignore")
            else:
                self._snapshot[path] = None

    def revert(self) -> None:
        for path, content in self._snapshot.items():
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


def build_agent_file_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for config in ANALYST_CONFIG.values():
        display_name = config.get("display_name")
        func = config.get("agent_func")
        if not display_name or func is None:
            continue
        try:
            source = inspect.getsourcefile(func)
        except TypeError:
            source = None
        if not source:
            continue
        mapping[_normalise_agent_name(display_name)] = Path(source).resolve()

    supplemental: dict[str, object] = {
        "riskmanagement": risk_management_agent,
        "portfoliomanager": portfolio_management_agent,
    }
    for key, func in supplemental.items():
        if func is None:
            continue
        try:
            source = inspect.getsourcefile(func)
        except TypeError:
            source = None
        if not source:
            continue
        mapping[key] = Path(source).resolve()

    return mapping


NOTE_HINTS = {
    "RangeRecovery": "Range Recovery Sentinel",
    "ShortSqueeze": "Short Squeeze Guardian",
    "BreakoutCover": "Breakout Cover Sentinel",
    "Momentum": "Momentum Guardian",
    "Trend": "Trend Regime",
    "EventCatalyst": "Event Catalyst",
    "Crash": "Crash Short Allocator",
    "Downside": "Downside Flow Sentinel",
    "Regime": "Regime Meta-Model",
    "Risk": "Risk Management",
}


def _collect_agents(issue_key: str, payload: Mapping[str, object]) -> set[str]:
    agents: set[str] = set()
    for field in (
        "bullish_agents",
        "bearish_agents",
        "guardrails",
        "other_direction",
        "conflicted_agents",
    ):
        values = payload.get(field)
        if isinstance(values, list):
            agents.update(str(value) for value in values)

    if issue_key == "risk_gating_bottlenecks":
        notes = payload.get("constraint_notes")
        if isinstance(notes, list):
            for note in notes:
                if not isinstance(note, str):
                    continue
                for hint, agent in NOTE_HINTS.items():
                    if hint.lower() in note.lower():
                        agents.add(agent)
        agents.add("Risk Management")

    return agents


def _default_recommendation(issue: str, payload: Mapping[str, object], agents: Iterable[str]) -> str:
    agent_names = ", ".join(sorted(agents)) or "the implicated agents"
    if issue == "sentinel_disagreements":
        guards = ", ".join(str(v) for v in payload.get("guardrails", [])) or agent_names
        other = ", ".join(str(v) for v in payload.get("other_direction", [])) or agent_names
        return f"Review guardrail prompts for {guards} so they're compatible with directional signals from {other}; " "consider tempering hard blocks or introducing conditional thresholds."
    if issue == "risk_gating_bottlenecks":
        overrides = payload.get("overrides")
        override_summary = json.dumps(overrides, sort_keys=True) if isinstance(overrides, dict) else "current overrides"
        return f"Relax risk overrides {override_summary} or retune {agent_names} prompts to restore short capacity " "once directional evidence turns."
    if issue == "bias_conflicts":
        return f"Harmonise valuation vs. momentum language across {agent_names} prompts; clarify tie-breaks to avoid persistent splits."
    if issue == "hedge_conflicts":
        majority = payload.get("sentinel_majority", "sentinel cohorts")
        return f"Align portfolio decisions with {majority}; adjust {agent_names} prompts or risk rules so final actions respect sentinel consensus."
    description = payload.get("description") or issue.replace("_", " ")
    return f"Investigate {description} and adjust {agent_names} prompts accordingly."


def generate_hypotheses(entries: Sequence[Mapping[str, object]]) -> list[IterationHypothesis]:
    hypotheses: list[IterationHypothesis] = []
    for entry in entries:
        issues = entry.get("issues")
        if not isinstance(issues, Mapping):
            continue
        for issue_key, payload_list in issues.items():
            if not isinstance(payload_list, list):
                continue
            for payload in payload_list:
                if not isinstance(payload, Mapping):
                    continue
                agents = sorted(_collect_agents(issue_key, payload))
                recommendation = _default_recommendation(issue_key, payload, agents)
                description = str(payload.get("description") or issue_key.replace("_", " "))
                hypotheses.append(
                    IterationHypothesis(
                        issue=issue_key,
                        agents=agents,
                        description=description,
                        recommendation=recommendation,
                    )
                )
    return hypotheses


def load_diagnostics(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {path}")
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        return [data]
    raise ValueError(f"Unexpected diagnostics payload in {path}")


def _render_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    display = _render_command(command)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(env) if env is not None else None,
            check=False,
        )
        status = "success" if proc.returncode == 0 else "failed"
        stdout = _truncate(proc.stdout)
        stderr = _truncate(proc.stderr)
        duration = time.perf_counter() - start
        return CommandResult(display, status, proc.returncode, duration, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = _truncate((exc.stdout or ""))
        stderr = _truncate((exc.stderr or "") + "\n<timeout>")
        return CommandResult(display, "timeout", None, duration, stdout, stderr)


def append_iteration_log(path: Path, record: IterationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def determine_guard_paths(
    hypotheses: Sequence[IterationHypothesis],
    agent_map: Mapping[str, Path],
    extra: Sequence[Path],
) -> list[Path]:
    guard: set[Path] = set()
    for hypothesis in hypotheses:
        for agent in hypothesis.agents:
            key = _normalise_agent_name(agent)
            path = agent_map.get(key)
            if path is not None:
                guard.add(path)
    for extra_path in extra:
        if extra_path.is_dir():
            guard.update(p.resolve() for p in extra_path.rglob("*.py"))
        else:
            guard.add(extra_path.resolve())
    return sorted(guard)


def _list_to_command(command: str) -> list[str]:
    if not command:
        raise ValueError("Command string is empty")
    return shlex.split(command)


def _format_hypotheses(hypotheses: Sequence[IterationHypothesis]) -> str:
    lines = []
    for item in hypotheses:
        lines.append(f"Issue: {item.issue}")
        if item.agents:
            lines.append(f"  Agents: {', '.join(item.agents)}")
        lines.append(f"  Description: {item.description}")
        wrapped = textwrap.fill(item.recommendation, width=88, subsequent_indent="    ")
        lines.append(f"  Recommendation: {wrapped}")
    return "\n".join(lines)


def iteration_loop(
    *,
    diagnostics: Path,
    label: str,
    smoke_command: str,
    pytest_targets: Sequence[str],
    pytest_args: Sequence[str],
    timeout: float | None,
    extra_guard: Sequence[Path],
    log_path: Path,
    stage: bool,
) -> IterationRecord:
    entries = load_diagnostics(diagnostics)
    hypotheses = generate_hypotheses(entries)

    print(_format_hypotheses(hypotheses))
    agent_map = build_agent_file_map()
    guard_paths = determine_guard_paths(hypotheses, agent_map, extra_guard)

    guarded = GuardedFiles(guard_paths)
    guarded.snapshot()

    command_results: list[CommandResult] = []
    status = "success"

    smoke_parts = _list_to_command(smoke_command)
    smoke_result = run_command(smoke_parts, timeout=timeout)
    command_results.append(smoke_result)
    if smoke_result.status != "success":
        status = smoke_result.status
        guarded.revert()
        record = IterationRecord(label, _now_timestamp(), hypotheses, command_results, status)
        append_iteration_log(log_path, record)
        return record

    if pytest_targets:
        pytest_cmd = [sys.executable, "-m", "pytest", *pytest_targets, *pytest_args]
        pytest_result = run_command(pytest_cmd, timeout=timeout)
        command_results.append(pytest_result)
        if pytest_result.status != "success":
            status = pytest_result.status
            guarded.revert()
            record = IterationRecord(label, _now_timestamp(), hypotheses, command_results, status)
            append_iteration_log(log_path, record)
            return record

    if stage and guard_paths:
        git_cmd = ["git", "add", *[str(path.relative_to(Path.cwd())) for path in guard_paths if path.exists()]]
        stage_result = run_command(git_cmd)
        command_results.append(stage_result)
        if stage_result.status != "success":
            status = stage_result.status
            guarded.revert()
            record = IterationRecord(label, _now_timestamp(), hypotheses, command_results, status)
            append_iteration_log(log_path, record)
            return record

    record = IterationRecord(label, _now_timestamp(), hypotheses, command_results, status)
    append_iteration_log(log_path, record)
    return record


def _now_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run guarded iteration loop for agent prompts.")
    parser.add_argument("--label", default="iteration", help="Label for the iteration log entry.")
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=DEFAULT_DIAGNOSTICS,
        help="Path to diagnostics JSON generated by llm_combo_diagnostics.",
    )
    parser.add_argument(
        "--smoke-command",
        default=DEFAULT_SMOKE_COMMAND,
        help="Command to run for the smoke test phase.",
    )
    parser.add_argument(
        "--pytest-target",
        dest="pytest_targets",
        action="append",
        default=[],
        help="Pytest target to run after the smoke test (repeatable).",
    )
    parser.add_argument(
        "--pytest-arg",
        dest="pytest_args",
        action="append",
        default=[],
        help="Additional arguments forwarded to pytest.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Timeout in seconds for each command (default 900).",
    )
    parser.add_argument(
        "--guard",
        dest="guard_paths",
        action="append",
        type=Path,
        default=[],
        help="Additional paths to guard (files or directories).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to append iteration records.",
    )
    parser.add_argument(
        "--no-stage",
        dest="stage",
        action="store_false",
        help="Skip staging guarded files on success.",
    )
    parser.set_defaults(stage=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    extra_guard = [path if path.is_absolute() else (Path.cwd() / path) for path in args.guard_paths]
    record = iteration_loop(
        diagnostics=args.diagnostics if args.diagnostics.is_absolute() else Path.cwd() / args.diagnostics,
        label=args.label,
        smoke_command=args.smoke_command,
        pytest_targets=args.pytest_targets,
        pytest_args=args.pytest_args,
        timeout=args.timeout,
        extra_guard=extra_guard,
        log_path=args.log_path if args.log_path.is_absolute() else Path.cwd() / args.log_path,
        stage=args.stage,
    )
    if record.status != "success":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
