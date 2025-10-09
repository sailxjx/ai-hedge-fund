#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<USAGE >&2
Usage: $(basename "$0") -- <codex exec arguments...>
       $(basename "$0") stop
       $(basename "$0") status

Examples:
  $(basename "$0") --prompt "bootstrap" --model gpt-4
  $(basename "$0") stop

Environment variables:
  CODEX_MONITOR_PATTERN  pattern matched by pgrep (default "codex exec")
  CODEX_MONITOR_LOG      log path for codex stdout/stderr (default /tmp/codex_exec.log)
  CODEX_MONITOR_WATCHER_LOG log path for watcher messages (default /tmp/codex_watcher.log)
  CODEX_MONITOR_INTERVAL polling interval in seconds (default 10)
  CODEX_MONITOR_PIDFILE  pidfile location (default /tmp/codex_monitor.pid)
USAGE
}

SCRIPT_NAME=$(basename "$0")
MODE="run"

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

case "$1" in
    __daemon)
        MODE="__daemon"
        shift
        ;;
    stop)
        MODE="stop"
        shift
        ;;
    status)
        MODE="status"
        shift
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    --)
        shift
        ;;
    *)
        MODE="run"
        ;;
esac

PGREP_PATTERN="${CODEX_MONITOR_PATTERN:-codex exec}"
LOG_PATH="${CODEX_MONITOR_LOG:-/tmp/codex_exec.log}"
WATCHER_LOG="${CODEX_MONITOR_WATCHER_LOG:-/tmp/codex_watcher.log}"
INTERVAL="${CODEX_MONITOR_INTERVAL:-10}"
PIDFILE="${CODEX_MONITOR_PIDFILE:-/tmp/codex_monitor.pid}"

stop_watcher() {
    if [[ ! -f "$PIDFILE" ]]; then
        echo "No watcher pidfile at $PIDFILE."
        return 0
    fi
    watcher_pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -z "$watcher_pid" ]]; then
        echo "Watcher pidfile $PIDFILE is empty."
        rm -f "$PIDFILE"
        return 0
    fi
    if ! kill -0 "$watcher_pid" >/dev/null 2>&1; then
        echo "No running watcher with PID $watcher_pid; removing stale pidfile $PIDFILE."
        rm -f "$PIDFILE"
        return 0
    fi
    echo "Stopping watcher PID $watcher_pid."
    kill "$watcher_pid" >/dev/null 2>&1 || true
    for _ in {1..20}; do
        if ! kill -0 "$watcher_pid" >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
    if kill -0 "$watcher_pid" >/dev/null 2>&1; then
        echo "Watcher PID $watcher_pid did not exit; sending SIGKILL."
        kill -9 "$watcher_pid" >/dev/null 2>&1 || true
    fi
    rm -f "$PIDFILE"
    echo "Watcher stopped."
    return 0
}

stop_codex_processes() {
    mapfile -t codex_pids < <(pgrep -f "$PGREP_PATTERN" 2>/dev/null || true)

    if [[ ${#codex_pids[@]} -eq 0 ]]; then
        echo "No codex exec process matched pattern '$PGREP_PATTERN'."
        return 1
    fi

    echo "Stopping codex exec processes (pattern '$PGREP_PATTERN'): ${codex_pids[*]}"
    kill "${codex_pids[@]}" >/dev/null 2>&1 || true

    for _ in {1..20}; do
        local still_running=0
        for pid in "${codex_pids[@]}"; do
            if kill -0 "$pid" >/dev/null 2>&1; then
                still_running=1
                break
            fi
        done
        if [[ $still_running -eq 0 ]]; then
            break
        fi
        sleep 0.1
    done

    local -a survivors=()
    for pid in "${codex_pids[@]}"; do
        if kill -0 "$pid" >/dev/null 2>&1; then
            survivors+=("$pid")
        fi
    done

    if [[ ${#survivors[@]} -gt 0 ]]; then
        echo "codex exec PIDs ${survivors[*]} did not exit; sending SIGKILL."
        kill -9 "${survivors[@]}" >/dev/null 2>&1 || true
    fi

    echo "codex exec processes stopped."
    return 0
}

status_watcher() {
    if [[ -f "$PIDFILE" ]]; then
        watcher_pid=$(cat "$PIDFILE" 2>/dev/null || true)
        if [[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" >/dev/null 2>&1; then
            echo "Watcher running with PID $watcher_pid (pidfile $PIDFILE)."
        else
            echo "Watcher pidfile $PIDFILE exists but process missing."
        fi
    else
        echo "No watcher pidfile found (expected $PIDFILE)."
    fi
    if pgrep -f "$PGREP_PATTERN" >/dev/null 2>&1; then
        echo "codex exec process detected (pattern $PGREP_PATTERN)."
    else
        echo "No codex exec process detected (pattern $PGREP_PATTERN)."
    fi
}

log_msg() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    printf '[%s] %s\n' "$timestamp" "$*" >> "$WATCHER_LOG"
    if [[ -n "${CODEX_MONITOR_VERBOSE:-}" ]]; then
        printf '[%s] %s\n' "$timestamp" "$*"
    fi
}

run_watcher() {
    local -a args=("$@")
    local cmd_pretty child_pid

    if [[ ${#args[@]} -eq 0 ]]; then
        echo "Error: no codex exec arguments provided to watcher." >&2
        exit 1
    fi

    if ! command -v codex >/dev/null 2>&1; then
        echo "Error: codex command not found in PATH." >&2
        exit 1
    fi

    if ! mkdir -p "$(dirname "$WATCHER_LOG")" >/dev/null 2>&1; then
        echo "Error: unable to create watcher log directory $(dirname "$WATCHER_LOG")." >&2
        exit 1
    fi
    if ! touch "$WATCHER_LOG" >/dev/null 2>&1; then
        echo "Error: unable to write to watcher log $WATCHER_LOG." >&2
        exit 1
    fi

    cmd_pretty=$(printf '%q ' "${args[@]}")
    cmd_pretty=${cmd_pretty% }

    trap 'rm -f "$PIDFILE"; exit 0' EXIT INT TERM
    trap '' HUP

    printf '%s\n' "$$" > "$PIDFILE"

    log_msg "Watcher $$ started (interval ${INTERVAL}s, pattern ${PGREP_PATTERN})."

    while true; do
        if ! pgrep -f "$PGREP_PATTERN" >/dev/null 2>&1; then
            log_msg "No codex process found. Starting: codex exec ${cmd_pretty}"
            nohup codex exec "${args[@]}" >> "$LOG_PATH" 2>&1 &
            child_pid=$!
            log_msg "codex exec started with PID ${child_pid}. Logs -> ${LOG_PATH}"
        fi
        sleep "$INTERVAL"
    done
}

if [[ "$MODE" == "stop" ]]; then
    stop_status=0
    stop_watcher || stop_status=$?
    stop_codex_processes || true
    exit $stop_status
fi

if [[ "$MODE" == "status" ]]; then
    status_watcher
    exit 0
fi

if [[ "$MODE" == "__daemon" ]]; then
    CMD_ARGS=("$@")
    run_watcher "${CMD_ARGS[@]}"
    exit 0
fi

if ! command -v codex >/dev/null 2>&1; then
    echo "Error: codex command not found in PATH." >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

if [[ -f "$PIDFILE" ]]; then
    existing_pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" >/dev/null 2>&1; then
        echo "Codex watcher already running with PID $existing_pid (pidfile: $PIDFILE)."
        exit 0
    fi
    echo "Removing stale pidfile $PIDFILE."
    rm -f "$PIDFILE"
fi

CMD_ARGS=("$@")

if command -v setsid >/dev/null 2>&1; then
    setsid -f "$0" __daemon "${CMD_ARGS[@]}" >/dev/null 2>&1 &
else
    nohup "$0" __daemon "${CMD_ARGS[@]}" >/dev/null 2>&1 &
fi

printf 'Codex watcher launching (pidfile %s, watcher log %s).\n' "$PIDFILE" "$WATCHER_LOG"

for _ in {1..50}; do
    if [[ -f "$PIDFILE" ]]; then
        watcher_pid=$(cat "$PIDFILE" 2>/dev/null || true)
        if [[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" >/dev/null 2>&1; then
            printf 'Codex watcher running with PID %s (interval %ss, codex log %s, watcher log %s).\n' "$watcher_pid" "$INTERVAL" "$LOG_PATH" "$WATCHER_LOG"
            exit 0
        fi
    fi
    sleep 0.1
done

echo "Watcher launch in progress; run '$SCRIPT_NAME status' to verify." >&2
exit 0
