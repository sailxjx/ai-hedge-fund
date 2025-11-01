#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Manage arena git worktrees for experimentation.

Usage:
  scripts/worktree_arena.sh create <path> [branch] [base]
  scripts/worktree_arena.sh remove <path>
  scripts/worktree_arena.sh prune
  scripts/worktree_arena.sh list

Commands:
  create  Create a new worktree at <path>. If [branch] is omitted the script
          will create/use a branch named arena/<path>. When the branch does
          not yet exist it is bootstrapped from [base] (default: HEAD).
  remove  Detach and delete the worktree located at <path>. The branch is not
          deleted; remove it manually when no longer required.
  prune   Run `git worktree prune` to clean up stale worktree metadata.
  list    Show existing worktrees (delegates to `git worktree list`).

Examples:
  scripts/worktree_arena.sh create arena-gen1
  scripts/worktree_arena.sh create arena-gen2 arena/gen2 origin/main
  scripts/worktree_arena.sh remove arena-gen1
  scripts/worktree_arena.sh prune

USAGE
}

command=${1:-}
case "$command" in
  create)
    path=${2:-}
    branch=${3:-}
    base=${4:-HEAD}
    if [[ -z "$path" ]]; then
      echo "error: missing worktree path" >&2
      usage
      exit 1
    fi
    if [[ -z "$branch" ]]; then
      sanitized=${path//[^a-zA-Z0-9._-]/-}
      branch="arena/${sanitized}"
    fi
    if git show-ref --verify --quiet "refs/heads/${branch}"; then
      git worktree add "${path}" "${branch}"
    else
      git worktree add -b "${branch}" "${path}" "${base}"
    fi
    ;;
  remove)
    path=${2:-}
    if [[ -z "$path" ]]; then
      echo "error: missing worktree path" >&2
      usage
      exit 1
    fi
    if [[ ! -d "$path" ]]; then
      echo "error: worktree path '$path' does not exist" >&2
      exit 1
    fi
    git worktree remove "${path}"
    ;;
  prune)
    git worktree prune
    ;;
  list)
    git worktree list
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown command '$command'" >&2
    usage
    exit 1
    ;;
esac
