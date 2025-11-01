# Arena Worktree Guide

Use git worktrees to sandbox evolutionary arena experiments without disrupting the primary development tree.

## Helper Script
`scripts/worktree_arena.sh` wraps common workflows:

```bash
# create ./arena-gen1 on branch arena/arena-gen1 from HEAD
scripts/worktree_arena.sh create arena-gen1

# create ./arena-gen2 on branch arena/gen2 sourced from origin/main
scripts/worktree_arena.sh create arena-gen2 arena/gen2 origin/main

# inspect / prune / remove when finished
scripts/worktree_arena.sh list
scripts/worktree_arena.sh prune
scripts/worktree_arena.sh remove arena-gen1
```

The script will create the branch if it does not exist and leaves branch deletion to the operator when the experiment concludes.

## Recommended Flow
1. Spin a detached worktree per generation or bold experiment (e.g., `arena-gen3`).
2. Run smoke tests + arena schedules inside the worktree, committing or stashing findings locally.
3. Export backtest logs and metadata back into the primary worktree via the shared `log/arena/` directory.
4. Tear down the worktree once the experiment is archived.
