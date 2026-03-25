## Search strategy

You are using simulated annealing to avoid getting stuck in local minima.

Maintain a file `sa_state.json` (untracked by git) with:
- `T`: current temperature
- `best_val_bpb`: best seen so far
- `current_val_bpb`: val_bpb of the commit you're currently standing on
- `no_improve_count`: experiments since last improvement
- `experiment_count`: total experiments run

Initial values: T=0.005, best_val_bpb=inf, current_val_bpb=inf, no_improve_count=0, experiment_count=0

After each run:
1. Compute delta = val_bpb_new - current_val_bpb
2. If delta < 0 (improvement): always accept. Update best_val_bpb if new best. Reset no_improve_count=0.
3. If delta >= 0 (worse): compute P = exp(-delta / T). Generate random float 0-1. If < P, accept anyway (log status as `keep-sa`). Otherwise discard.
4. Decay: T = T * 0.97 every experiment.
5. Reheat: if no_improve_count >= 15, set T = 0.003, reset no_improve_count=0.
6. Save updated sa_state.json.

"Accept" means keep the commit and stand on it for the next experiment.
"Discard" means git reset to current_val_bpb commit (not necessarily best).

## Search strategy

You are using a population-based search to maintain diversity.

Maintain a file `population.json` (untracked by git) containing the top 5 experiments:
- Each entry: {commit, val_bpb, description}
- Sorted best to worst.

After setup and baseline run, add baseline to population.

Each experiment:
1. Select parent: 60% chance pick the best, 40% pick randomly from the rest.
2. git checkout <parent commit> before modifying train.py.
3. Make your modification, commit, run.
4. If result would enter top 5: add to population, evict worst.
5. Every 20 experiments: attempt a crossover. Pick two population members, read both diffs vs baseline, try manually combining their changes into one experiment.

Never revert to a single "current best." Always branch from a population member.