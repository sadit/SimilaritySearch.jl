# The goals' objective for 1.6: tradeoff, width and the hinge's shape

The runs behind the redesign of `MinRecall`/`MaxMatchError`'s objective (`goalvalue`: the log
cost plus a finite-support hinge on the target; see README's 1.6 notes). One `SearchGraph`
over SISAP 2025 `ccnews` (603,664 x 384), tuned with external queries drawn from the first
500 of `itest`, scored on the held-out `itest` queries (501 on) and 500 of `otest` by
recall@10 and visits per query, 8 seeds per cell.

- `tradeoff-grid.jl`: `tradeoff` x `width` x `numqueries` for a target of 0.9. It was run
  against the first, softplus hinge; `width=1e-4` is the hard threshold of old.
- `hinge-grid.jl`: the hinge's transition zone, centered / below / above the target, over
  three tradeoffs. Originally it compared softplus against the finite-support hinge by
  monkeypatching; softplus was dropped and the script now takes the zones from the API.
- `results/`: the per-run CSVs and the logs with the summary tables of the original runs.

## What was found

Softplus overshoots the target and the overshoot grows with the tradeoff, because its tail
never reaches zero (at the target it still charges `log(2) · width`):

| hinge | tuning queries | tradeoff 1.2 / 1.5 / 3 | visits |
|---|---|---|---|
| softplus | 64 | 0.910 / 0.922 / 0.939 | 648 / 719 / 892 |
| softplus | 256 | 0.913 / 0.921 / 0.931 | 643 / 692 / 773 |
| finite support, centered `(-1, 1)` | 64 | 0.900 / 0.907 / 0.912 | 615 / 640 / 682 |
| finite support, centered `(-1, 1)` | 256 | 0.907 / 0.911 / 0.913 | 621 / 636 / 649 |
| finite support, below `(0, 2)` | 64 | 0.877 / 0.882 / 0.888 | 550 / 563 / 581 |
| finite support, below `(0, 2)` | 256 | 0.892 / 0.896 / 0.901 | 579 / 594 / 603 |
| hard threshold (`width=1e-4`) | 64 / 256 | 0.893-0.905 / 0.899-0.905 | 602-644 / 619-638 |

The centered finite-support hinge lands within a width above the target and barely moves
with the tradeoff between 1.2 and 3, which is why `(-1, 1)` and `tradeoff=1.5` are the
defaults. The spread across seeds is set by the number of tuning queries (0.016 at 64,
0.005 at 256), not by the hinge. `otest` sits 0.05-0.06 below `itest` for every
configuration.
