# matcherror's cap (issue #97)

Why a position's deviation is capped at `maxdeviation` since 1.6, measured on SISAP 2025
`ccnews` with `MaxMatchError` tuned on external queries and scored outside by the external
macro match error over 10,500 held-out queries.

- `matcherror-grid.jl`: targets 0.02 and 0.05, the three transition zones, 64/256 tuning
  queries, 8 seeds. `results/matcherror-grid.log` is the original uncapped run,
  `results/matcherror-grid-capped.log` the run with the cap (the package's behaviour now).
- `matcherror-tail.jl`: the per-query distribution on one tuned configuration, with and
  without the cap.

## What was found

Uncapped, the external match error was 0.03-0.78 against targets of 0.02 and 0.05, its spread
across seeds as large as itself, and both targets tuned to the same configuration. On one
tuned configuration: median 0, p99 0.26, max 143; the top 1% of queries made 92% of the
mean, the top 10 queries 85%. The worst queries had all ten gold neighbors at distance 0
(exact duplicates of the query) and the graph answering at distance 1.4: spread 0, the
floor as denominator, 143 per position.

| `matcherror` | mean | median | p99 | max |
|---|---|---|---|---|
| uncapped, `spreadfloor` 0.01 | 0.114 | 0 | 0.262 | 143.3 |
| uncapped, `spreadfloor` 0.05 | 0.033 | 0 | 0.195 | 28.7 |
| capped at `maxdeviation = 1` | 0.015 | 0 | 0.262 | 1.0 |

With the cap, the external match error lands on the target (0.019 for 0.02 and 0.040 for 0.05
with 256 queries, zone `(-1, 1)`), the two targets tune apart, the zones order as for recall,
and the spread across seeds drops by an order of magnitude.
