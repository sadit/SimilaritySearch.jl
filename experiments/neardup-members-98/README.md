# Near duplicates as members (issue #98)

The runs behind `Neighborhood(neardup=ϵ)` folding near duplicates into members of one node
per cluster, with `expand`/`expand!` as the second stage of a query. SISAP 2025 `ccnews`,
one graph per variant, 10,500 held-out `itest` queries, fixed beams.

- `neardup-impact.jl`: the census of exact duplicates and three ways of linking them: the
  default (ties keep, cliques), members, and the tie rule alone (monkeypatched: a kept twin
  rejects on `<=`, so a duplicate is a degree-1 node reachable through its twin). Needs `SHA`.
- `members-impact.jl`: the default against members, first stage and expanded, plus the
  tuning path with external queries.
- `results/`: the logs of the original runs (the first before members existed, when
  `neardup=0` only thinned the clique).

## What was found

27.3% of the points are exact copies of another: 45,833 clusters, 1,816 with ten or more
copies, the largest 1,555. 828 of the held-out queries have all ten gold neighbors at
distance 0.

| variant | build | edges | max degree | recall@10 b=8 / b=16 | visits | tied-gold queries (858), b=8 / b=16 |
|---|---|---|---|---|---|---|
| default (ties keep) | 48.8 s | 13.1 M | 1,614 | 0.863 / 0.959 | 544 / 1,212 | 0.724 / 0.745 |
| tie rule alone | 30.9 s | 9.1 M | 1,579 | 0.889 / 0.977 | 557 / 1,232 | 0.859 / 0.892 |
| members, `neardup=0`, expanded | 18.7 s | 8.3 M | 273 | 0.864 / 0.959 | 529 / 1,234 + 18 member evaluations | 0.836 / 0.856 |

The tie rule confirms the mechanism but leaves a star (the 1,555-cluster's representative
collects every reverse link); members remove it, build the graph in a third of the time and
match the default's recall at the same visits. On clusters larger than `k` the file's gold
ids are an arbitrary subset of the copies, so a few expanded queries read 0 while holding
the right cluster; `matcherror` is the score to use there.
