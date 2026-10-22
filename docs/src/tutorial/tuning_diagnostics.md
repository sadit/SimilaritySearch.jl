```@meta
CurrentModule = SimilaritySearch
```

# Checking a tuning

`optimize_index!` chooses `bsize`, `Δ` and `maxvisits` for a `SearchGraph`. It measures candidate
configurations on a sample of queries and installs the best one. Most of the time the result
needs no attention. This page shows how to read what the tuner measured. It also shows how to
recognize two situations where the result is not what you want, and what to do in each:

- the tuning stopped far from the target, and
- the queries you serve do not look like the queries the tuner used.

This page complements [Choosing the queries that tune an index](tuning_queries.md), which
explains masking and how to declare internal and external queries.

## A running example

The examples use 100,000 Gaussian vectors in 32 dimensions and two query sets. The first query
set comes from the same distribution as the data. The second one is shifted by 1.5 in every
coordinate, so its queries lie away from the data.

```julia
using SimilaritySearch, Random, Statistics

Random.seed!(0)
dim, n = 32, 100_000
X = MatrixDatabase(randn(Float32, dim, n))
Qin = MatrixDatabase(randn(Float32, dim, 1000))            # like the data
Qout = MatrixDatabase(randn(Float32, dim, 1000) .+ 1.5f0)  # away from the data

dist = Dist.L2()
ctx = SearchGraphContext()
G = SearchGraph(dist, X)
index!(G, ctx)
```

The numbers in the comments below come from one run of this page, in this order, with 16
threads. Your numbers will differ a little.

## Reading a tuning

`optimize_index!` returns the final population of its search over configurations, best first.
Each element is a pair `configuration => performance`. The tuner installs the first configuration
in the index.

```julia
best = optimize_index!(G, ctx, MinRecall(0.9))
config, perf = first(best)
perf.recall      # recall on the tuning queries: 0.913
perf.recallstd   # the standard error of that recall: 0.013
perf.visited     # distance evaluations per query on them: (min = 1755, mean = 2967.5, max = 4880)
G.algo[]         # the installed configuration: BeamSearch(bsize=12, Δ=1.1025, maxvisits=9760)
```

The tuner measures recall on a small sample, 64 queries by default, so its estimate has an error.
`perf.recallstd` is the standard error of that estimate. The reported recall may lie a little
below the target. The recall on your queries may differ from the reported one by a few standard
errors. This difference is the expected precision of the tuner, not a failure.

`maxvisits` is twice the largest number of distance evaluations that the installed configuration
needed on the sample. It bounds the cost of a single query: the index cuts a search that needs
more.

## Recognizing a failed tuning

A tuning has failed when the reported recall is far below the target, by much more than a few
standard errors. One way to cause it is an initial range of `bsize` and `Δ` that is far too small
for the data:

```julia
bad = optimize_index!(G, ctx, MinRecall(0.9); space=BeamSearchSpace(; bsize=2:2:4, Δ=0.6:0.025:0.7))
last(first(bad)).recall   # 0.021
G.algo[]                  # BeamSearch(bsize=4, Δ=0.70875, maxvisits=664)
```

Every configuration of the initial population then has a recall close to zero. The goal is almost
flat around them, its improvement falls below the tolerance, and the search stops early. The tuner
still installs the best configuration it evaluated, although that configuration is useless here.
On this index, the range above failed in ten of ten draws. A range with only `Δ` between 0.6 and
0.8 failed in two of ten. The library also warns with `OPT low recall` when a configuration it
evaluates has a recall below 0.3.

The tuner reports what it measured, so the failure is easy to detect. The index still holds the
failed configuration. The function below repeats the tuning until the reported recall is within
three standard errors of the target, at most three times:

```julia
function repeat_if_failed(G, ctx, result, target; attempts=3)
    for attempt in 1:attempts
        config, perf = first(result)
        perf.recall >= target - 3 * perf.recallstd && return result
        result = optimize_index!(G, ctx, MinRecall(target))   # repeat with the default space
    end
    result
end

best = repeat_if_failed(G, ctx, bad, 0.9)
last(first(best)).recall   # 0.922
G.algo[]                   # BeamSearch(bsize=9, Δ=1.12, maxvisits=7976)
```

In this run the first repetition also stopped short, at a recall of 0.686, and the second one
reached 0.922. The search stops when the worst configuration of its population stops improving.
This can happen before the best configuration reaches the target. A new draw of the initial
population usually reaches it: on this index, ten repetitions with fresh draws all passed the
check.

The function repeats with the default space. `BeamSearchSpace()` starts with `bsize` between 2
and 16 and `Δ` between 0.9 and 1.1. Its mutations can move `bsize` up to 20 and `Δ` up to 1.75.
Two other changes help when the default space is not enough:

- **A wider initial range**, for instance `BeamSearchSpace(; bsize=2:2:32, Δ=0.9:0.05:1.5)`. It
  helps when the data needs a larger beam or more slack than the default range starts with.
- **More tuning queries**, for instance `numqueries=256`. The standard error falls with the
  square root of the number of queries. The tuning also costs more, because the tuner computes
  the true neighbors of every tuning query by exhaustive search.

A wide range can also stop far from the target. In that case, the distances of the data may be
too concentrated for a graph search at that target.

## Queries from another distribution

By default, the tuning queries are objects of the index, masked from their own search. They
follow the distribution of the data. When the queries you serve come from somewhere else, the
tuned configuration can deliver less than the target on them.

```julia
k = 10
ex = ExhaustiveSearch(dist, X)
gold_in, _ = searchbatch(ex, ctx, Qin, k)
gold_out, _ = searchbatch(ex, ctx, Qout, k)

R, _ = searchbatch(G, ctx, Qin, k)
macrorecall(gold_in, R)    # 0.930
R, _ = searchbatch(G, ctx, Qout, k)
macrorecall(gold_out, R)   # 0.847
```

### Noticing it

The first check needs no true neighbors. Compare the distance to the `k`-th neighbor that the
index finds for your queries with the same distance for objects of the index:

```julia
S = SubDatabase(X, rand(1:n, 1000))
_, DS = searchbatch(G, ctx, S, k + 1)   # an indexed object finds itself first, so ask for one more
_, Din = searchbatch(G, ctx, Qin, k)
_, Dout = searchbatch(G, ctx, Qout, k)
median(DS[end, :]), median(Din[end, :]), median(Dout[end, :])   # (4.81, 4.86, 8.17)
```

The queries like the data find their neighbors at the same distance as the objects of the index
do. The shifted queries find them much farther away. Such queries lie in regions where the graph
has fewer vertices and fewer useful links. The same configuration then reaches less of their
answer.

The second check is direct. Compute the true neighbors of a few hundred of your queries with
`ExhaustiveSearch`, and measure the recall as in the block above. This check costs one exhaustive
search per query.

### Tuning with your queries

Give the tuner a sample of your queries, and keep the rest to measure the result:

```julia
sample = SubDatabase(Qout, 1:256)
heldout = 257:1000
best = optimize_index!(G, ctx, MinRecall(0.9); queries=sample, numqueries=length(sample))
G.algo[]   # BeamSearch(bsize=8, Δ=1.155, maxvisits=22030)

R, _ = searchbatch(G, ctx, SubDatabase(Qout, heldout), k)
macrorecall(gold_out[:, heldout], R)   # 0.907
```

Three points matter here:

- The sample has no identifiers, so its queries are external and the tuner masks nothing.
  [Choosing the queries that tune an index](tuning_queries.md) explains the difference.
- The tuner computes the true neighbors of the sample by exhaustive search. The cost of the tuning
  therefore grows with the size of the sample.
- Measure on queries that were not in the sample, as above. Otherwise you measure how well the
  configuration fits the sample, not your workload.

The installed configuration serves every query. Here its `maxvisits` grew from 7976 to 22030,
because the shifted queries need more distance evaluations. If your workload mixes both kinds of
query, tune on a sample of the mix.

## When to tune again

Tuning is cheap compared with construction. In our measurements it took between 0.2% and 1.1% of
the building time. Tune again when the queries you serve change, or when one of the checks above
shows a drop. While the index grows, `OptimizeParameters` tunes it as part of construction. A
later call is needed only to set the final configuration or to adapt it to a new workload.
