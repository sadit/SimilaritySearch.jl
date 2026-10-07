```@meta
CurrentModule = SimilaritySearch
```

# Choosing the queries that tune an index

`optimize_index!` searches for hyperparameters. To compare two configurations it needs queries,
and it needs to know their true neighbors. Which queries it uses decides what it measures. This
page explains the two kinds of query, how to declare which kind you are giving, and what changes
when the index stores codes or folds near duplicates.

## Internal queries and external queries

An **external** query comes from outside the index. A **internal** query is an object the index
already stores.

The difference is not small. An internal query is a vertex of the graph. A search can reach that
vertex at distance 0 and read its adjacency list in one step, and that list is close to the
answer. An external query has no such vertex, so it must reach its neighbors through ordinary
links. If the optimizer tunes with internal queries and does not account for this, it chooses
parameters for an easier problem than the one users pose. Measured on two SISAP 2025 benchmarks,
`bsize` and `Δ` came out at the cheap end of their ranges and recall@10 against real queries fell
from 0.90 to 0.69.

The package handles this by **masking**. Before a tuning search runs, the identifiers of the
query itself are marked as visited, so the search cannot use them. The same identifiers are
removed from the gold standard. Recall stays measurable and the shortcut is removed.

Masking requires knowing which identifiers to mask. That is what you declare.

## Declaring the queries

Two keyword arguments work together:

- `queries`: the objects to search with.
- `queries_identifiers`: where those objects are stored in the index.

Four combinations are accepted:

| `queries` | `queries_identifiers` | result |
|:--|:--|:--|
| `nothing` | `nothing` | `numqueries` identifiers are sampled at random; the queries are those objects |
| `nothing` | identifiers | the queries are the objects at those identifiers |
| objects | `nothing` | the queries are external; nothing is masked |
| objects | identifiers | the objects are the ones at those identifiers; their clusters are masked |

The rule is short: **a query is masked if and only if you give its identifier.** The package does
not infer it from the type of the container.

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 8, 10_000))
G = SearchGraph(Dist.L2(), X)
ctx = SearchGraphContext()
index!(G, ctx)

# external: a validation set that is not in the index
V = MatrixDatabase(rand(Float32, 8, 256))
optimize_index!(G, ctx, MinRecall(0.9); queries=V)

# internal: 256 objects of the index, named by their identifiers
ids = UInt32.(1:256)
optimize_index!(G, ctx, MinRecall(0.9); queries_identifiers=ids)
```

## Why identifiers are separate from objects

For an ordinary `SearchGraph` the identifiers are enough. The objects they name are in
`database(index)`, and that is what the distance function takes.

A quantized index is different. An `AsymmetricSearchGraph` stores codes and compares a raw query
against them. Its database holds codes, so an identifier alone cannot produce a raw query object.
You must pass both: the raw objects, and the identifiers that say where they are stored.

```julia
# the raw objects are not in database(G), which holds codes
optimize_index!(G, ctx, MinRecall(0.9); queries=raw_objects, queries_identifiers=ids)
```

This is also what the package does for you during construction. See the next section.

## The pool, and tuning during construction

`optimize_index!` does not only run when you call it. `OptimizeParameters` is a callback, and the
graph runs it once every so many insertions while the index is being built.

A fixed set of identifiers is set aside once per insertion, and each optimization draws
`numqueries` of them. The set is called the pool and its size is [`TUNINGPOOLSIZE`](@ref).

`numqueries` means how many queries one optimization uses, whatever the source. If you give a set
smaller than `numqueries`, all of it is used. If you want a given set used whole, write
`numqueries=length(queries)`.

Two reasons for a pool instead of a new random sample on every callback:

1. Successive optimizations are scored on the same population, so their results can be compared
   to each other.
2. A `SearchGraph` and an `AsymmetricSearchGraph` tune the same way. This matters when the two are
   compared, because otherwise the tuning procedure changes together with the thing under study.

Identifiers that the graph has not inserted yet are used as external queries for that call. Their
objects exist, and none of them is its own vertex yet, so there is nothing to mask. The pool names
the whole range at the start, and the index reaches it gradually, so the early callbacks tune
mostly on queries from outside the index. Every callback gets `numqueries` queries, however large
the range is.

## Near duplicates change what a mask removes

When the graph folds near duplicates (`Neighborhood(neardup=ϵ)`), the mask removes the query's
whole cluster, not only the query. The representative of a member sits at distance 0 and is the
same shortcut, so it must go as well.

One consequence is worth stating. A held-out set of identifiers is fixed in name but not in
effect. With `k=10`, a query whose cluster holds 11 objects or more loses its entire gold
standard, and such a query is dropped from the tuning set. The same list of identifiers therefore
yields fewer usable queries on a collection with many duplicates than on a clean one. On SISAP
2025 `ccnews`, 7.9% of the objects are in such a cluster.

## What is reported

With `verbose(ctx)` enabled, each optimization reports which queries it used, how many came from
a pool, and how many identifiers were not inserted yet and counted as external.

```@docs
tuningmask
tuningpool
TUNINGPOOLSIZE
```
