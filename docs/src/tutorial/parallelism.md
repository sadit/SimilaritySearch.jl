```@meta
CurrentModule = SimilaritySearch
```

# Parallelism: what to expect, what not to do

Every batch operation in this tutorial (`searchbatch`, `allknn`, `closestpair`,
`neardup`, `index!`, ...) is already internally parallel, using this package's own
`@BATCHES` macro on top of `Threads.@threads`. You never need to write your own
threading code to get that parallelism -- and, as this page explains, writing your own
threading code *around* these functions is where things go wrong.

## You must start Julia with more than one thread

```sh
julia -t8 --project=.        # 8 threads
```

If you don't, `Threads.nthreads() == 1`, and every batch function above silently takes
its serial fast path -- no error, no warning, it just runs on one thread. This is the
single most common way to conclude "parallelism didn't help" when actually parallelism
never ran at all:

```julia-repl
julia> Threads.nthreads()
1   # started as `julia --project=.`, not `julia -t8 --project=.`
```

Check `Threads.nthreads()` first whenever a benchmark looks slower than expected.

## The right way to parallelize: use the batch functions

```julia
using SimilaritySearch, Distances

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 5000))
Q = MatrixDatabase(rand(Float32, 4, 500))
G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)                       # parallel across the dataset, internally

knns = searchbatch(G, ctx, Q, 8)     # parallel across the 500 queries, internally
```

That's it -- `searchbatch`/`allknn`/`index!`/etc. all split their work into batches and
dispatch them across `Threads.nthreads()` tasks for you, correctly. This is the
supported, safe way to parallelize a query workload.

## What *not* to do: hand-rolled `Threads.@threads` over `search`

You may see code (including in older tutorials/demos for this package) that instead
loops over queries by hand:

```julia
# DON'T DO THIS
res = [knnqueue(ctx, 8) for _ in eachindex(Q)]
Threads.@threads for i in eachindex(Q)
    search(G, ctx, Q[i], res[i])   # every task shares the *same* ctx
end
```

This *looks* reasonable -- each task gets its own query and its own result buffer -- but
every task is passed the exact same `ctx` object. `SearchGraphContext` holds small
internal scratch caches (visited-vertex state, beam buffers) that `search` reuses across
calls for performance; `searchbatch`/`allknn` internally hand each parallel batch its own
tagged copy of `ctx` so those caches never collide, but a raw `search(G, ctx, ...)` call
has no way to know it's one of several concurrent calls sharing one `ctx` -- it always
uses the same scratch slot. Multiple threads calling `search` on the same shared `ctx`
concurrently means multiple threads writing to that same slot concurrently: a data race,
silently producing wrong or inconsistent results (not a crash you can rely on seeing).

The fix is simply: don't hand-roll this loop. Use [`searchbatch`](@ref)/`searchbatch!`
(shown above), which already does exactly this in a way that's actually safe. If you
have a genuine reason to run many independent `search` calls concurrently outside of
`searchbatch`, give each concurrent task its own context (`SearchGraphContext()` is cheap
to create) rather than sharing one:

```julia
# OK: each task has its own context, no sharing
Threads.@threads for i in eachindex(Q)
    search(G, SearchGraphContext(), Q[i], res[i])
end
```

...though at that point you're just reimplementing what `searchbatch` already does for
you, with none of its batching/memory-reuse benefits -- prefer `searchbatch` whenever
your workload fits its shape (a batch of independent queries against one index).

## Don't nest two parallel index types

[`SimilaritySearch.Exact.ParallelExhaustiveSearch`](@ref) parallelizes *within* a single
query (splitting the dataset comparisons for that one query across threads). If you then
also run it through `searchbatch` (which parallelizes *across* queries), both layers
compete for the same fixed pool of `Threads.nthreads()` threads at once, which typically
makes things slower, not faster, rather than actually running "more" parallelism. Use
either `ParallelExhaustiveSearch` alone (best when you have very few queries, or even
just one, and want to parallelize that single query), or a plain `ExhaustiveSearch`
through `searchbatch` (best when you have many queries and want to parallelize across
them) -- not both together.

## Capping memory for very large datasets

Every parallel batch operation allocates a small amount of scratch memory per batch, not
per thread and not per element -- the number of batches is chosen automatically (roughly
proportional to `Threads.nthreads()`, not to dataset size), and stays modest regardless
of how large your dataset is. If you ever do need to cap it further (e.g. a machine with
many threads and a very large per-batch buffer), `SearchGraphContext`/`GenericContext`
accept a `maxbatches` keyword:

```julia
ctx = SearchGraphContext(; maxbatches=32)   # hard ceiling on internal batch count
```

Lowering this can only reduce parallelism/increase batch size, never correctness --
unlike the shared-context anti-pattern above, this knob is always safe to adjust.

Next: [saving and loading indexes with JLD2](persistence.md).
