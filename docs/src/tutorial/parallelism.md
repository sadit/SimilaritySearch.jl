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

## Writing custom parallel code: `@BATCHES`

If your workload doesn't fit any of the built-in batch functions above, `@BATCHES` is the
primitive they are all built on internally -- prefer it over hand-rolled `Threads.@threads`
(see above) for any new parallel code written in this style.

The simple form splits `range` into consecutive chunks and runs each chunk as one task:

```julia-repl
julia> using SimilaritySearch

julia> n = 100_000; out = zeros(Int, n);

julia> @BATCHES getminbatch(n) for i in 1:n
           out[i] = i^2
       end

julia> out == [i^2 for i in 1:n]
true
```

[`getminbatch`](@ref)`(n)` picks a reasonable chunk size (aiming for a handful of batches
per thread) -- use it instead of hand-picking `minbatch`.

When each batch needs its own scratch state (so concurrent batches never write to the
same memory), `@BATCHES` accepts five optional sections instead of a bare loop:

```julia-repl
julia> function sumsq(n, minbatch)
           local total
           @BATCHES minbatch begin
           @BEGIN
               partial = zeros(Float64, @nbatches())      # one slot per batch
           @BEGINBATCH
               acc = 0.0                                  # this batch's running total
           @LOOP for i in 1:n
               acc += abs2(i)
           end
           @ENDBATCH
               partial[@batchid()] = acc                  # race-free: batch ids are disjoint
           @END
               total = sum(partial)
           end
           total
       end;

julia> sumsq(1000, getminbatch(1000)) == sum(abs2, 1:1000)
true
```

- `@BEGIN` runs once, before any batch starts -- typically to size a shared,
  [`@nbatches()`](@ref)-sized array.
- `@BEGINBATCH` runs once per batch, before that batch's `@LOOP` iterations.
- `@LOOP for i in range ... end` is the only mandatory section: the per-element body.
- `@ENDBATCH` runs once per batch, after that batch's `@LOOP` iterations.
- `@END` runs once, after every batch has joined -- typically to reduce the now-populated
  array from `@BEGIN` (`total = sum(partial)` above).

[`@batchid()`](@ref) is each batch's fixed, 1-based index, stable for that batch's whole
lifetime -- indexing a shared array by it (`partial[@batchid()]` above) is race-free by
construction, since no two concurrently-running batches ever share one; prefer it over
`Threads.threadid()`, which can alias or migrate mid-batch under some schedulers (below).

Always write `@batchid()`/`@nbatches()` with the explicit, empty parentheses shown above,
even though they take no arguments: it means exactly the same thing as the bare macro
call, but a bare `@batchid` directly followed by a unary `-` parses as `@batchid` being
*passed* `-1` as an argument, not as subtraction on its result -- `2 * @batchid - 1` parses
as `2 * @batchid(-1)`, an error. `@batchid()` cannot be misparsed that way.

`@BATCHES` dispatches batches via `Threads.@threads`, under a scheduler that defaults to
`:static` (one task per thread, never migrates) and can be overridden globally with
[`set_batch_scheduler!`](@ref) or per call with a `scheduler=` keyword. If your batch body
captures a shared handle (e.g. a `SearchGraphContext`) and re-derives per-batch state from
it several calls deep, read the [`@BATCHES`](@ref) docstring's tagged-handle hazard warning
before writing that pattern yourself -- it's a real bug this package's own code hit once.

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
