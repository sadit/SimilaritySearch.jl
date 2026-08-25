```@meta
CurrentModule = SimilaritySearch
```

# Parallelism and Multithreading

Batch operations in `SimilaritySearch.jl` (including `searchbatch`, `allknn`, `closestpair`, `neardup`, and `index!`) are parallelized internally using the package's `@BATCHES` multithreading macro built on top of Julia's native `Threads.@threads`.

---

## Configuring Julia Threads

Multithreading requires launching Julia with multiple execution threads:

```sh
julia -t auto --project=.   # Automatically use available CPU cores
# or
julia -t 8 --project=.      # Allocate 8 worker threads
```

When `Threads.nthreads() == 1`, batch operations execute via a single-threaded path. Always verify thread availability during performance benchmarking:

```julia-repl
julia> Threads.nthreads()
8
```

---

## Standard Parallel Execution: Batch APIs

The primary and recommended method for executing parallel operations is through the high-level batch functions:

```julia
using SimilaritySearch, Distances

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 5000))
Q = MatrixDatabase(rand(Float32, 4, 500))

G = SearchGraph(dist, X)
ctx = SearchGraphContext()

# Internally parallelized across dataset objects
index!(G, ctx)

# Internally parallelized across the 500 queries
knns = searchbatch(G, ctx, Q, 8)
```

Functions such as `searchbatch`, `allknn`, and `index!` partition the workload into disjoint batches and schedule them across available threads.

---

## Thread Safety and Context Sharing

A common anti-pattern involves wrapping individual `search` calls inside a manual `Threads.@threads` loop with a single shared context:

```julia
# INCORRECT: Introduces data races across threads
res = [knnqueue(ctx, 8) for _ in eachindex(Q)]
Threads.@threads for i in eachindex(Q)
    search(G, ctx, Q[i], res[i])   # Multiple threads write to the same ctx scratch buffers
end
```

### The Mechanism of the Data Race

To maximize query throughput, [`SearchGraphContext`](@ref) maintains pre-allocated, reusable scratch buffers:
- Visited vertex states (`vstates`).
- Beam traversal queues (`beams`).

When `search` is called concurrently using a single shared `ctx` instance, multiple worker threads write simultaneously to the same internal buffer slots, causing silent state corruption.

### Correct Approaches

1. **Preferred**: Use [`searchbatch`](@ref) or [`searchbatch!`](@ref), which automatically assigns isolated scratch slots to each batch.
2. **Manual Task Isolation**: If writing custom concurrent loops, assign an independent context instance to each concurrent task:

```julia
# CORRECT: Distinct context per concurrent task
Threads.@threads for i in eachindex(Q)
    search(G, SearchGraphContext(), Q[i], res[i])
end
```

---

## Custom Parallel Loops with `@BATCHES`

For parallel algorithms not covered by built-in batch functions, `SimilaritySearch.jl` exports the `@BATCHES` macro.

### Simple Loop Form

The simple form splits a range into contiguous chunks:

```julia
using SimilaritySearch

n = 100_000
out = zeros(Int, n)

@BATCHES getminbatch(n) for i in 1:n
    out[i] = i^2
end
```

[`getminbatch`](@ref)`(n)` computes an optimal batch size based on `Threads.nthreads()`.

### Structured Batch Form with Scratch State

When each batch requires isolated intermediate state, use the structured sections:

```julia
function sum_squares(n::Integer, minbatch::Integer)
    local total
    @BATCHES minbatch begin
    @BEGIN
        # Runs once before task dispatch; allocate per-batch reduction buffers
        partial = zeros(Float64, @nbatches())
    @BEGINBATCH
        # Runs once per batch before processing elements
        acc = 0.0
    @LOOP for i in 1:n
        acc += abs2(i)
    @ENDBATCH
        # Runs once per batch upon completion; write to disjoint slot
        partial[@batchid()] = acc
    @END
        # Runs once after all batches join
        total = sum(partial)
    end
    total
end
```

### Safety Rule: Indexing with `@batchid()`, not `threadid()`

- **Always index per-batch scratch buffers by `@batchid()`**, which returns a stable, disjoint integer in $\{1, \dots, \text{@nbatches()}\}$. This prevents race conditions under dynamic and work-stealing schedulers.
- **Do not index by `Threads.threadid()`**, as tasks can migrate across threads or share thread IDs under non-static schedulers.
- Always include parentheses (`@batchid()`, `@nbatches()`) to prevent macro parsing ambiguities with following arithmetic operators.

---

## Batch Schedulers

`@BATCHES` supports multiple execution schedulers:
- `:static`: Partitions iterations evenly across threads without task migration.
- `:default`: Standard Julia task scheduler.
- `:greedy`: Dynamic work-stealing scheduler (Julia $\ge$ 1.11).
- `:sequential`: Disables multithreading, running iterations sequentially in the caller task. Useful for deterministic debugging and benchmarking.

### Context-Level Scheduler Configuration

[`GenericContext`](@ref) and [`SearchGraphContext`](@ref) store a `scheduler` field (defaulting to [`get_batch_scheduler`](@ref)). All internal batch operations routed through that context adopt its scheduler:

```julia
# Enforce serial execution across all operations using this context
ctx = SearchGraphContext(; scheduler=:sequential)
index!(G, ctx)
knns = searchbatch(G, ctx, Q, 8)
```

---

## Avoiding Nested Parallel Index Types

[`SimilaritySearch.Exact.ParallelExhaustiveSearch`](@ref) parallelizes distance evaluations *within* a single query across threads.

Do not combine `ParallelExhaustiveSearch` with `searchbatch` (which parallelizes *across* queries). Doing so causes nested task over-subscription and thread contention. Use:
- `ParallelExhaustiveSearch` for executing individual queries when $|Q| \approx 1$.
- `ExhaustiveSearch` with `searchbatch` when processing multiple queries simultaneously ($|Q| \gg 1$).

---

## Bounding Memory with `maxbatches`

Scratch buffer memory scales with the number of active batches. To enforce a strict ceiling on buffer allocation in high-core environments, set `maxbatches`:

```julia
ctx = SearchGraphContext(; maxbatches=32)
```

---

In the next section, [Index Persistence and Serialization](persistence.md), we discuss saving and loading search indexes using JLD2.
