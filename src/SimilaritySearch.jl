# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type AbstractSearchIndex end
using Accessors

include("parallel.jl")

import Base: push!, append!
using Statistics: mean
export AbstractSearchIndex, AbstractContext, GenericContext, ExhaustiveSearch,
    search, searchbatch, searchbatch!, database, distance,
    SearchResult, push_item!, append_items!, getminbatch,
    IdDist, Dist, Exact, Special, ScalarQuant, Intersections, InvertedFiles,
    distance_evaluations, block_evaluations, distance_stats, block_stats,
    KCenters, fft, dnet, randsel, multirandsel,
    Bichromatic, closestpair, bichromatic_closestpair, closestpairs, bichromatic_kclosestpairs,
    bichromatic_metricjoin,
    PQueue, AbstractMetricQueue, AbstractKnnQueue, AbstractRadiusQueue,
    KnnHeap, KnnSorted, RadiusSorted, RadiusHeap, knnqueue,
    covradius, maxlength, reuse!, sortitems!, pop_max!, pop_min!, nearest, frontier,
    DistView, IdView, IdDistView, knn_matrices

"""
    abstract type AbstractContext end

Base type for context objects (e.g. [`GenericContext`](@ref), [`SearchGraphContext`](@ref)):
per-call configuration, hyperparameters, caches, and a logger, passed alongside an index to
[`search`](@ref), [`searchbatch`](@ref), [`index!`](@ref), and similar functions.
"""
abstract type AbstractContext end
function searchbatch! end
function search end
function push_item! end
function append_items! end
function index! end
function reuse! end
function knnqueue end

"""
    getminbatch(n::Int, nt::Int=Threads.nthreads();
                blocks_per_thread::Int=4, maxbatches::Int=n)

The official, always-valid way to compute a `minbatch` size for [`@BATCHES`](@ref). Always
returns a value `>= 1` (or `n` itself when `n <= 0` or `nt <= 1`), so callers do not need
to clamp its result themselves.

`maxbatches` is a plain `Int`, deliberately with **no special/sentinel value** (no `0`
meaning "off", no `Union{Nothing,Int}`) -- it is simply a hard ceiling on the batch count,
always in effect, and defaults to `n` because `n` is already the largest a batch count
could ever sensibly be (a batch needs >= 1 element, so more than `n` batches is
meaningless). That default is therefore a genuine no-op, not a disguised "disabled" flag:
whatever `blocks_per_thread * nt` computes is used as-is. Pass anything smaller and it
takes direct, immediate effect on the result -- there is nothing else to know. This also
keeps the function fully type-stable/monomorphic and total (every `Int`, including `0` or
negative, produces a well-defined result; see below), cheap to call from
performance-sensitive code.

- `blocks_per_thread` (default `4`): the natural batch-count target is
  `blocks_per_thread * nt` -- always tied to the thread count, never an
  independent/arbitrary number.
- `maxbatches` (default `n`, i.e. no effective restriction): a **hard ceiling** on the
  batch count, overriding the natural target above whenever it would be larger. Use this
  to directly bound the memory of per-batch scratch allocations (e.g. [`@BATCHES`](@ref)'s
  `@BEGIN`-declared, [`@nbatches()`](@ref)-sized buffers) for very large `n`. When a context
  object is available, prefer the `getminbatch(ctx::AbstractContext, n)` overload
  (`searchgraph/context.jl`) instead, which derives this from `ctx.maxbatches`.

!!! warning "Extreme cases / contraindications"
    - `maxbatches < nt`: some threads get **no work at all** (`@BATCHES` only dispatches
      `nbatches` tasks; if `nbatches < nthreads()` the remaining threads sit idle).
      Deliberately trading away parallelism for memory -- know that you're doing it.
    - `maxbatches` very small (e.g. `1`, or even `0`/negative -- all collapse to the same
      single-batch result) with large `n`: essentially serial execution despite having
      many threads. Only sensible when per-batch memory, not speed, is the dominant
      constraint.
    - Fewer, larger batches worsen load-balancing under uneven per-element cost: one
      batch can straggle while others finish early and idle -- the classic
      granularity-vs-balance trade-off, and exactly why the default target is
      `blocks_per_thread=8`, not `1`.
    - `maxbatches` has no effect once it exceeds `n` (a batch needs >= 1 element; the
      result is already clamped to at most `n` batches regardless) -- this is exactly why
      `n` is the default: it is the natural "no restriction" value.
    - A large `maxbatches`/small `blocks_per_thread` combination can still land inside
      `@BATCHES`'s own small-`n` fast path (`n <= minbatch` -> single serial batch, no
      threading at all) -- consistent, not a bug, but easy to trip over unexpectedly.

# Arguments
- `n`: the number of elements to process
- `nt`: number of threads to use (defaults to `Threads.nthreads()`)

# Keyword Arguments
- `blocks_per_thread`: target batches per thread (default `8`)
- `maxbatches`: hard cap on the total batch count, for bounding per-batch memory directly
  regardless of `nt` (defaults to `n`, a no-op unless set to something smaller)
"""
function getminbatch(n::Int, nt::Int=Threads.nthreads();
                      blocks_per_thread::Int=4, maxbatches::Int=n)
    n <= 0 && return 1
    nt <= 1 && return n

    nblocks = blocks_per_thread * nt
    nblocks = min(nblocks, maxbatches)
    nblocks = clamp(nblocks, 1, n)

    max(1, ceil(Int, n / nblocks))
end

using Distances: Metric, SemiMetric, PreMetric, evaluate
include("dist/Dist.jl")

#using .Dist  # keep as a separate module

include("db/db.jl")
include("sq/sq.jl")
include("distsample.jl")
include("iddist.jl")
include("adj/Adj.jl")
include("intersections/Intersections.jl")
include("special/special.jl")
include("proj/Projections.jl")

#using .Adj

include("log.jl")
include("pqueue/pqueue.jl")
using .PQueue

@inline Base.length(searchctx::AbstractSearchIndex) = length(database(searchctx))
@inline Base.eachindex(searchctx::AbstractSearchIndex) = 1:length(searchctx)
@inline Base.eltype(searchctx::AbstractSearchIndex) = eltype(database(searchctx))

"""
    database(index)

Gets the entire indexed database
"""
@inline database(searchctx::AbstractSearchIndex) = searchctx.db

"""
    database(index, i)

Gets the i-th object from the indexed database
"""
@inline database(searchctx::AbstractSearchIndex, i) = getindex(database(searchctx), i)
@inline Base.getindex(searchctx::AbstractSearchIndex, i::Integer) = database(searchctx, i)


"""
    distance(index)

Gets the distance function used in the index
"""
@inline distance(searchctx::AbstractSearchIndex) = searchctx.dist

"""
    GenericContext(KnnType::Type{<:AbstractKnnQueue}=KnnSorted;
        verbose::Bool=true, logger=InformativeLog(),
        maxbatches::Integer=8Threads.nthreads(), batchid::Integer=1,
        scheduler::Symbol=get_batch_scheduler()) -> GenericContext

Lightweight [`AbstractContext`](@ref) implementation used by exact indexes
([`ExhaustiveSearch`](@ref), [`ParallelExhaustiveSearch`](@ref)) that need no per-thread
scratch caches.

# Keyword Arguments
- `verbose`: controls the number of output messages.
- `logger`: how to handle and log events.
- `maxbatches`: hard cap on the batch count used by [`getminbatch`](@ref) for operations
  driven by this context (e.g. [`searchbatch!`](@ref), [`allknn`](@ref),
  [`closestpair`](@ref), [`search`](@ref search(::ParallelExhaustiveSearch, ::GenericContext, ::Any, ::AbstractKnnQueue))).
  Defaults to `8 * Threads.nthreads()`, matching [`getminbatch`](@ref)'s own default
  `blocks_per_thread`.
- `batchid`: the batch slot this context is tagged with; not meaningful on the root
  context returned here (always `1`) -- per-batch copies tagging the running `@batchid()`
  are minted internally via `Accessors.@set`, one per batch, not per call.
- `scheduler`: the [`@BATCHES`](@ref) scheduler used by every `@BATCHES` call driven by this
  context (passed through as `scheduler=ctx.scheduler`). Defaults to whatever
  [`get_batch_scheduler`](@ref) currently returns, captured once at construction time (later
  calls to [`set_batch_scheduler!`](@ref) do not retroactively change an already-built
  context). Pass `scheduler=:sequential` to force every `@BATCHES` call driven by this
  context to run unthreaded, regardless of `Threads.nthreads()`.
- `costdists`/`costblocks`: per-batch distance/block-evaluation counters (size `maxbatches`,
  indexed by `batchid`), accumulated via `add_distance_evaluations!`/`add_block_evaluations!`
  and read via [`distance_evaluations`](@ref)/[`distance_stats`](@ref) and their block
  counterparts. Never reset automatically -- they accumulate for the lifetime of the context.
"""
struct GenericContext{KnnType} <: AbstractContext
    verbose::Bool
    logger
    maxbatches::Int32
    batchid::Int32
    scheduler::Symbol
    costdists::Vector{Int}
    costblocks::Vector{Int}
end

GenericContext(KnnType::Type{<:AbstractKnnQueue}=KnnSorted; verbose::Bool=true, logger=InformativeLog(),
    maxbatches::Integer=8Threads.nthreads(), batchid::Integer=1, scheduler::Symbol=get_batch_scheduler(),
    costdists=zeros(Int, maxbatches), costblocks=zeros(Int, maxbatches)) =
    GenericContext{KnnType}(verbose, logger, convert(Int32, maxbatches), convert(Int32, batchid), scheduler, costdists, costblocks)

# GenericContext has a phantom type parameter (KnnType, not derivable from any field), so
# ConstructionBase's default reconstruction (used by Accessors.@set) can't infer it -- this
# override makes `@set ctx.batchid = ...`/`@set ctx.maxbatches = ...` work.
Accessors.ConstructionBase.constructorof(::Type{<:GenericContext{K}}) where {K} = (args...) -> GenericContext{K}(args...)

knnqueue(::GenericContext{KnnType}, args...) where {KnnType<:AbstractKnnQueue} = knnqueue(KnnType, args...)
verbose(ctx::GenericContext) = ctx.verbose

# A slot counts toward these stats if it's nonzero -- a real search always performs >= 1
# evaluation, so 0 reliably means "never touched" (lifetime form) / "untouched since the
# snapshot" (diff form below) -- true regardless of how large the raw cumulative value is.
function _batchstats(v::AbstractVector{Int})
    active = filter(!iszero, v)
    isempty(active) && return (min=0, mean=0.0, max=0)
    (min=minimum(active), mean=mean(active), max=maximum(active))
end

@inline add_distance_evaluations!(ctx::AbstractContext, v) = (ctx.costdists[ctx.batchid] += v)
@inline add_block_evaluations!(ctx::AbstractContext, v) = (ctx.costblocks[ctx.batchid] += v)

"""
    distance_stats(ctx::AbstractContext) -> (; min, mean, max)
    distance_stats(ctx::AbstractContext, snapshot::Vector{Int}) -> (; min, mean, max)

Min/mean/max distance evaluations across `ctx`'s active batch slots, at batch granularity
(not query granularity). The 1-arg form reads `ctx.costdists` as-is: since it's never reset,
this is a **lifetime** statistic (since `ctx` was created). The 2-arg form measures a single
operation instead: pass a `snapshot = copy(ctx.costdists)` taken before that operation, and
this diffs the *raw* vectors (`ctx.costdists .- snapshot`) before computing stats on the
result -- **not** the other way around. Diffing precomputed stats instead is mathematically
wrong for `min`/`max` (only `mean`/`sum` commute with subtraction) and would also break
"active slot" detection, since a slot's raw value is cumulative garbage from every prior
call -- only the delta reliably reflects whether that slot was touched during the snapshot
window.

Note: reads `ctx.costdists` with no synchronization -- only call this once every `@BATCHES`
region writing into `ctx` has already joined (i.e. from sequential code).
"""
distance_stats(ctx::AbstractContext) = _batchstats(ctx.costdists)
distance_stats(ctx::AbstractContext, snapshot::Vector{Int}) = _batchstats(ctx.costdists .- snapshot)

"Total distance evaluations across all batch slots (1-arg: lifetime; 2-arg: diffed against `snapshot`, see [`distance_stats`](@ref))."
distance_evaluations(ctx::AbstractContext) = sum(ctx.costdists)
distance_evaluations(ctx::AbstractContext, snapshot::Vector{Int}) = sum(ctx.costdists .- snapshot)

"Block-evaluations counterpart of [`distance_stats`](@ref) (1-arg or 2-arg form)."
block_stats(ctx::AbstractContext) = _batchstats(ctx.costblocks)
block_stats(ctx::AbstractContext, snapshot::Vector{Int}) = _batchstats(ctx.costblocks .- snapshot)

"Total block evaluations across all batch slots (1-arg: lifetime; 2-arg: diffed against `snapshot`, see [`block_stats`](@ref))."
block_evaluations(ctx::AbstractContext) = sum(ctx.costblocks)
block_evaluations(ctx::AbstractContext, snapshot::Vector{Int}) = sum(ctx.costblocks .- snapshot)

include("perf.jl")
include("exact/Exact.jl")

using SimilaritySearch.Exact
include("kcenters/KCenters.jl")
using .KCenters

using SimilaritySearch.Exact

function Base.show(io::IO, idx::AbstractSearchIndex; prefix="", indent="  ")
    println(io, prefix, typeof(idx), ":")
    prefix = prefix * indent
    println(io, prefix, "dist: ", typeof(idx.dist))
    println(io, prefix, "length: ", length(idx))
    show(io, database(idx); prefix, indent)
end

include("opt.jl")
include("searchgraph/SearchGraph.jl")
include("invertedfiles/InvertedFiles.jl")
include("permindex.jl")
include("deprecated.jl")

include("allknn.jl")
include("neardup.jl")
include("bichromatic/Bichromatic.jl")
using .Bichromatic
include("hsp.jl")
include("rerank.jl")

"""
    searchbatch(index, ctx, Q, k::Integer) -> (ids::Matrix{UInt32}, dists::Matrix{Float32})
    searchbatch(index, Q, k::Integer) -> (ids::Matrix{UInt32}, dists::Matrix{Float32})

Searches a batch of queries in the given index (searches for k neighbors).
Returns a tuple `(ids, dists)` where both are `(k, length(Q))` matrices.

# Arguments
- `index`: The search structure
- `Q`: The set of queries
- `k`: The number of neighbors to retrieve
- `ctx`: caches, hyperparameters, and meta data
- `sorted=true`: ensures that the results are sorted by distance.

Note: The i-th column in `ids`/`dists` corresponds to the i-th query in `Q`.
Note: Unused slots (fewer than `k` neighbors found) are filled with `0`/`Inf32`.
"""
function searchbatch(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, k::Integer; sorted::Bool=true)
    ids   = zeros(UInt32,  k, length(Q))
    dists = fill(typemax(Float32), k, length(Q))
    searchbatch!(index, ctx, Q, ids, dists; sorted)
end

"""
    searchbatch!(index, ctx, Q, ids, dists; sorted) -> (ids, dists)

In-place batch search. Fills `ids::AbstractMatrix{UInt32}` and `dists::AbstractMatrix{Float32}`
(each of size `(k, length(Q))`) with the `k` nearest neighbors of each query in `Q`.

# Arguments
- `index`: The search structure
- `ctx`: Context of the search algorithm
- `Q`: The set of queries
- `ids`: Output matrix of `UInt32` identifiers, size `(k, length(Q))`
- `dists`: Output matrix of `Float32` distances, size `(k, length(Q))`

# Keyword arguments
- `sorted`: whether each column should be sorted by distance (default `false`).
"""
function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase,
                      ids::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}; sorted::Bool=false)
    m = length(Q)
    m > 0 || throw(ArgumentError("empty set of queries"))
    m == size(ids, 2) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, m)
    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
    @LOOP for j in 1:m
        res = knnqueue(bctx, view(ids, :, j), view(dists, :, j))
        search(index, bctx, Q[j], res)
        sorted && sortitems!(res)
    end
    end

    ids, dists
end

"""
    searchbatch!(index, ctx, Q, knns::AbstractVector{<:AbstractMetricQueue}) -> knns

In-place batch search using caller-provided, per-query result containers instead of
pre-sized `(k, length(Q))` matrices. Unlike the matrix-based `searchbatch!` above, `knns`
need not hold a uniform, fixed-size container per query: each `knns[i]` can be any
[`AbstractMetricQueue`](@ref) -- a fixed-`k` [`KnnSorted`](@ref)/[`KnnHeap`](@ref), or a
growable, radius-thresholded [`RadiusSorted`](@ref)/[`RadiusHeap`](@ref) -- and they need not
even share the same concrete type. This is the only entry point radius-bounded containers are
meant to be driven through; they are not wired into the `(ids, dists)` matrix form (which
requires a uniform `k`) or into `GenericContext`/`SearchGraphContext`'s automatic
`knnqueue(ctx, k)` construction (which has no notion of a radius).

Does **not** call [`reuse!`](@ref) on the elements of `knns` -- pass already-fresh containers.

# Arguments
- `index`: The search structure
- `ctx`: Context of the search algorithm
- `Q`: The set of queries
- `knns`: One result container per query, `length(knns) == length(Q)`

# Examples

```julia
# radius-bounded search: every point within 0.3 of each query, however many that is
knns = [RadiusSorted(0.3f0) for _ in 1:length(Q)]
searchbatch!(index, ctx, Q, knns)
for (q, res) in zip(Q, knns)
    for p in IdDistView(res)
        println(p.id, " ", p.dist)
    end
end
```
"""
function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractVector{<:AbstractMetricQueue})
    m = length(Q)
    m > 0 || throw(ArgumentError("empty set of queries"))
    m == length(knns) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, m)
    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
    @LOOP for i in 1:m
        search(index, bctx, Q[i], knns[i])
    end
    end

    knns
end

end
