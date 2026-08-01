# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type AbstractSearchIndex end
using Accessors

include("parallel.jl")

import Base: push!, append!
export AbstractSearchIndex, AbstractContext, GenericContext, ExhaustiveSearch,
    search, searchbatch, searchbatch!, database, distance,
    SearchResult, push_item!, append_items!, getminbatch,
    IdDist, Dist, Exact, Special, ScalarQuant

abstract type AbstractContext end
function searchbatch! end
function search end
function push_item! end
function append_items! end
function index! end

"""
    getminbatch(n::Int, nt::Int=Threads.nthreads();
                blocks_per_thread::Int=8, maxbatches::Int=n)

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

- `blocks_per_thread` (default `8`): the natural batch-count target is
  `blocks_per_thread * nt` -- always tied to the thread count, never an
  independent/arbitrary number.
- `maxbatches` (default `n`, i.e. no effective restriction): a **hard ceiling** on the
  batch count, overriding the natural target above whenever it would be larger. Use this
  to directly bound the memory of per-batch scratch allocations (e.g. [`@BATCHES`](@ref)'s
  `@BEGIN`-declared, [`@nbatches`](@ref)-sized buffers) for very large `n`. When a context
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
  regardless of `nt` (defaults to `n`, a genuine no-op unless set to something smaller)
"""
function getminbatch(n::Int, nt::Int=Threads.nthreads();
                      blocks_per_thread::Int=8, maxbatches::Int=n)
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
include("special/special.jl")
include("proj/Projections.jl")

#using .Adj

include("log.jl")
include("pqueue/pqueue.jl")

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
    GenericContext(KnnType::Type{<:AbstractKnn}=KnnSorted;
        verbose::Bool=true, logger=InformativeLog(),
        maxbatches::Integer=8Threads.nthreads(), batchid::Integer=1) -> GenericContext

Lightweight [`AbstractContext`](@ref) implementation used by exact indexes
([`ExhaustiveSearch`](@ref), [`ParallelExhaustiveSearch`](@ref)) that need no per-thread
scratch caches.

# Keyword Arguments
- `verbose`: controls the number of output messages.
- `logger`: how to handle and log events.
- `maxbatches`: hard cap on the batch count used by [`getminbatch`](@ref) for operations
  driven by this context (e.g. [`searchbatch!`](@ref), [`allknn`](@ref),
  [`closestpair`](@ref), [`search`](@ref search(::ParallelExhaustiveSearch, ::GenericContext, ::Any, ::AbstractKnn))).
  Defaults to `8 * Threads.nthreads()`, matching [`getminbatch`](@ref)'s own default
  `blocks_per_thread`.
- `batchid`: the batch slot this context is tagged with; not meaningful on the root
  context returned here (always `1`) -- per-batch copies tagging the running `@batchid`
  are minted internally via `Accessors.@set`, one per batch, not per call.
"""
struct GenericContext{KnnType} <: AbstractContext
    verbose::Bool
    logger
    maxbatches::Int32
    batchid::Int32
end

GenericContext(KnnType::Type{<:AbstractKnn}=KnnSorted; verbose::Bool=true, logger=InformativeLog(),
    maxbatches::Integer=8Threads.nthreads(), batchid::Integer=1) =
    GenericContext{KnnType}(verbose, logger, convert(Int32, maxbatches), convert(Int32, batchid))

# GenericContext has a phantom type parameter (KnnType, not derivable from any field), so
# ConstructionBase's default reconstruction (used by Accessors.@set) can't infer it -- this
# override makes `@set ctx.batchid = ...`/`@set ctx.maxbatches = ...` work.
Accessors.ConstructionBase.constructorof(::Type{<:GenericContext{K}}) where {K} = (args...) -> GenericContext{K}(args...)

knnqueue(::GenericContext{KnnType}, arg) where {KnnType<:AbstractKnn} = knnqueue(KnnType, arg)
verbose(ctx::GenericContext) = ctx.verbose

include("perf.jl")
include("fft.jl")
include("exact/Exact.jl")

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
include("permindex.jl")
include("deprecated.jl")

include("allknn.jl")
include("neardup.jl")
include("closestpair.jl")
include("hsp.jl")
include("rerank.jl")

"""
    searchbatch(index, ctx, Q, k::Integer) -> indices, distances
    searchbatch(index, Q, k::Integer) -> indices, distances

Searches a batch of queries in the given index (searches for k neighbors).

# Arguments
- `index`: The search structure
- `Q`: The set of queries
- `k`: The number of neighbors to retrieve
- `context`: caches, hyperparameters, and meta data
- `sorted=true`: ensures that the results are sorted by distance.

Note: The i-th column in indices and distances correspond to the i-th query in `Q`
Note: The final indices at each column can be `0` if the search process was unable to retrieve `k` neighbors.
"""
function searchbatch(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, k::Integer; sorted::Bool=true)
    knns = zeros(IdDist, k, length(Q))
    searchbatch!(index, ctx, Q, knns; sorted)
end

"""
    searchbatch!(index, ctx, Q, knns; sorted) -> knns

Searches a batch of queries in the given index and use `knns` as output (searches for `k=size(I, 1)`)

# Arguments
- `index`: The search structure
- `ctx`: Context of the search algorithm, environment for running searches (hyperparameters and caches)
- `Q`: The set of queries
- `knns`: Output, a matrix of IdDist elements (initialized with `zeros`); an array of KnnAbstract elements, use this form to retrieve search costs.

# Keyword arguments
- `sorted`: indicates whether the output should be sorted or not.
"""
function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractMatrix{IdDist}; sorted::Bool=false)
    m = length(Q)
    m > 0 || throw(ArgumentError("empty set of queries"))
    m == size(knns, 2) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, m)
    # @info m => Threads.nthreads() => minbatch
    @BATCHES minbatch begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid
    @LOOP for j in 1:m
        res = knnqueue(bctx, view(knns, :, j))
        search(index, bctx, Q[j], res)
        sorted && sortitems!(res)
    end
    end

    knns
end

function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractVector{<:AbstractKnn})
    m = length(Q)
    m > 0 || throw(ArgumentError("empty set of queries"))
    m == length(knns) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, m)
    @BATCHES minbatch begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid
    @LOOP for i in 1:m
        search(index, bctx, Q[i], knns[i])
    end
    end

    knns
end

end
