# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type AbstractSearchIndex end

using Polyester
using Accessors

import Base: push!, append!
export AbstractSearchIndex, AbstractContext, GenericContext,
    SemiMetric, evaluate, search, searchbatch, searchbatch!, database, distance,
    getcontext, getminbatch,
    SearchResult, push_item!, append_items!, IdWeight, StaticAdjacencyList, AdjacencyList

abstract type AbstractContext end
function searchbatch! end
function search end
function push_item! end
function append_items! end
function index! end

include("distances/Distances.jl")
include("db/db.jl")
include("distsample.jl")
include("adj.jl")

using .AdjacencyLists

include("log.jl")
include("pqueue/pqueue.jl")

@inline Base.length(searchctx::AbstractSearchIndex) = length(database(searchctx))
@inline Base.eachindex(searchctx::AbstractSearchIndex) = 1:length(searchctx)
@inline Base.eltype(searchctx::AbstractSearchIndex) = eltype(searchctx.db)

"""
    getminbatch(minbatch::Int, n::Int=0)
    getminbatch(ctx::GenericContext, n::Int=0)

Used by functions that use parallelism based on `Polyester.jl` minibatches specify how many queries (or something else) are solved per thread whenever
the thread is used (in minibatches). 

# Arguments
- `minbatch`
  - Integers ``1 ≤ minbatch ≤ n`` are valid values (where n is the number of objects to process, i.e., queries)
  - Defaults to 0 which computes a default number based on the number of available cores and `n`.
  - Set `minbatch=-1` to avoid parallelism.

"""
function getminbatch(minbatch::Int, n::Int=0)
    minbatch < 0 && return n
    nt = Threads.nthreads()
    if minbatch == 0
        # it seems to work for several workloads
        n <= 2nt && return 1
        n <= 4nt && return 2
        #n <= 8nt && return 4
        return 4
        # n <= 2nt ? 2 : min(4, ceil(Int, n / nt))
    else
        return ceil(Int, minbatch)
    end
end


"""
    database(index)

Gets the entire indexed database
"""
@inline database(searchctx::AbstractSearchIndex) = searchctx.db

"""
    database(index, i)

Gets the i-th object from the indexed database
"""
@inline database(searchctx::AbstractSearchIndex, i) = database(searchctx)[i]
@inline Base.getindex(searchctx::AbstractSearchIndex, i::Integer) = database(searchctx, i)


"""
    distance(index)

Gets the distance function used in the index
"""
@inline distance(searchctx::AbstractSearchIndex) = searchctx.dist

struct GenericContext{KnnType} <: AbstractContext
    minbatch::Int
    verbose::Bool
    logger
end

GenericContext(KnnType::Type{<:AbstractKnn}=KnnSorted; minbatch::Integer=0, verbose::Bool=true, logger=InformativeLog()) =
    GenericContext{KnnType}(minbatch, verbose, logger)

getcontext(s::AbstractSearchIndex) = error("Not implemented method for $s")
knnqueue(::GenericContext{KnnType}, arg) where {KnnType<:AbstractKnn} = knnqueue(KnnType, arg)
verbose(ctx::GenericContext) = ctx.verbose
getminbatch(ctx::AbstractContext, n::Int=0) = getminbatch(ctx.minbatch, n)

include("perf.jl")
include("sequential-exhaustive.jl")
include("parallel-exhaustive.jl")


function Base.show(io::IO, idx::AbstractSearchIndex; prefix="", indent="  ")
    println(io, prefix, typeof(idx),":")
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
include("fft.jl")
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
- `context`: caches, hyperparameters, and meta data (defaults to `getcontext(index)`)
- `sorted=true`: ensures that the results are sorted by distance.

Note: The i-th column in indices and distances correspond to the i-th query in `Q`
Note: The final indices at each column can be `0` if the search process was unable to retrieve `k` neighbors.
"""
function searchbatch(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, k::Integer; sorted::Bool=true)
    knns = zeros(IdWeight, k, length(Q))
    searchbatch!(index, ctx, Q, knns; sorted)
end

"""
    searchbatch!(index, ctx, Q, knns; sorted) -> knns

Searches a batch of queries in the given index and use `knns` as output (searches for `k=size(I, 1)`)

# Arguments
- `index`: The search structure
- `ctx`: Context of the search algorithm, environment for running searches (hyperparameters and caches)
- `Q`: The set of queries
- `knns`: Output, a matrix of IdWeight elements (initialized with `zeros`); an array of KnnAbstract elements, use this form to retrieve search costs.

# Keyword arguments
- `sorted`: indicates whether the output should be sorted or not.
"""
function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractMatrix{IdWeight}; sorted::Bool=false)
    length(Q) > 0 || throw(ArgumentError("empty set of queries"))
    length(Q) == size(knns, 2) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, length(Q))

    @batch minbatch = minbatch per = thread for i in eachindex(Q)
        #Threads.@threads :static for i in eachindex(Q)
        res = knnqueue(ctx, view(knns, :, i))
        search(index, ctx, Q[i], res)
        # @assert length(res) == size(knns, 1)
        sorted && sortitems!(res)
    end

    knns
end

function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractVector{<:AbstractKnn})
    length(Q) > 0 || throw(ArgumentError("empty set of queries"))
    length(Q) == length(knns) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, length(Q))

    @batch minbatch = minbatch per = thread for i in eachindex(Q)
        # Threads.@threads :static for i in eachindex(Q)
        search(index, ctx, Q[i], knns[i])
    end

    knns
end

end
