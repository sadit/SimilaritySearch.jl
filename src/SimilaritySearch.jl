# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type AbstractSearchIndex end

using Parameters
using Polyester
using JLD2

import Base: push!, append!
export AbstractSearchIndex, AbstractContext, GenericContext, 
       SemiMetric, evaluate, search, searchbatch, getknnresult, database, distance,
       getcontext, getminbatch, saveindex, loadindex,
       SearchResult, push_item!, append_items!, IdWeight

include("distances/Distances.jl")

include("db/db.jl")
include("adj.jl")
include("log.jl")

using .AdjacencyLists

include("knnresult.jl")
include("io.jl")

@inline Base.length(searchctx::AbstractSearchIndex) = length(database(searchctx))
@inline Base.eachindex(searchctx::AbstractSearchIndex) = 1:length(searchctx)
@inline Base.eltype(searchctx::AbstractSearchIndex) = eltype(searchctx.db)

abstract type AbstractContext end

struct GenericContext <: AbstractContext
    knn::Vector{KnnResult}
    minbatch::Int
    logger
end

GenericContext(; k::Integer=32, minbatch::Integer=0, logger=InformativeLog()) =
    GenericContext([KnnResult(k) for _ in 1:Threads.nthreads()], minbatch, logger)

GenericContext(ctx::AbstractContext; knn=ctx.knn, minbatch=ctx.minbatch, logger=ctx.logger) =
    GenericContext(knn, minbatch, logger)

function getcontext(s::AbstractSearchIndex)
    error("Not implemented method for $s")
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

"""
    struct SearchResult
        res::KnnResult  # result struct
        cost::Int  # number of distances (if algorithm allow it)
    end

Response of a typical search knn query
"""
struct SearchResult
    res::KnnResult
    cost::Int
end

include("perf.jl")
include("sequential-exhaustive.jl")
include("parallel-exhaustive.jl")
include("opt.jl")
include("searchgraph/SearchGraph.jl")
include("permindex.jl")
include("deprecated.jl")

include("allknn.jl")
include("neardup.jl")
include("closestpair.jl")


"""
    getknnresult(k::Integer, ctx::AbstractContext) -> KnnResult

Generic function to obtain a shared result set for the same thread and avoid memory allocations.
This function should be specialized for indexes and caches that use shared results or threads in some special way.
"""
@inline function getknnresult(k::Integer, ctx::AbstractContext)
    res = ctx.knn[Threads.threadid()]
    reuse!(res, k)
end

"""
    searchbatch(index, ctx, Q, k::Integer) -> indices, distances

Searches a batch of queries in the given index (searches for k neighbors).

# Arguments
- `index`: The search structure
- `Q`: The set of queries
- `k`: The number of neighbors to retrieve

# Keyword arguments
- `context`: caches, hyperparameters, and meta data

Note: The i-th column in indices and distances correspond to the i-th query in `Q`
Note: The final indices at each column can be `0` if the search process was unable to retrieve `k` neighbors.
"""
function searchbatch(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, k::Integer)
    m = length(Q)
    I = Matrix{Int32}(undef, k, m)
    D = Matrix{Float32}(undef, k, m)
    searchbatch(index, ctx, Q, I, D)
end

"""
    searchbatch(index, ctx, Q, I::AbstractMatrix{Int32}, D::AbstractMatrix{Float32}) -> indices, distances

Searches a batch of queries in the given index and `I` and `D` as output (searches for `k=size(I, 1)`)

# Arguments
- `index`: The search structure
- `Q`: The set of queries
- `k`: The number of neighbors to retrieve
- `ctx`: environment for running searches (hyperparameters and caches)
"""
function searchbatch(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, I::AbstractMatrix{Int32}, D::AbstractMatrix{Float32}) 
    @assert size(I) == size(D)
    minbatch = getminbatch(ctx.minbatch, length(Q))
    I_ = PtrArray(I)
    D_ = PtrArray(D)
    if minbatch < 0
        for i in eachindex(Q)
            solve_single_query(index, ctx, Q, i, I_, D_)
        end
    else
        @batch minbatch=minbatch per=thread for i in eachindex(Q)
            solve_single_query(index, ctx, Q, i, I_, D_)
        end
    end

    I, D
end

function solve_single_query(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, i, knns_, dists_)
    k = size(knns_, 1)
    q = @inbounds Q[i]
    res = getknnresult(k, ctx)
    search(index, ctx, q, res)
    _k = length(res)
    @inbounds for j in 1:_k
        u = res.items[j]
        knns_[j, i] = u.id
        dists_[j, i] = u.weight
    end

    for j in _k+1:k
        knns_[j, i] = zero(Int32)
    end
end


"""
    searchbatch(index, context, Q, KNN::AbstractVector{KnnResult}) -> indices, distances

Searches a batch of queries in the given index using an array of KnnResult's; each KnnResult object can specify different `k` values.

# Arguments
- `context`: contain caches to reduce memory allocations and hyperparameters for search customization

"""
function searchbatch(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, KNN::AbstractVector{KnnResult})
    minbatch = getminbatch(ctx.minbatch, length(Q))

    if minbatch < 0
        @inbounds for i in eachindex(Q)
            search(index, ctx, Q[i], KNN[i])
        end
    else
        @batch minbatch=minbatch per=thread for i in eachindex(Q)
            @inbounds search(index, ctx, Q[i], KNN[i])
        end
    end

    KNN
end

"""
    getminbatch(minbatch, n)

Used by functions that use parallelism based on `Polyester.jl` minibatches specify how many queries (or something else) are solved per thread whenever
the thread is used (in minibatches). 

# Arguments
- `minbatch`
  - Integers ``1 ≤ minbatch ≤ n`` are valid values (where n is the number of objects to process, i.e., queries)
  - Defaults to 0 which computes a default number based on the number of available cores and `n`.
  - Set `minbatch=-1` to avoid parallelism.

"""
function getminbatch(minbatch, n)
    minbatch < 0 && return n
    nt = Threads.nthreads()
    if minbatch == 0
        # it seems to work for several workloads
        n <= 2nt && return 1
        n <= 4nt && return 2
        n <= 8nt && return 4
        return 8
        # n <= 2nt ? 2 : min(4, ceil(Int, n / nt))
    else
        return ceil(Int, minbatch)
    end
end

DEFAULT_CONTEXT = Ref(GenericContext())
DEFAULT_SEARCH_GRAPH_CONTEXT = Ref(SearchGraphContext())
function __init__()
    DEFAULT_CONTEXT[] = GenericContext()
    DEFAULT_SEARCH_GRAPH_CONTEXT[] = SearchGraphContext()
end

end  # end SimilaritySearch module
