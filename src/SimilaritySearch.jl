# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type AbstractSearchIndex end

using Parameters
using Polyester
using JLD2

import Base: push!, append!
export AbstractSearchIndex, AbstractContext, GenericContext, 
       SemiMetric, evaluate, search, searchbatch, searchbatch!, database, distance,
       getcontext, getminbatch, saveindex, loadindex,
       SearchResult, push_item!, append_items!, IdWeight

include("distances/Distances.jl")

include("db/db.jl")
include("distsample.jl")
include("adj.jl")
include("log.jl")

using .AdjacencyLists

include("pqueue/pqueue.jl")
include("io.jl")

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

abstract type AbstractContext end

struct GenericContext{KnnType} <: AbstractContext
    minbatch::Int
    verbose::Bool
    logger
end

GenericContext(KnnType::Type{<:AbstractKnn}=KnnSorted; minbatch::Integer=0, verbose::Bool=false, logger=InformativeLog()) =
    GenericContext{KnnType}(minbatch, verbose, logger)

getcontext(s::AbstractSearchIndex) = error("Not implemented method for $s")
knnqueue(::GenericContext{KnnType}, arg) where {KnnType<:AbstractKnn} = knnqueue(KnnType, arg)
getminbatch(ctx::GenericContext, n::Int=0) = getminbatch(ctx.minbatch, n)
verbose(ctx::GenericContext) = ctx.verbose


include("perf.jl")
include("sequential-exhaustive.jl")
include("parallel-exhaustive.jl")
include("opt.jl")
include("searchgraph/SearchGraph.jl")
include("permindex.jl")
include("deprecated.jl")

include("allknn.jl")
include("neardup.jl")
include("fft.jl")
include("closestpair.jl")
include("hsp.jl")


#=
@inline function getknnresult(k::Integer)
    res = DEFAULT_CONTEXT[].knn[Threads.threadid()]
    reuse!(res, k)
end
=#

"""
    searchbatch(index, ctx, Q, k::Integer) -> indices, distances
    searchbatch(index, Q, k::Integer) -> indices, distances

Searches a batch of queries in the given index (searches for k neighbors).

# Arguments
- `index`: The search structure
- `Q`: The set of queries
- `k`: The number of neighbors to retrieve
- `context`: caches, hyperparameters, and meta data (defaults to `getcontext(index)`)


Note: The i-th column in indices and distances correspond to the i-th query in `Q`
Note: The final indices at each column can be `0` if the search process was unable to retrieve `k` neighbors.
"""
function searchbatch(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, k::Integer)
    knns = zeros(IdWeight, k, length(Q))
    searchbatch!(index, ctx, Q, knns; sorted=true)
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

    @batch minbatch=minbatch per=thread for i in eachindex(Q)
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

    @batch minbatch=minbatch per=thread for i in eachindex(Q)
    # Threads.@threads :static for i in eachindex(Q)
        search(index, ctx, Q[i], knns[i])
    end
    
    knns
end

end
