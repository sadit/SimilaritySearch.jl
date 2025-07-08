# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type AbstractSearchIndex end

using Parameters
using Polyester
using JLD2

import Base: push!, append!
export AbstractSearchIndex, AbstractContext, GenericContext, 
       SemiMetric, evaluate, search, searchbatch, database, distance,
       getcontext, getminbatch, saveindex, loadindex,
       SearchResult, push_item!, append_items!, IdWeight


include("distances/Distances.jl")

include("db/db.jl")
include("distsample.jl")
include("adj.jl")
include("log.jl")

using .AdjacencyLists

include("knnresult/KnnResult.jl")
include("io.jl")

@inline Base.length(searchctx::AbstractSearchIndex) = length(database(searchctx))
@inline Base.eachindex(searchctx::AbstractSearchIndex) = 1:length(searchctx)
@inline Base.eltype(searchctx::AbstractSearchIndex) = eltype(searchctx.db)

abstract type AbstractContext end

struct GenericContext <: AbstractContext
    minbatch::Int
    logger
end

GenericContext(; minbatch::Integer=0, logger=InformativeLog()) =
    GenericContext(minbatch, logger)

GenericContext(ctx::AbstractContext; minbatch=ctx.minbatch, logger=ctx.logger) =
    GenericContext(minbatch, logger)

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
    m = length(Q)
    knns = Matrix{IdWeight}(undef, k, m)
    searchbatch!(index, ctx, Q, knns)
    knns
end

"""
    searchbatch!(index, ctx, Q, knns; costs, eblocks, check_args) -> knns, costs, eblocks

Searches a batch of queries in the given index and `I` and `D` as output (searches for `k=size(I, 1)`)

# Arguments
- `index`: The search structure
- `ctx`: Context of the search algorithm
- `Q`: The set of queries
- `knns`: Matrix of outputs
- `k`: The number of neighbors to retrieve
- `ctx`: environment for running searches (hyperparameters and caches)

# Keyword arguments
- `cost`: nothing or a vector like collection to store the number of distance evaluations to solve each query
- `eblocks`: nothing or a vector like collection to store the the number of evaluated blocks for each query (the precise definition depends on the index)
- `check_args`: activate or deactivate checking arguments, this is useful for reusing structures without using views
- `sorted`: indicates whether the output should be sorted or not.
"""
function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractMatrix{IdWeight}; cost=nothing, eblocks=nothing, check_args::Bool=true, sorted=true)
    if check_args
        length(Q) > 0 || throw(ArgumentError("empty set of queries"))
        (length(Q) == size(knns, 2)) || throw(ArgumentError("the number of queries is different from the given output containers"))
        (cost === nothing || length(Q) == length(cost)) || throw(ArgumentError("the number of queries is different from the costs output vector"))
    end

    minbatch = getminbatch(ctx.minbatch, length(Q))

    if cost === nothing && eblocks === nothing
        @batch minbatch=minbatch per=thread for i in eachindex(Q)
            res = knndefault(@view knns[:, i])
            search(index, ctx, Q[i], res)
            sorted && sortitems!(res)
        end
    else
        @batch minbatch=minbatch per=thread for i in eachindex(Q)
            res = knndefault(@view knns[:, i])
            search(index, ctx, Q[i], res)
            sorted && sortitems!(res)
            cost !== nothing && (cost[i] = res.cost)
            eblocks !== nothing && (eblocks[i] = res.eblocks)
        end
    end

    knns, cost, eblocks
end


"""
    searchbatch!(index, ctx, Q, knns) -> knns

Searches a batch of queries in the given index using an array of KnnResult's; each KnnResult object can specify different `k` values.

# Arguments
- `ctx`: contain caches to reduce memory allocations and hyperparameters for search customization

"""
function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractVector{<:AbstractKnn})
    length(Q) > 0 || throw(ArgumentError("empty set of queries"))
    length(Q) == length(knns) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx.minbatch, length(Q))

    @batch minbatch=minbatch per=thread for i in eachindex(Q)
        search(index, ctx, Q[i], knns[i])
    end
    
    knns
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
        #n <= 8nt && return 4
        return 4
        # n <= 2nt ? 2 : min(4, ceil(Int, n / nt))
    else
        return ceil(Int, minbatch)
    end
end

#=
using PrecompileTools
@setup_workload begin
    X = rand(Float32, 2, 64)
    Q = rand(Float32, 2, 8)
    k = 8
    for c in eachcol(X) normalize!(c) end
    for c in eachcol(Q) normalize!(c) end
    
    @compile_workload begin
        for (db, queries) in [(MatrixDatabase(X), MatrixDatabase(Q)),
                              (StrideMatrixDatabase(X), StrideMatrixDatabase(Q))
                             ]
            for dist in [L1Distance(), L2Distance(), SqL2Distance(), CosineDistance(), NormalizedCosineDistance(), TurboL2Distance(), TurboSqL2Distance(), TurboCosineDistance(), TurboNormalizedCosineDistance()]
                G = SearchGraph(; dist, db)  
                E = ExhaustiveSearch(; dist, db)  
                for idx in [G, E]
                    ctx = getcontext(idx)
                    index!(idx, ctx)
                    knns, dists = searchbatch(idx, ctx, queries, k)

                    #knns, dists = allknn(idx, ctx, k)
                    #closestpair(idx, ctx)
                    hsp_queries(idx, queries, k)
                end 
            end
        end
    end
end
=#
end  # end SimilaritySearch module
