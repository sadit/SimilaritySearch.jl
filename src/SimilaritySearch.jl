# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type AbstractSearchIndex end
using Polyester
using Accessors

import Base: push!, append!
export AbstractSearchIndex, AbstractContext, GenericContext,
    search, searchbatch, searchbatch!, database, distance,
    getcontext, getminbatch,
    SearchResult, push_item!, append_items!, IdWeight, StaticAdjacencyList, AdjacencyList

abstract type AbstractContext end
function searchbatch! end
function search end
function push_item! end
function append_items! end
function index! end

include("dist/Dist.jl")

using .Dist

include("db/db.jl")
include("sq/sq.jl")
include("distsample.jl")
include("adj.jl")

using .AdjacencyLists

include("log.jl")
include("pqueue/pqueue.jl")

@inline Base.length(searchctx::AbstractSearchIndex) = length(database(searchctx))
@inline Base.eachindex(searchctx::AbstractSearchIndex) = 1:length(searchctx)
@inline Base.eltype(searchctx::AbstractSearchIndex) = eltype(database(searchctx))

"""
    getminbatch(ctx::GenericContext, n::Int)
    getminbatch(n::Int, nt::Int, expnt::Int)

Used by functions that use parallelism on ``n / (nt * expnt)`` blocks; each block is processed by a single thread

# Arguments
- `ctx`: The search context
- `n`: the number of elements to process
- `nt`: number of threads to use
- `expnt`: expansion factor for nt (larger numbers decrease the batch size thus increasing the number of batches/blocks to process)

Notes on `expnt`
  - Integers ``1 ≤ expnt ≤ n`` are valid values (where n is the number of objects to process, i.e., queries)
  - `expnt == 0` uses a small default value of `8`
  - `expnt < 0` makes `getminbatch` returns `n` (defines a large block, i.e., single thread)

"""
function getminbatch(n::Int, nt::Int, expnt::Int)
    expnt < 0 && return n # defines a single thread for the loop
    n == 0 && return 4 # n==0 is a non sense, but return some valid value to avoid errors in range specifications
    expnt = expnt == 0 ? 4 : expnt # just a nice good value
    return ceil(Int, n / (expnt * nt))
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
@inline database(searchctx::AbstractSearchIndex, i) = getindex(database(searchctx), i)
@inline Base.getindex(searchctx::AbstractSearchIndex, i::Integer) = database(searchctx, i)


"""
    distance(index)

Gets the distance function used in the index
"""
@inline distance(searchctx::AbstractSearchIndex) = searchctx.dist

struct GenericContext{KnnType} <: AbstractContext
    expnt::Int
    verbose::Bool
    logger
end

GenericContext(KnnType::Type{<:AbstractKnn}=KnnSorted; expnt::Integer=0, verbose::Bool=true, logger=InformativeLog()) =
    GenericContext{KnnType}(expnt, verbose, logger)

function getminbatch(ctx::GenericContext, n::Int)
    getminbatch(n, Threads.nthreads(), ctx.expnt)
end

getcontext(s::AbstractSearchIndex) = error("Not implemented method for $s")
knnqueue(::GenericContext{KnnType}, arg) where {KnnType<:AbstractKnn} = knnqueue(KnnType, arg)
verbose(ctx::GenericContext) = ctx.verbose

include("perf.jl")
include("sequential-exhaustive.jl")
include("parallel-exhaustive.jl")


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
    m = length(Q)
    m > 0 || throw(ArgumentError("empty set of queries"))
    m == size(knns, 2) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, m)

    # @show m, minbatch
    #Threads.@threads :static for j in 1:minbatch:m
    @batch per=thread minbatch=4 for j in 1:minbatch:m
        m_ = min(m, j + minbatch - 1)
        res = knnqueue(ctx, view(knns, :, j))
        search(index, ctx, Q[j], res)
        sorted && sortitems!(res)
        i = j + 1
        @inbounds while i <= m_
            reuse!(res, view(knns, :, i))
            search(index, ctx, Q[i], res)
            sorted && sortitems!(res)
            i += 1
        end
    end

    knns
end

function searchbatch!(index::AbstractSearchIndex, ctx::AbstractContext, Q::AbstractDatabase, knns::AbstractVector{<:AbstractKnn})
    m = length(Q)
    m > 0 || throw(ArgumentError("empty set of queries"))
    m == length(knns) || throw(ArgumentError("the number of queries is different from the given output containers"))
    minbatch = getminbatch(ctx, m)

    # @batch minbatch = minbatch per = thread for i in eachindex(Q)
    @batch per=thread minbatch=4 for j in 1:minbatch:m
    #Threads.@threads :static for j in 1:minbatch:m
        @inbounds for i in j:min(m, j + minbatch - 1)
            search(index, ctx, Q[i], knns[i])
        end
    end

    knns
end

end
