# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type Index end
abstract type AbstractSearchContext end

using Parameters
import Distances: evaluate, PreMetric
export AbstractSearchContext, PreMetric, evaluate, search, searchbatch, knnresults

include("db.jl")
include("utils/knnresult.jl")
include("utils/knnresultshifted.jl")

const GlobalKnnResult = [KnnResult(32)]   # see __init__ function at the end of this file

@inline getknnresult(res=nothing) = res !== nothing ? res : @inbounds GlobalKnnResult[Threads.threadid()]

"""
    search(searchctx::AbstractSearchContext, q, k::Integer=maxlength(searchctx.res))
    search(searchctx::AbstractSearchContext, q)

This is the most generic search function. It calls almost all implementations whenever an integer k is given.

"""
function search(searchctx::AbstractSearchContext, q, k::Integer=10)
    res = getknnresult()
    empty!(res, k)
    search(searchctx, q, res)
end

function searchbatch(searchctx, Q, k::Integer=10; parallel=false)
    m = length(Q)
    I = Matrix{Int32}(undef, k, m)
    D = Matrix{Float32}(undef, k, m)
    searchbatch(searchctx, Q, I, D; parallel)
end

function searchbatch(searchctx, Q, I::Matrix{Int32}, D::Matrix{Float32}; parallel=false)
    if parallel
        Threads.@threads for i in eachindex(Q)
            @inbounds search(searchctx, Q[i], KnnResult(I, D, i))
        end
    else
        @time @inbounds for i in eachindex(Q)
            search(searchctx, Q[i], KnnResult(I, D, i))
        end
    end

    I, D
end

@inline Base.length(searchctx::AbstractSearchContext) = length(searchctx.db)
@inline Base.getindex(searchctx::AbstractSearchContext, i) = searchctx.db[i]
@inline Base.eachindex(searchctx::AbstractSearchContext) = 1:length(searchctx)
@inline Base.eltype(searchctx::AbstractSearchContext) = eltype(searchctx.db)

include("distances/bits.jl")
include("distances/sets.jl")
include("distances/strings.jl")
include("distances/vectors.jl")
include("distances/cos.jl")
include("distances/cloud.jl")
include("utils/perf.jl")
include("indexes/seq.jl")
include("graph/graph.jl")

function __init__()
    __init__visitedvertices()
    __init__beamsearch()
    __init__neighborhood()
    for i in 2:Threads.nthreads()
        push!(GlobalKnnResult, KnnResult(32))
    end
end

end
