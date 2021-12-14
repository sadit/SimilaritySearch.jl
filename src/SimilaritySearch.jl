# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type Index end
abstract type AbstractSearchContext end

using Parameters
import Distances: evaluate, PreMetric
export AbstractSearchContext, PreMetric, evaluate, search, searchbatch, knnresults

include("db.jl")
include("utils/knnresultmatrix.jl")
include("utils/knnresultvector.jl")

const GlobalKnnResult = [KnnResult(32)]   # see __init__ function at the end of this file

@inline function getknnresult(k::Integer)
    res = @inbounds GlobalKnnResult[Threads.threadid()]
    reuse!(res, k)
end

"""
    search(searchctx::AbstractSearchContext, q, k::Integer=maxlength(searchctx.res))
    search(searchctx::AbstractSearchContext, q)

This is the most generic search function. It calls almost all implementations whenever an integer k is given.

"""
function search(searchctx::AbstractSearchContext, q, k::Integer=10)
    res = getknnresult(k)
    search(searchctx, q, res)
end

function searchbatch(index, Q, k::Integer=10; parallel=false)
    m = length(Q)
    I = zeros(Int32, k, m)
    D = Matrix{Float32}(undef, k, m)
    searchbatch(index, Q, I, D; parallel)
end

function searchbatch(index, Q, I::Matrix{Int32}, D::Matrix{Float32}; parallel=false)
    if parallel
        Threads.@threads for i in eachindex(Q)
            @inbounds search(index, Q[i], KnnResultMatrix(I, D, i))
        end
    else
        @time @inbounds for i in eachindex(Q)
            search(index, Q[i], KnnResultMatrix(I, D, i))
        end
    end

    I, D
end

function searchbatch(index, Q, KNN::AbstractVector{KnnResult}; parallel=false)
    if parallel
        Threads.@threads for i in eachindex(Q)
            @inbounds search(index, Q[i], KNN[i])
        end
    else
        @time @inbounds for i in eachindex(Q)
            search(index, Q[i], KNN[i])
        end
    end

    KNN
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
include("graph/rebuild.jl")

function __init__()
    __init__visitedvertices()
    __init__beamsearch()
    __init__neighborhood()
    for i in 2:Threads.nthreads()
        push!(GlobalKnnResult, KnnResult(32))
    end
end

end
