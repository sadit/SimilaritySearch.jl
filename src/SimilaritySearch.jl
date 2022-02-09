# This file is a part of SimilaritySearch.jl

module SimilaritySearch
abstract type Index end
abstract type AbstractSearchContext end

using Parameters
import Distances: evaluate, SemiMetric
export AbstractSearchContext, SemiMetric, evaluate, search, searchbatch

include("db.jl")
include("knnresult.jl")
include("knnresultshift.jl")

const GlobalKnnResult = [KnnResult(32)]   # see __init__ function at the end of this file

"""
    getknnresult(k::Integer) -> KnnResult

Internal function to share result sets for the same thread and avoid memory allocations.
"""
@inline function getknnresult(k::Integer)
    res = @inbounds GlobalKnnResult[Threads.threadid()]
    reuse!(res, k)
end

"""
    searchbatch(index, Q, k::Integer=10; parallel=false) -> indices, distances

Searches a batch of queries in the given index (searches for k neighbors).

Note: The i-th column in indices and distances correspond to the i-th query in `Q`
Note: The final indices at each column can be `0` if the search process was unable to retrieve `k` neighbors.
"""
function searchbatch(index, Q, k::Integer=10; parallel=false)
    m = length(Q)
    I = zeros(Int32, k, m)
    D = Matrix{Float32}(undef, k, m)
    searchbatch(index, Q, I, D; parallel)
end

"""
    searchbatch(index, Q, I::AbstractMatrix{Int32}, D::AbstractMatrix{Float32}; parallel=false) -> indices, distances

Searches a batch of queries in the given index and `I` and `D` as output (searches for `k=size(I, 1)`)
"""
function searchbatch(index, Q, I::AbstractMatrix{Int32}, D::AbstractMatrix{Float32}; parallel=false)
    k = size(I, 1)
    if parallel
        Threads.@threads for i in eachindex(Q)
            res, cost = search(index, Q[i], getknnresult(k))
            I[:, i] .= res.id
            D[:, i] .= res.dist
        end
    else
        @inbounds for i in eachindex(Q)
            res, cost = search(index, Q[i], getknnresult(k))
            I[:, i] .= res.id
            D[:, i] .= res.dist
        end
    end

    I, D
end

"""
    searchbatch(index, Q, KNN::AbstractVector{KnnResult}; parallel=false) -> indices, distances

Searches a batch of queries in the given index using an array of KnnResult's; each KnnResult object can specify different `k` values.

"""
function searchbatch(index, Q, KNN::AbstractVector{KnnResult}; parallel=false)
    if parallel
        Threads.@threads for i in eachindex(Q)
            @inbounds search(index, Q[i], KNN[i])
        end
    else
        @inbounds for i in eachindex(Q)
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
include("perf.jl")
include("seq.jl")
include("graph/graph.jl")
include("graph/rebuild.jl")
include("allknn.jl")

function __init__()
    __init__visitedvertices()
    __init__beamsearch()
    __init__neighborhood()
    for i in 2:Threads.nthreads()
        push!(GlobalKnnResult, KnnResult(32))
    end
end

end
