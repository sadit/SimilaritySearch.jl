# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LogSatNeighborhood

struct LogSatNeighborhood <: NeighborhoodAlgorithm
    base::Float64

    LogSatNeighborhood(base=1.1) = new(base)
end

function optimize_neighborhood!(algo::LogSatNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where {T}
    # optimize_neighborhood!(algo.g, index, perf, recall)
end

function neighborhood(algo::LogSatNeighborhood, index::SearchGraph{T}, dist::Function, item::T, knn::KnnResult, N::Vector) where {T}
    n = length(index.db)
    k = max(1, ceil(Int, log(algo.base, n)))
    reset!(knn, k)
    empty!(N)
    search(index, dist, item, knn)

    near = KnnResult(1)
    # reverse!(knn.pool)
    @inbounds for p in knn.pool
        dqp = p.dist
        pobj = index.db[p.objID]
        empty!(near)
        push!(near, p.objID, p.dist)
        for nearID in N
            d = convert(Float32, dist(index.db[nearID], pobj))
            push!(near, nearID, d)
        end

        f = first(near)
        if f.objID == p.objID
            push!(N, p.objID)
        end
    end
end
