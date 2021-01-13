# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LogNeighborhood

struct LogNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function LogNeighborhood()
    return LogNeighborhood(2)
end

function optimize_neighborhood!(algo::LogNeighborhood, index::SearchGraph{T}, dist, perf, recall) where T
end

function neighborhood(algo::LogNeighborhood, index::SearchGraph{T}, dist, item::T, knn, N, searchctx) where T
    n = length(index.db)
    k = max(1, log(algo.base, n) |> ceil |> Int)
    empty!(knn, k)
    empty!(N)
    knn = search(index, dist, item, KnnResult(k), searchctx=searchctx)

    for p in knn
        push!(N, p.id)
    end
end
