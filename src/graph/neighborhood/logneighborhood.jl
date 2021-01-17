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

function find_neighborhood(algo::LogNeighborhood, index::SearchGraph, item)
    n = length(index.db)
    k = max(1, log(algo.base, n) |> ceil |> Int)
    knn = search(index, item, k)

    [p.id for p in search(index, item, k)]
end
