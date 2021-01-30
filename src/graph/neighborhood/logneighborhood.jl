# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export LogNeighborhood

struct LogNeighborhood <: NeighborhoodAlgorithm
    base::Float64
    LogNeighborhood(b=2) = new(b)
end

StructTypes.StructType(::Type{LogNeighborhood}) = StructTypes.Struct()
Base.copy(algo::LogNeighborhood) = LogNeighborhood(algo.base)

function find_neighborhood(algo::LogNeighborhood, index::SearchGraph, item)
    n = length(index.db)
    k = max(1, log(algo.base, n) |> ceil |> Int)
    knn = search(index, item, k)

    [p.id for p in search(index, item, k)]
end
