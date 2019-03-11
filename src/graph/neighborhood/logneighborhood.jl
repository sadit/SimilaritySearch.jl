export LogNeighborhood

struct LogNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function LogNeighborhood()
    return LogNeighborhood(2)
end

function optimize_neighborhood!(algo::LogNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where T
end

function neighborhood(algo::LogNeighborhood, index::SearchGraph{T}, dist::Function, item::T) where T
    n = length(index.db)
    k = max(1, log(algo.base, n) |> ceil |> Int)

    knn = search(index, dist, item, KnnResult(k))
    nbuffer::Vector{Int32} = Vector{Int32}(undef, length(knn))

    for (i, p) in enumerate(knn)
        nbuffer[i] = p.objID
    end

    return knn, nbuffer
end
