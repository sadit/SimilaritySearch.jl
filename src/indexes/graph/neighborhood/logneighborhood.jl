export LogNeighborhood

struct LogNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function LogNeighborhood()
    return LogNeighborhood(2)
end

function optimize_neighborhood!{T}(algo::LogNeighborhood, index::LocalSearchIndex{T}, perf, recall)
end

function neighborhood{T}(algo::LogNeighborhood, index::LocalSearchIndex{T}, item::T)
    n = length(index.db)
    k = max(1, log(algo.base, n) |> ceil |> Int)

    knn = search(index, item, KnnResult(k))
    nbuffer::Vector{Int32} = Vector{Int32}(length(knn))

    for (i, p) in enumerate(knn)
        nbuffer[i] = p.objID
    end

    return knn, nbuffer
end
