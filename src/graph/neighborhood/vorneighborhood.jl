export VorNeighborhood

struct VorNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function VorNeighborhood()
    return VorNeighborhood(1.1)
end

function optimize_neighborhood!(algo::VorNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where T
end

function neighborhood(algo::VorNeighborhood, index::SearchGraph{T}, dist::Function, item::T) where T
    n = length(index.db)
    k = max(1, ceil(Int, log(algo.base, n)))
    knn = search(index, dist, item, KnnResult(k))
    N = Int32[]

    @inbounds for p in knn
        pobj = index.db[p.objID]
        covered = false
        for nearID in N
            d = convert(Float32, dist(index.db[nearID], pobj))
            if d <= p.dist
                covered = true
                break
            end
        end
        if !covered
            push!(N, p.objID)
        end

    end

    return knn, N
end
