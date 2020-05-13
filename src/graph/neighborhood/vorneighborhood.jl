# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export VorNeighborhood

struct VorNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function VorNeighborhood()
    return VorNeighborhood(1.1)
end

function optimize_neighborhood!(algo::VorNeighborhood, index::SearchGraph{T}, dist, perf, recall) where T
end

function neighborhood(algo::VorNeighborhood, index::SearchGraph{T}, dist, item::T, knn, N, searchctx) where T
    k = max(1, ceil(Int, log(algo.base, length(index.db))))
    reset!(knn, k)
    empty!(N)
    n = length(index.db)
    search(index, dist, item, knn; searchctx=searchctx)

    @inbounds for p in knn
        pobj = index.db[p.id]
        covered = false
        for nearID in N
            d = convert(Float32, dist(index.db[nearID], pobj))
            if d <= p.dist
                covered = true
                break
            end
        end

        !covered && push!(N, p.id)
    end
end
