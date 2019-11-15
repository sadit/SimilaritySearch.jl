# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export SatNeighborhood

struct SatNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function SatNeighborhood()
    return SatNeighborhood(32)
end

function optimize_neighborhood!(algo::SatNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where {T}
end

function neighborhood(algo::SatNeighborhood, index::SearchGraph{T}, dist::Function, item::T) where {T}
    N = Int32[]
    knn = search(index, dist, item, KnnResult(algo.k))
    @inbounds for p in knn
        pobj = index.db[p.objID]
        near = KnnResult(1)
        push!(near, zero(Int32), p.dist)
        for nearID in N
            d = dist(index.db[nearID], pobj)
            push!(near, nearID, d)
        end

        if first(near).objID == 0
            push!(N, p.objID)
        end
    end

    return knn, N
end
