export GallopingSatNeighborhood

mutable struct GallopingSatNeighborhood <: NeighborhoodAlgorithm
    g::GallopingNeighborhood
end

# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

function GallopingSatNeighborhood()
    return GallopingSatNeighborhood(GallopingNeighborhood())
end

function optimize_neighborhood!(algo::GallopingSatNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where {T}
    optimize_neighborhood!(algo.g, index, dist, perf, recall)
end

function neighborhood(algo::GallopingSatNeighborhood, index::SearchGraph{T}, dist::Function, item::T, knn, N) where {T}
    reset!(knn, algo.g.neighborhood)
    empty!(N)
    knn = search(index, dist, item, knn)

    @inbounds for p in knn
        dqp = p.dist
        pobj = index.db[p.objID]
        near = KnnResult(1)
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
