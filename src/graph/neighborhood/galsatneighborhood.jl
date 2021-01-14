export GallopingSatNeighborhood

mutable struct GallopingSatNeighborhood <: NeighborhoodAlgorithm
    g::GallopingNeighborhood
end

# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

function GallopingSatNeighborhood()
    return GallopingSatNeighborhood(GallopingNeighborhood())
end

function optimize_neighborhood!(algo::GallopingSatNeighborhood, index::SearchGraph{T}, dist::PreMetric, perf, recall) where {T}
    optimize_neighborhood!(algo.g, index, dist, perf, recall)
end

function neighborhood(algo::GallopingSatNeighborhood, index::SearchGraph{T}, dist::PreMetric, item::T, knn::KnnResult, N::Vector, searchctx) where {T}
    empty!(knn, algo.g.neighborhood)
    empty!(N)
    knn = search(index, dist, item, knn; searchctx=searchctx)

    near = KnnResult(1)
    @inbounds for p in knn
        empty!(near)
        dqp = p.dist
        pobj = index.db[p.id]
        push!(near, p.id, p.dist)
        for nearID in N
            push!(near, nearID, evaluate(dist, index.db[nearID], pobj))
        end

        f = first(near)
        if f.id == p.id
            push!(N, p.id)
        end
    end

end
