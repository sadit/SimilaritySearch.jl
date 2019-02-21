export GallopingSatNeighborhood

mutable struct GallopingSatNeighborhood <: NeighborhoodAlgorithm
    g::GallopingNeighborhood
end

function GallopingSatNeighborhood()
    return GallopingSatNeighborhood(GallopingNeighborhood())
end

function optimize_neighborhood!(algo::GallopingSatNeighborhood, index::LocalSearchIndex{T}, perf, recall) where {T}
    optimize_neighborhood!(algo.g, index, perf, recall)
end

function neighborhood(algo::GallopingSatNeighborhood, index::LocalSearchIndex{T}, item::T) where {T}
    knn = search(index, item, KnnResult(algo.g.neighborhood))
    N = Vector{Int32}(undef, 0)

    @inbounds for p in knn
        dqp = p.dist
        pobj = index.db[p.objID]
        near = NnResult()
        push!(near, p.objID, p.dist)
        for nearID in N
            d = convert(Float32, index.dist(index.db[nearID], pobj)) 
            push!(near, nearID, d)
        end

        f = first(near)
        if f.objID == p.objID
            push!(N, p.objID)
        end
    end

    return knn, N
end
