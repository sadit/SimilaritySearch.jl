export EssencialNeighborhood

struct EssencialNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function EssencialNeighborhood()
    return EssencialNeighborhood(32)
end

function optimize_neighborhood!{T}(algo::EssencialNeighborhood, index::LocalSearchIndex{T}, perf, recall)
end

function neighborhood{T}(algo::EssencialNeighborhood, index::LocalSearchIndex{T}, item::T)
    nbuffer::Vector{Int32} = Vector{Int}(0)
    knn = search(index, item, KnnResult(algo.k))
    visible = Set{Int32}()

    @inbounds for p in knn
        in(p.objID, visible) && continue
        for neighbor in index.links[p.objID]
            push!(visible, neighbor)
        end

        push!(nbuffer, p.objID)
    end

    return knn, nbuffer
end
