export EssencialNeighborhood

struct EssencialNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function EssencialNeighborhood()
    return EssencialNeighborhood(32)
end

function optimize_neighborhood!(algo::EssencialNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where {T}
end

function neighborhood(algo::EssencialNeighborhood, index::SearchGraph{T}, dist::Function, item::T) where {T}
    nbuffer::Vector{Int32} = Vector{Int}(undef, 0)
    knn = search(index, dist::Function, item, KnnResult(algo.k))
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
