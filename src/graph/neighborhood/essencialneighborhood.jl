# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export EssencialNeighborhood

struct EssencialNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function EssencialNeighborhood()
    return EssencialNeighborhood(32)
end

function optimize_neighborhood!(algo::EssencialNeighborhood, index::SearchGraph{T}, dist, perf, recall) where {T}
end

function neighborhood(algo::EssencialNeighborhood, index::SearchGraph{T}, dist, item::T) where {T}
    nbuffer::Vector{Int32} = Vector{Int}(undef, 0)
    knn = search(index, dist, item, KnnResult(algo.k))
    visible = Set{Int32}()

    @inbounds for p in knn
        in(p.id, visible) && continue
        for neighbor in index.links[p.id]
            push!(visible, neighbor)
        end

        push!(nbuffer, p.id)
    end

    return knn, nbuffer
end
