export FixedNeighborhood

struct FixedNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function FixedNeighborhood()
    return FixedNeighborhood(8)
end

function optimize_neighborhood!(algo::FixedNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where T
end

function neighborhood(algo::FixedNeighborhood, index::SearchGraph{T}, dist::Function, item::T) where T
    nbuffer::Vector{Int32} = Vector{Int}(undef, 0)
    knn = search(index, dist, item, KnnResult(algo.k))
    visible = Set{Int32}()
    
    for p in knn
        push!(nbuffer, p.objID)
    end

    return knn, nbuffer
end
