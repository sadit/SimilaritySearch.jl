# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export FixedNeighborhood

struct FixedNeighborhood <: NeighborhoodAlgorithm
    k::Int
end

function FixedNeighborhood()
    return FixedNeighborhood(8)
end

function optimize_neighborhood!(algo::FixedNeighborhood, index::SearchGraph{T}, dist::Function, perf, recall) where T
end

function neighborhood(algo::FixedNeighborhood, index::SearchGraph{T}, dist::Function, item::T, knn::KnnResult, N::Vector) where T
    reset!(knn, algo.k)
    empty!(N)
    search(index, dist, item, knn)
    
    for p in knn
        push!(N, p.objID)
    end

end
