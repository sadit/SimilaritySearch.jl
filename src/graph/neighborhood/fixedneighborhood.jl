# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export FixedNeighborhood

"""
    FixedNeighborhood(k=8)

New items are connected with a fixed number of neighbors
"""
struct FixedNeighborhood <: NeighborhoodAlgorithm
    k::Int
    FixedNeighborhood(k=8) = new(k)
end

Base.copy(algo::FixedNeighborhood) = FixedNeighborhood(algo.k)

"""
    find_neighborhood(algo::FixedNeighborhood, index::SearchGraph, item)

Finds a list of neighbors using the `FixedNeighborhood` criterion of item in the index
"""
function find_neighborhood(algo::FixedNeighborhood, index::SearchGraph, item)
    copy(search(index, item, algo.k).id)
end
