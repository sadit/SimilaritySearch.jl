# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export VorNeighborhood

struct VorNeighborhood <: NeighborhoodAlgorithm
    base::Float64
end

function VorNeighborhood()
    return VorNeighborhood(1.1)
end

function find_neighborhood(algo::VorNeighborhood, index::SearchGraph, item)
    k = max(1, ceil(Int, log(algo.base, length(index.db))))
    N = Int32[]
    n = length(index.db)
   
    @inbounds for p in search(index, item, k)
        pobj = index.db[p.id]
        covered = false
        for nearID in N
            d = convert(Float32, evaluate(index.dist, index.db[nearID], pobj))
            if d <= p.dist
                covered = true
                break
            end
        end

        !covered && push!(N, p.id)
    end

    N
end
