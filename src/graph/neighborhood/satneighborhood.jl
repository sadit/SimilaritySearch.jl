# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export SatNeighborhood

struct SatNeighborhood <: NeighborhoodAlgorithm
    k::Int
    near::KnnResult
end

function SatNeighborhood()
    return SatNeighborhood(32, KnnResult(1))
end

function find_neighborhood(algo::SatNeighborhood, index::SearchGraph, item)
    near = algo.near
    N = Int32[]
    @inbounds for p in search(index, item, algo.k)
        pobj = index.db[p.id]
        empty!(near)
        push!(near, zero(Int32), p.dist)
        for nearID in N
            d = evaluate(index.dist, index.db[nearID], pobj)
            push!(near, nearID, d)
        end

        if first(near).id == 0
            push!(N, p.id)
        end
    end

    N
end
