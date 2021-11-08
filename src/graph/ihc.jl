# This file is a part of SimilaritySearch.jl

export IHCSearch

# Iterated Hill Climbing Search

"""
    IHCSearch(; restarts=length(hints))
    
IHCSearch is an iterated hill climbing algorithma, a local search algorithm. It greedily navigates the search graph
and restart the search `restarts` times.

- `restarts`: The number of restarts.
"""
@with_kw mutable struct IHCSearch <: LocalSearchAlgorithm
    restarts::Int32 = 32
end

Base.copy(ihc::IHCSearch; restarts=ihc.restarts) =
    IHCSearch(; restarts)

function Base.copy!(dst::IHCSearch, src::IHCSearch)
    dst.restarts = src.restarts
end

"""
    hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer)

Runs a single hill climbing search process starting in vertex `nodeID`
"""
function hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer, vstate, visited_)
    imin = 0
 
    while true
        dmin = typemax(Float32)
        imin = 0
        @inbounds for childID in keys(index.links[nodeID])
            visited(vstate, childID) && continue
            visit!(vstate, childID)
            visited_ += 1
            d = convert(Float32, evaluate(index.dist, index[childID], q))
    
            if push!(res, childID, d) && d < dmin
                dmin = d
                imin = childID
            end
        end

        imin === 0 && break
        nodeID = imin
    end

    visited_
end

function searchat(isearch::IHCSearch, index::SearchGraph, q, res, startpoint, vstate, visited_)
    if !visited(vstate, startpoint)
        visit!(vstate, startpoint)
        visited_ += 1

        d = convert(Float32, evaluate(index.dist, q, index[startpoint]))
        push!(res, startpoint, d)
        visited_ = hill_climbing(isearch, index, q, res, startpoint, vstate, visited_)
    end

    visited_
end

"""
    search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)

Performs an iterated hill climbing search for `q`.
"""
function search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)
    n = length(index)
    visited_ = 0
    for startpoint in hints
        visited_ = searchat(isearch, index, q, res, startpoint, vstate, visited_)
    end

    if length(res) == 0
        _range = 1:n
        for i in 1:min(isearch.restarts, n)
            startpoint = rand(_range)
            visited_ = searchat(isearch, index, q, res, startpoint, vstate, visited_)
        end
    end

    res, visited_
end
