# This file is a part of SimilaritySearch.jl

export IHCSearch

# Iterated Hill Climbing Search

"""
    IHCSearch(hints::Vector; restarts=length(hints), localimprovements=false)
    IHCSearch(restarts::Integer=20; hints=Int32[], localimprovements=false)
    IHCSearch(hints, restarts, localimprovements)

IHCSearch is an iterated hill climbing algorithma, a local search algorithm. It greedily navigates the search graph
and restart the search `restarts` times.
Multithreading applications must have copies of this object due to shared cache objects.

- `restarts`: The number of restarts.
- `hints`: An initial hint for the exploration (if it is not empty, then superseeds the initial points of the random starting points).
- `localimprovements`: An experimental technique that if it is true it will achieve very high quality results, at cost of increasing searching time.
"""
@with_kw mutable struct IHCSearch <: LocalSearchAlgorithm
    hints::Vector{Int32} = Int32[]
    restarts::Int32 = 16
    localimprovements::Bool = false
end

Base.copy(ihc::IHCSearch; hints=ihc.hints, restarts=ihc.restarts, localimprovements=ihc.localimprovements) =
    IHCSearch(; hints, restarts, localimprovements)

function Base.copy!(dst::IHCSearch, src::IHCSearch)
    dst.restarts = src.restarts
    dst.hints = src.hints
    dst.localimprovements = src.localimprovements
end

"""
    hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer)

Runs a single hill climbing search process starting in vertex `nodeID`
"""
function hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer, vstate)
    omin::Int32 = -1
    dmin::Float32 = typemax(Float32)
    localimprovements = isearch.localimprovements
    while true
        dmin = typemax(Float32)
        omin = -1
        vstate[nodeID] = EXPLORED
        @inbounds for childID in keys(index.links[nodeID])
            S = get(vstate, childID, UNKNOWN)
            S != UNKNOWN && continue
            vstate[childID] = VISITED
            d = convert(Float32, evaluate(index.dist, index[childID], q))
            if localimprovements  ## this yields to better quality but has no early stopping
                push!(res, childID, d)
                if d < dmin
                    dmin = d
                    omin = childID
                end
            else
                if push!(res, childID, d) && d < dmin
                    dmin = d
                    omin = childID
                end
            end
        end

        if omin < 0
            break
        else
            nodeID = omin
        end
    end
end

function searchat(isearch::IHCSearch, index::SearchGraph, q, res, startpoint, vstate)
    S = get(vstate, startpoint, UNKNOWN)
    if S == UNKNOWN
        vstate[startpoint] = VISITED
        d = convert(Float32, evaluate(index.dist, q, index[startpoint]))
        push!(res, startpoint, d)
        hill_climbing(isearch, index, q, res, startpoint, vstate)
    end
end

"""
    search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)

Performs an iterated hill climbing search for `q`.
"""
function search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)
    n = length(index)

    for startpoint in hints
        searchat(isearch, index, q, res, startpoint, vstate)
    end

    if length(vstate) == 0
        _range = 1:n
        for i in 1:min(isearch.restarts, n)
            startpoint = rand(_range)
            searchat(isearch, index, q, res, startpoint, vstate)
        end
    end

    res
end
