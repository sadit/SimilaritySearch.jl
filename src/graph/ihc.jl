# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export IHCSearch

# Iterated Hill Climbing Search

"""
    IHCSearch(hints::Vector; restarts=length(hints), localimprovements=false)
    IHCSearch(restarts::Integer=20; hints=Int32[], localimprovements=false)
    IHCSearch(hints, restarts, localimprovements, vstate)

IHCSearch is an iterated hill climbing algorithma, a local search algorithm. It greedily navigates the search graph
and restart the search `restarts` times.
Multithreading applications must have copies of this object due to shared cache objects.

- `restarts`: The number of restarts.
- `hints`: An initial hint for the exploration (if it is not empty, then superseeds the initial points of the random starting points).
- `localimprovements`: An experimental technique that if it is true it will achieve very high quality results, at cost of increasing searching time.
- `vstate`: A cache object for reducing memory allocations
"""
@with_kw mutable struct IHCSearch <: LocalSearchAlgorithm
    hints::Vector{Int32} = Int32[]
    restarts::Int32 = 16
    localimprovements::Bool = false
    vstate::VisitedVertices = VisitedVertices()
end

Base.copy(ihc::IHCSearch; hints=ihc.hints, restarts=ihc.restarts, localimprovements=ihc.localimprovements, vstate=VisitedVertices()) =
    IHCSearch(; hints, restarts, localimprovements, vstate)

function Base.copy!(dst::IHCSearch, src::IHCSearch)
    dst.restarts = src.restarts
    dst.vstate = src.vstate
    dst.hints = src.hints
    dst.localimprovements = src.localimprovements
end

"""
    hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer)

Runs a single hill climbing search process starting in vertex `nodeID`
"""
function hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer)
    omin::Int32 = -1
    dmin::Float32 = typemax(Float32)
    vstate = isearch.vstate
    localimprovements = isearch.localimprovements
    while true
        dmin = typemax(Float32)
        omin = -1
        vstate[nodeID] = EXPLORED
        @inbounds for childID in index.links[nodeID]
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

function searchat(isearch::IHCSearch, index::SearchGraph, q, res, startpoint)
    S = get(isearch.vstate, startpoint, UNKNOWN)
    if S == UNKNOWN
        isearch.vstate[startpoint] = VISITED
        d = convert(Float32, evaluate(index.dist, q, index[startpoint]))
        push!(res, startpoint, d)
        hill_climbing(isearch, index, q, res, startpoint)
    end
end

"""
    search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, hints)

Performs an iterated hill climbing search for `q`.
"""
function search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, hints)
    n = length(index)
    empty!(isearch.vstate)
    _range = 1:n
    if length(hints) == 0
         @inbounds for i in 1:min(isearch.restarts, n)
            startpoint = rand(_range)
            searchat(isearch, index, q, res, startpoint)
        end
    else
        @inbounds for startpoint in hints
            searchat(isearch, index, q, res, startpoint)
        end
    end

    res
end

"""
    opt_expand_neighborhood(fun, ihc::IHCSearch, n::Integer, iter::Integer, probes::Integer)

Generates configurations of the IHCSearch that feed the `optimize!` function (internal function)
"""
function opt_expand_neighborhood(fun, ihc::IHCSearch, n::Integer, iter::Integer, probes::Integer)
    logn = ceil(Int, log(2, n+1))
    probes = probes == 0 ? logn : probes
    f(x) = max(1, x + rand(-logn:logn))

    for i in 1:probes
        copy(ihc, restarts=f(ihc.restarts)) |> fun
    end
end
