# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export IHCSearch

# Iterated Hill Climbing Search
struct IHCSearch <: LocalSearchAlgorithm
    hints::Vector{Int32}
    vstate::VisitedVertices
    restarts::Int32
    use_local_improvement::Bool
end

function IHCSearch(hints::Vector, restarts=length(hints); use_local_improvement=false)
    IHCSearch(hints, VisitedVertices(), restarts, use_local_improvement)
end

function IHCSearch(restarts::Integer=20; use_local_improvement=false)
    IHCSearch(Int32[], VisitedVertices(), restarts, use_local_improvement)
end

IHCSearch(ihc::IHCSearch; hints=ihc.hints, vstate=VisitedVertices(), restarts=ihc.restarts, use_local_improvement=ihc.use_local_improvement) =
    IHCSearch(hunts, vstate, restarts, use_local_improvement)

"""
    hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer)

Runs a single hill climbing search process starting in vertex `nodeID`
"""
function hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer)
    omin::Int32 = -1
    dmin::Float32 = typemax(Float32)
    vstate = isearch.vstate
    use_local_improvement = isearch.use_local_improvement
    while true
        dmin = typemax(Float32)
        omin = -1
        vstate[nodeID] = EXPLORED
        @inbounds for childID in index.links[nodeID]
            S = get(vstate, childID, UNKNOWN)
            S != UNKNOWN && continue
            vstate[childID] = VISITED
            d = convert(Float32, evaluate(index.dist, index.db[childID], q))
            if use_local_improvement  ## this yields to better quality but has no early stopping
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

function search_at(isearch::IHCSearch, index::SearchGraph, q, res, startpoint)
    S = get(isearch.vstate, startpoint, UNKNOWN)
    if S == UNKNOWN
        isearch.vstate[startpoint] = VISITED
        d = convert(Float32, evaluate(index.dist, q, index.db[startpoint]))
        push!(res, startpoint, d)
        hill_climbing(isearch, index, q, res, startpoint)
    end
end

"""
    search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult)

Performs an iterated hill climbing search for `q`.
"""
function search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult)
    n = length(index.db)
    restarts = min(isearch.restarts, n)
    empty!(isearch.vstate)
    randomrange = 1:n

    if length(isearch.hints) == 0
         @inbounds for i in 1:isearch.restarts
            startpoint = rand(randomrange)
            search_at(isearch, index, q, res, startpoint)
        end
    else
        @inbounds for startpoint in isearch.hints
            search_at(isearch, index, q, res, startpoint)
        end
    end

    res
end

"""
    opt_expand_neighborhood(fun, algo::IHCSearch, n::Integer, iter::Integer, probes::Integer)

Generates configurations of the IHCSearch that feed the `optimize!` function (internal function)
"""
function opt_expand_neighborhood(fun, algo::IHCSearch, n::Integer, iter::Integer, probes::Integer)
    logn = ceil(Int, log(2, n+1))
    probes = probes == 0 ? logn : probes
    f(x) = max(1, x + rand(-logn:logn))

    for i in 1:probes
        IHCSearch(n, f(algo.restarts)) |> fun
    end
end
