# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export IHCSearch

# Iterated Hill Climbing Search
struct IHCSearch <: LocalSearchAlgorithm
    restarts::Int
    IHCSearch() = new(8)
    IHCSearch(r) = new(r)
end


struct IHCSearchContext
    vstate::VisitedVertices
    hints::Vector{Int}
    restarts::Int
    IHCSearchContext(vstate, hints) = new(vstate, hints, length(hints))
    IHCSearchContext(restarts::Integer, n::Integer) =
        new(VisitedVertices(), n == 0 ? Int32[] : unique(rand(1:n, restarts)), restarts)
end

search_context(ihc::IHCSearch, n::Integer) = IHCSearchContext(ihc.restarts, n)

function reset!(searchctx::IHCSearchContext; n=0)
    empty!(searchctx.vstate)

    if n > 0
        searchctx.restarts != length(searchctx.hints) && resize!(searchctx.hints, searchctx.restarts)
        
        for i in eachindex(searchctx.hints)
            searchctx.hints[i] = rand(1:n)
        end
        unique!(searchctx.hints)
    end

    searchctx
end

"""
    hill_climbing(index::SearchGraph, dist, q, res::KnnResult, vstate, nodeID::Int64, use_local_improvement::Bool)

Runs a single hill climbing search process starting in vertex `nodeID`
"""
function hill_climbing(index::SearchGraph, dist::Function, q, res::KnnResult, vstate, nodeID::Integer; use_local_improvement::Bool=false)
    omin::Int = -1
    dmin::Float32 = typemax(Float32)

    while true
        dmin = typemax(Float32)
        omin = -1
        vstate[nodeID] = EXPLORED
        @inbounds for childID in index.links[nodeID]
            S = get(vstate, childID, UNKNOWN)
            S != UNKNOWN && continue
            vstate[childID] = VISITED
            d = convert(Float32, dist(index.db[childID], q))
            if use_local_improvement  ## this yields to better quality but can't be tuned for early stopping
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

"""
    search(isearch::IHCSearch, index::SearchGraph, dist, q, res::KnnResult, vstate, hints=EMPTY_INT_VECTOR)

Performs an iterated hill climbing search for `q`. The given `hints` are used as starting points of the search; a random
selection is performed otherwise.
"""
function search(isearch::IHCSearch, index::SearchGraph, dist, q, res::KnnResult, searchctx)
    n = length(index.db)
    restarts = min(isearch.restarts, n)

    @inbounds for start_point in searchctx.hints
        # start_point = rand(range)
        S = get(searchctx.vstate, start_point, UNKNOWN)
        if S == UNKNOWN
            searchctx.vstate[start_point] = VISITED
            d = convert(Float32, dist(q, index.db[start_point]))
            push!(res, start_point, d)
            hill_climbing(index, dist, q, res, searchctx.vstate, start_point)
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
        IHCSearch(f(algo.restarts)) |> fun
    end
end
