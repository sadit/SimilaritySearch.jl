#  Copyright 2016-2019 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

export IHCSearch

# Iterated Hill Climbing Search
struct IHCSearch <: LocalSearchAlgorithm
    restarts::Int
    use_local_improvement::Bool
    IHCSearch() = new(1, false)
    IHCSearch(r; use_local_improvement=false) = new(r, use_local_improvement)
end

"""
    hill_climbing(index::SearchGraph, dist::Function, q, res::KnnResult, navigation_state, nodeID::Int64, use_local_improvement::Bool)

Runs a single hill climbing search process starting in vertex `nodeID`
"""
function hill_climbing(index::SearchGraph, dist::Function, q, res::KnnResult, navigation_state, nodeID::Int64, use_local_improvement::Bool)
    omin::Int = -1
    dmin::Float32 = typemax(Float32)

    while true
        dmin = typemax(Float32)
        omin = -1
        navigation_state[nodeID] = EXPLORED
        @inbounds for childID in index.links[nodeID]
            S = get(navigation_state, childID, UNKNOWN)
            S != UNKNOWN && continue
            navigation_state[childID] = VISITED
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
    search(isearch::IHCSearch, index::SearchGraph, dist::Function, q, res::KnnResult, navigation_state, hints=EMPTY_INT_VECTOR)

Performs an iterated hill climbing search for `q`. The given `hints` are used as starting points of the search; a random
selection is performed otherwise.
"""
function search(isearch::IHCSearch, index::SearchGraph, dist::Function, q, res::KnnResult, navigation_state, hints=EMPTY_INT_VECTOR)
    n = length(index.db)
    restarts = min(isearch.restarts, n)
    if length(hints) == 0
        hints = rand(1:n, isearch.restarts)
    end

    @inbounds for start_point in hints
        # start_point = rand(range)
        S = get(navigation_state, start_point, UNKNOWN)
        if S == UNKNOWN
            navigation_state[start_point] = VISITED
            d = convert(Float32, dist(q, index.db[start_point]))
            push!(res, start_point, d)
            hill_climbing(index, dist, q, res, navigation_state, start_point, isearch.use_local_improvement)
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
