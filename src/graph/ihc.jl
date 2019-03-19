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

function hill_climbing(index::SearchGraph{T}, dist::Function, q::T, res::KnnResult, navigation_state, nodeID::Int64, use_local_improvement::Bool) where T
    omin::Int = -1
    dmin::Float32 = typemax(Float32)
    # @info "STARTING HILL CLIMBING"
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

        # @info dmin
        if omin < 0
            break
        else
            nodeID = omin
        end
    end
end

function search(isearch::IHCSearch, index::SearchGraph{T}, dist::Function, q::T, res::KnnResult, navigation_state, hints=EMPTY_INT_VECTOR) where T
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

function opt_expand_neighborhood(fun, algo::IHCSearch, n::Int, iter::Int)
    f_(w) = ceil(Int, w * (rand() - 0.5))
    f(x, w) = max(1, x + f_(w))
    logn = ceil(Int, log(2, n+1))
    w = 1

    while w <= logn  ## log log n
        IHCSearch(f(algo.restarts, w)) |> fun
        w += w
    end
end
