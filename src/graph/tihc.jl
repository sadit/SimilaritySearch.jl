#  Copyright 2019 Eric S. Tellez <eric.tellez@infotec.mx>
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

using Random
export TIHCSearch

struct TIHCSearch <: LocalSearchAlgorithm
    ssize::Int32 # sample size
    restarts::Int32
    use_local_improvement::Bool
    TIHCSearch() = new(1, 1, false)
    TIHCSearch(ssize, restarts; use_local_improvement=false) = new(Int32(ssize), Int32(restarts), use_local_improvement)
 end

function randomsearch(index::SearchGraph{T}, dist::Function, q::T, res::KnnResult, best::KnnResult, navigation_state, ssize::Integer) where T
    n = length(index.db)
    ssize = min(n, max(maxlength(res), ssize))
    range = 1:n
    @inbounds for i in 1:ssize
        # while abs(prev_dist - last(res).dist) < rs.tol
        objID = rand(range)
        S = get(navigation_state, objID, UNKNOWN)
        if S == UNKNOWN
            navigation_state[objID] = VISITED
            d = convert(Float32, dist(q, index.db[objID]))
            push!(res, objID, d)
            push!(best, objID, d)
        end
    end

    res
end

function search(isearch::TIHCSearch, index::SearchGraph{T}, dist::Function, q::T, res::KnnResult, navigation_state, hints=EMPTY_INT_VECTOR) where T
    n = length(index.db)
    restarts = min(isearch.restarts, n)
    best = KnnResult(1)
    # if length(hints) == 0 hints = rand(1:n, bs.bsize) end
 
    @inbounds for i in 1:restarts
        empty!(best)
        randomsearch(index, dist, q, res, best, navigation_state, isearch.ssize)
        length(best) == 0 && break  # all items in the neighborhood were visited
        # @show i, restarts, isearch, best, length(best)
        start_point = first(best).objID
        navigation_state[start_point] = EXPLORED
        d = convert(Float32, dist(q, index.db[start_point]))
        push!(res, start_point, d)
        hill_climbing(index, dist, q, res, navigation_state, start_point, isearch.use_local_improvement)
    end

    res
end

function opt_expand_neighborhood(fun, algo::TIHCSearch, n::Int, iter::Int)
    f_(w) = ceil(Int, w * (rand() - 0.5))
    f(x, w) = max(1, x + f_(w))
    logn = ceil(Int, log(2, n+1))
    w = 1

    while w <= logn  ## log log n
        #TIHCSearch(min(200, n), 1, use_local_improvement=true) |> fun
        TIHCSearch(algo.ssize, f(algo.restarts, w)) |> fun
        TIHCSearch(f(algo.ssize, w), algo.restarts) |> fun
        w += w
    end
end

