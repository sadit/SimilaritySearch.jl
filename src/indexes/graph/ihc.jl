#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http:#www.apache.org/licenses/LICENSE-2.0
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
    montecarlo_size::Int
end

IHCSearch() = IHCSearch(1, 1)
IHCSearch(ihc::IHCSearch) = IHCSearch(ihc.restarts, ihc.montecarlo_size)

function greedy_search_with_tabu(isearch::IHCSearch, index::LocalSearchIndex{T}, q::T, res::Result, tabu::MemoryType, nodeID::Int64) where {T,MemoryType}
    omin::Int=-1
    dmin::Float32 = typemax(Float32)

    while true
        dmin = typemax(Float32)
        omin = -1
        @inbounds for childID in index.links[nodeID]
            if ! tabu[childID]
                d = convert(Float32, index.dist(index.db[childID], q))
                tabu[childID] = true

                push!(res, childID, d)
                if d < dmin
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

function search(isearch::IHCSearch, index::LocalSearchIndex{T}, q::T, res::Result) where {T}
    n = length(index.db)
    tabu = falses(n)
    restarts = min(isearch.restarts, ceil(Int, log2(n)))

    for i=1:restarts
        candidates = estimate_knearest(index.db, index.dist, 1, isearch.montecarlo_size, q, tabu, res)
        if length(candidates) > 0
            greedy_search_with_tabu(isearch, index, q, res, tabu, first(candidates).objID)
        end
    end

    return res
end

function opt_expand_neighborhood(fun, algo::IHCSearch, n::Int, iter::Int)
    f(x, w) = max(1, x + w)
    g(x) = max(1, x + ceil(Int, (rand()-0.5) * log2(n)))

    if iter <= 3
        IHCSearch(algo.restarts |> g, algo.montecarlo_size |> g) |> fun
        IHCSearch(algo.restarts |> g, algo.montecarlo_size) |> fun
        IHCSearch(algo.restarts, algo.montecarlo_size |> g) |> fun
    end

    w = 2
    while w <= div(32,iter)
        IHCSearch(f(algo.restarts,  w), algo.montecarlo_size) |> fun
        IHCSearch(f(algo.restarts, -w), algo.montecarlo_size) |> fun
        IHCSearch(algo.restarts, f(algo.montecarlo_size,  w)) |> fun
        IHCSearch(algo.restarts, f(algo.montecarlo_size, -w)) |> fun
        w += w
    end
end
