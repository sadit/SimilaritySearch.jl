#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
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

export NeighborhoodSearch

struct NeighborhoodSearch <: LocalSearchAlgorithm
    candidates_size::Int
    montecarlo_size::Int
end

NeighborhoodSearch() = NeighborhoodSearch(1, 1)
NeighborhoodSearch(g::NeighborhoodSearch) = NeighborhoodSearch(g.candidates_size, g.montecarlo_size)

###
### Steady state local search
###

function neighborhood_search(gsearch::NeighborhoodSearch, index::LocalSearchIndex{T}, q::T, res::Result, tabu::MemoryType, candidates::Result) where {T,MemoryType}
    @inbounds while length(candidates) > 0
        nodeID = shift!(candidates).objID
        cov = last(res).dist

        for childID in index.links[nodeID]
            if !tabu[childID]
                d = convert(Float32, index.dist(index.db[childID], q))
                tabu[childID] = true
                if d <= cov
                    push!(candidates, childID, d) && push!(res, childID, d)
                end
            end
        end
    end
end

function search(gsearch::NeighborhoodSearch, index::LocalSearchIndex{T}, q::T, res::Result) where {T}
    n = length(index.db)
    tabu = falses(n)
    candidates = SlugKnnResult(gsearch.candidates_size)
    if isnull(index.options.oracle)
        estimate_knearest(index.db, index.dist, gsearch.candidates_size, gsearch.montecarlo_size, q, tabu, res, candidates)
    else
        estimate_from_oracle(index, q, candidates, tabu, res, get(index.options.oracle))
    end

    xtabu = Set{Int}()
    while length(candidates) > 0
        neighborhood_search(gsearch, index, q, res, tabu, candidates)
        for p in res
            if !(p.objID in xtabu)
                push!(xtabu, p.objID)
                push!(candidates, p.objID, p.dist)
            end
        end
    end

    return res
end

function opt_create_random_state(gsearch::NeighborhoodSearch, max_value)
    a = max(1, round(Int, rand() * max_value))
    b = max(1, round(Int, rand() * max_value))
    return NeighborhoodSearch(a, b)
end

function opt_expand_neighborhood(fun, gsearch::NeighborhoodSearch, n::Int, iter::Int)
    f(x, w) = max(1, x + w)
    g(x) = max(1, x + ceil(Int, (rand()-0.5) * log2(n)))

    if iter == 1
        for i in 1:8
            opt_create_random_state(gsearch, ceil(Int, log2(n))) |> fun
        end

        NeighborhoodSearch(gsearch.candidates_size |> g, gsearch.montecarlo_size |> g) |> fun
        NeighborhoodSearch(gsearch.candidates_size |> g, gsearch.montecarlo_size) |> fun
        NeighborhoodSearch(gsearch.candidates_size, gsearch.montecarlo_size |> g) |> fun
    end

    w = 2
    while w <= div(32,iter)
        NeighborhoodSearch(f(gsearch.candidates_size,  w), gsearch.montecarlo_size) |> fun
        NeighborhoodSearch(f(gsearch.candidates_size, -w), gsearch.montecarlo_size) |> fun
        NeighborhoodSearch(gsearch.candidates_size, f(gsearch.montecarlo_size,  w)) |> fun
        NeighborhoodSearch(gsearch.candidates_size, f(gsearch.montecarlo_size, -w)) |> fun
        w += w
    end
end
