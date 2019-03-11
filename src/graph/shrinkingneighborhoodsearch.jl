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

export ShrinkingNeighborhoodSearch

struct ShrinkingNeighborhoodSearch <: LocalSearchAlgorithm
    candidates_size::Int
    montecarlo_size::Int
    shrinking_cov::Float64
end

ShrinkingNeighborhoodSearch() = ShrinkingNeighborhoodSearch(1, 1, 1.0f0)
ShrinkingNeighborhoodSearch(other::ShrinkingNeighborhoodSearch) = ShrinkingNeighborhoodSearch(other.candidates_size, other.montecarlo_size, other.shrinking_cov)

function shrinking_neighborhood_search(nsearch::ShrinkingNeighborhoodSearch, index::SearchGraph{T}, dist::Function, q::T, res::R, tabu::MemoryType, candidates::Result) where {T,R <: Result,MemoryType}
    @inbounds while length(candidates) > 0
        best = popfirst!(candidates)
        cov = last(res).dist
        if best.dist > cov * nsearch.shrinking_cov
            return
        end

        for childID in index.links[best.objID]
            if !tabu[childID]
                d = convert(Float32, dist(index.db[childID], q))
                tabu[childID] = true
                if d <= cov
                    push!(candidates, childID, d) && push!(res, childID, d)
                end
            end
        end
    end
end

function search(nsearch::ShrinkingNeighborhoodSearch, index::SearchGraph{T}, dist::Function, q::T, res::R; oracle=nothing) where {T,R <: Result}
    n = length(index.db)
    tabu = falses(n)
    candidates = KnnResult(nsearch.candidates_size)

    if oracle == nothing
        estimate_knearest(dist, index.db, nsearch.candidates_size, nsearch.montecarlo_size, q, tabu, res, candidates)
    else
        estimate_from_oracle(index, dist, q, candidates, tabu, res, oracle)
    end

    xtabu = Set{Int}()

    while length(candidates) > 0
        shrinking_neighborhood_search(nsearch, index, dist, q, res, tabu, candidates)
        empty!(candidates)
        for p in res
            if !(p.objID in xtabu)
                push!(xtabu, p.objID)
                push!(candidates, p.objID, p.dist)
            end
        end
    end

    return res
end

function opt_create_random_state(nsearch::ShrinkingNeighborhoodSearch, max_value)
    a = max(1, rand() * max_value |> round |> Int)
    b = max(1, rand() * max_value |> round |> Int)
    c = 1 + (rand() - 0.5) / 5
    return ShrinkingNeighborhoodSearch(a, b, c)
end

function opt_expand_neighborhood(fun, algo::ShrinkingNeighborhoodSearch, n::Int, iter::Int)
    f(x::Int, w) = max(1, x + w)
    f(x::Float64, w) = max(1, x + sign(w)*0.01)
    # g(x::Int) = max(1, x + ceil(Int, (rand()-0.5) * log2(n)))
    g(x::Int) = max(1, x + ceil(Int, (rand()-0.5) * 32))
    g(x::Float64) = max(1, x + (rand()-0.5) / 10)

    if iter == 1
        for i in 1:8
            opt_create_random_state(algo, ceil(Int, log2(n))) |> fun
        end
    end
    #if iter <= 3
        ShrinkingNeighborhoodSearch(algo.candidates_size |> g, algo.montecarlo_size |> g, algo.shrinking_cov |> g) |> fun
        ShrinkingNeighborhoodSearch(algo.candidates_size |> g, algo.montecarlo_size, algo.shrinking_cov) |> fun
        ShrinkingNeighborhoodSearch(algo.candidates_size, algo.montecarlo_size |> g, algo.shrinking_cov) |> fun
        ShrinkingNeighborhoodSearch(algo.candidates_size, algo.montecarlo_size, algo.shrinking_cov |> g) |> fun
    #end

    w = 2
    while w <= div(32,iter)
        ShrinkingNeighborhoodSearch(f(algo.candidates_size,  w), algo.montecarlo_size, algo.shrinking_cov) |> fun
        ShrinkingNeighborhoodSearch(f(algo.candidates_size, -w), algo.montecarlo_size, algo.shrinking_cov) |> fun
        ShrinkingNeighborhoodSearch(algo.candidates_size, f(algo.montecarlo_size,  w), algo.shrinking_cov) |> fun
        ShrinkingNeighborhoodSearch(algo.candidates_size, f(algo.montecarlo_size, -w), algo.shrinking_cov) |> fun
        ShrinkingNeighborhoodSearch(algo.candidates_size, algo.montecarlo_size, f(algo.shrinking_cov,  w)) |> fun
        ShrinkingNeighborhoodSearch(algo.candidates_size, algo.montecarlo_size, f(algo.shrinking_cov, -w)) |> fun
        w += w
    end
end
