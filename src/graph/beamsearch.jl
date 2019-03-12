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

using Random
export BeamSearch

struct BeamSearch <: LocalSearchAlgorithm
    ssize::Int32  # sample size
    bsize::Int32  # beam size

    BeamSearch() = new(1, 1)
    BeamSearch(a::Integer, b::Integer) = new(a, b)
    BeamSearch(other::BeamSearch) =  new(other.ssize, other.bsize)
end

@enum VertexSearchState begin
    UNKNOWN = 0
    VISITED = 1
    EXPLORED = 2
end

# const BeamType = typeof((objID=Int32(0), dist=0.0))
### local search algorithm

"""
    beam_search(bsearch::BeamSearch, index::SearchGraph{T}, q::T, res::Result, tabu::MemoryType, oracle::Union{Function,Nothing}) where {T, MemoryType}

Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bsearch`: the parameters of `BeamSearch`
- `index`: the local search index
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
"""
function beam_search(bsearch::BeamSearch, index::SearchGraph{T}, dist::Function, q::T, res::Result) where T
    n = length(index.db)
    m = ceil(Int, log(2, n+1))
    # m = min(n, bsearch.ssize)
    exploration = Dict{Int32,VertexSearchState}()
    # exploration = zeros(Int8, n)
    beam = KnnResult(bsearch.bsize)
    # B = BeamType[]
    for i in 1:m
        objID = rand(1:n)
        S = get(exploration, objID, UNKNOWN)
        # S = exploration[objID]
        if S == UNKNOWN
            exploration[objID] = VISITED
            d = convert(Float32, dist(q, index.db[objID]))
            # push!(B, (objID=objID, dist=d))
            push!(beam, objID, d) && push!(res, objID, d)
        end
    end

    # sort!(B, by=x -> x.dist)
    # length(B) > bsearch.bsize && resize!(B, bsearch.bsize)
    prev_score = typemax(Float64)
    
    while abs(prev_score - last(beam).dist) > 0.0  # prepared to allow early stopping
        prev_score = last(beam).dist

        for prev in beam
            cov = last(beam).dist
            S = get(exploration, prev.objID, UNKNOWN)
            # S = exploration[prev.objID]
            S == EXPLORED && continue
            exploration[prev.objID] = EXPLORED
            for childID in index.links[prev.objID]
                S = get(exploration, childID, UNKNOWN)
                # S = exploration[childID]
                if S == UNKNOWN
                    exploration[childID] = VISITED
                    # d = convert(Float32, dist(q, index.db[childID]))
                    d = dist(q, index.db[childID])
                    if d <= cov
                        push!(beam, childID, d) && push!(res, childID, d)
                    end
                end
            end
        end

        #sort!(B, by=x -> x.dist)
        #length(B) > bsearch.bsize && resize!(B, bsearch.bsize)
    end

    res
end

function search(bsearch::BeamSearch, index::SearchGraph{T}, dist::Function, q::T, res::Result; oracle=nothing) where T
    length(index.db) == 0 && return res
    # if oracle == nothing
    #    estimate_knearest(dist, index.db, bsearch.candidates_size, bsearch.ssize, q, tabu, res, beam)
    # else
    #    estimate_from_oracle(index, dist, q, beam, tabu, res, oracle)
    # end

    beam_search(bsearch, index, dist, q, res)
end

function opt_create_random_state(algo::BeamSearch, max_value)
    a = max(1, rand() * max_value |> round |> Int)
    b = max(1, rand() * max_value |> round |> Int)
    return BeamSearch(a, b)
end

function opt_expand_neighborhood(fun, gsearch::BeamSearch, n::Int, iter::Int)
    f(x, w) = max(1, x + w)
    # g(x) = max(1, x + ceil(Int, (rand()-0.5) * log2(n)))
    logn = ceil(Int, log(2, n+1))

    if iter == 0
        for i in 1:logn
            opt_create_random_state(gsearch, logn) |> fun
        end
    end

    w = 1
    while w <= logn  ## log log n
        w1 = ceil(Int, w*(rand()-0.5))
        w2 = ceil(Int, w*(rand()-0.5))
        BeamSearch(f(gsearch.ssize,  w1), gsearch.bsize) |> fun
        BeamSearch(gsearch.ssize, f(gsearch.bsize, w2)) |> fun
        w += w
    end
end
