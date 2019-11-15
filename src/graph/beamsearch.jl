# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random
export BeamSearch

struct BeamSearch <: LocalSearchAlgorithm
    ssize::Int32  # sample size
    bsize::Int32  # beam size

    BeamSearch() = new(1, 1)
    BeamSearch(a::Integer, b::Integer) = new(a, b)
    BeamSearch(other::BeamSearch) =  new(other.ssize, other.bsize)
end

# const BeamType = typeof((objID=Int32(0), dist=0.0))
### local search algorithm
function beam_init(bs::BeamSearch, index::SearchGraph, dist::Function, q, res::KnnResult, navigation_state, hints)
    beam = KnnResult(bs.bsize)
    n = length(index.db)
    # range = 1:n
    # @inbounds for i in 1:bs.bsize
    @inbounds for objID in hints
        S = get(navigation_state, objID, UNKNOWN)
        # S = navigation_state[objID]
        if S == UNKNOWN
            navigation_state[objID] = VISITED
            d = dist(q, index.db[objID])
            push!(beam, objID, d) && push!(res, objID, d)
        end
    end

    beam
end

"""
Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bs`: the parameters of `BeamSearch`
- `index`: the local search index
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
"""
function search(bs::BeamSearch, index::SearchGraph, dist::Function, q, res::KnnResult, navigation_state, hints=EMPTY_INT_VECTOR)
    n = length(index.db)
    n == 0 && return res
    if length(hints) == 0
        hints = rand(1:n, bs.bsize)
    end

    beam = beam_init(bs, index, dist, q, res, navigation_state, hints)
    prev_score = typemax(Float64)
    
    @inbounds while abs(prev_score - last(beam).dist) > 0.0  # prepared to allow early stopping
        prev_score = last(beam).dist

        for prev in beam
            cov = last(beam).dist
            S = get(navigation_state, prev.objID, UNKNOWN)
            # S = navigation_state[prev.objID]
            S == EXPLORED && continue
            navigation_state[prev.objID] = EXPLORED
            for childID in index.links[prev.objID]
                S = get(navigation_state, childID, UNKNOWN)
                # S = navigation_state[childID]
                if S == UNKNOWN
                    navigation_state[childID] = VISITED
                    d = dist(q, index.db[childID])

                    if d <= cov
                        push!(beam, childID, d) && push!(res, childID, d)
                    end
                end
            end
        end
    end

    res
end

function opt_expand_neighborhood(fun, gsearch::BeamSearch, n::Integer, iter::Integer, probes::Integer)
    #f_(w) = ceil(Int, w * (rand() - 0.5))
    #f(x, w) = max(1, x + #f_(w))

    # g(x) = max(1, x + ceil(Int, (rand()-0.5) * log2(n)))
    logn = ceil(Int, log(2, n+1))
    probes = probes == 0 ? logn : probes
    f(x) = max(1, x + rand(-logn:logn))
    for i in 1:probes
        BeamSearch(f(gsearch.ssize), f(gsearch.bsize)) |> fun
    end
    ### f(x, w) = max(1, x + w)
    ### w = 1
    ### while w <= logn  ## log log n
    ###    BeamSearch(f(gsearch.ssize,  w), gsearch.bsize) |> fun
    ###    BeamSearch(f(gsearch.ssize,  -w), gsearch.bsize) |> fun
    ###    BeamSearch(gsearch.ssize, f(gsearch.bsize, w)) |> fun
    ###    BeamSearch(gsearch.ssize, f(gsearch.bsize, -w)) |> fun
    ###    w += w
    ### end
end
