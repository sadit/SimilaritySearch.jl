# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random
export BeamSearch

struct BeamSearch <: LocalSearchAlgorithm
    bsize::Int32  # beam size

    BeamSearch() = new(3)
    BeamSearch(bsize::Integer) = new(bsize)
    BeamSearch(other::BeamSearch) =  new(other.bsize)
end

struct BeamSearchContext
    vstate::VisitedVertices
    beam::KnnResult
    hints::Vector{Int}
    BeamSearchContext(vstate, beam, hints) = new(vstate, beam, hints)
    BeamSearchContext(bsize::Integer, n::Integer, ssize::Integer=bsize) = new(VisitedVertices(), KnnResult(bsize), unique(rand(1:n, ssize)))
end

search_context(bs::BeamSearch, n::Integer, ssize::Integer=bs.bsize) = BeamSearchContext(bs.bsize, n, ssize)

function reset!(searchctx::BeamSearchContext; n=0)
    empty!(searchctx.vstate)
    empty!(searchctx.beam)

    if n > 0
        for i in eachindex(searchctx.hints)
            searchctx.hints[i] = rand(1:n)
        end
        unique!(searchctx.hints)
    end

    searchctx
end

# const BeamType = typeof((objID=Int32(0), dist=0.0))
### local search algorithm
function beam_init(bs::BeamSearch, index::SearchGraph, dist::Function, q, res::KnnResult, searchctx)
    n = length(index.db)
    @inbounds for objID in searchctx.hints
        S = get(searchctx.vstate, objID, UNKNOWN)
        if S == UNKNOWN
            searchctx.vstate[objID] = VISITED
            d = dist(q, index.db[objID])
            push!(searchctx.beam, objID, d) && push!(res, objID, d)
        end
    end
end


"""
Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bs`: the parameters of `BeamSearch`
- `index`: the local search index
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
"""
function search(bs::BeamSearch, index::SearchGraph, dist::Function, q, res::KnnResult, searchctx::BeamSearchContext)
    n = length(index.db)
    n == 0 && return res

    vstate = searchctx.vstate
    beam_init(bs, index, dist, q, res, searchctx)
    beam = searchctx.beam
    prev_score = typemax(Float64)
    # rand() < 0.3 && @info length(beam) length(searchctx.hints) searchctx.hints 
    
    @inbounds while abs(prev_score - last(beam).dist) > 0.0  # prepared to allow early stopping
        prev_score = last(beam).dist
        for prev in beam
            S = get(vstate, prev.objID, UNKNOWN)
            S == EXPLORED && continue
            vstate[prev.objID] = EXPLORED
            for childID in index.links[prev.objID]
                S = get(vstate, childID, UNKNOWN)
                if S == UNKNOWN
                    vstate[childID] = VISITED
                    d = dist(q, index.db[childID])
                    push!(res, childID, d) && push!(beam, childID, d)
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
        BeamSearch(f(gsearch.bsize)) |> fun
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
