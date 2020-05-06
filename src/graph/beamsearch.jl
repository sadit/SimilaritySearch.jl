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
    ssize::Int
    BeamSearchContext(vstate; beam=KnnResult(64), hints=Int32[]) =
        new(vstate, beam, hints, length(hints))
    BeamSearchContext(bsize::Integer, n::Integer, ssize::Integer=bsize) =
        new(VisitedVertices(n), KnnResult(bsize), n == 0 ? Int32[] : unique(rand(1:n, ssize)), ssize)
end

search_context(bs::BeamSearch, n::Integer, ssize::Integer=bs.bsize) = BeamSearchContext(bs.bsize, n, ssize)

function reset!(searchctx::BeamSearchContext; n=0)
    # @info (typeof(searchctx.vstate), length(searchctx.vstate), n)
    if searchctx.vstate isa AbstractVector
        n > length(searchctx.vstate) && resize!(searchctx.vstate, n)
        fill!(searchctx.vstate, 0)
    else
        empty!(searchctx.vstate)
    end

    empty!(searchctx.beam)

    if n > 0
        searchctx.ssize != length(searchctx.hints) && resize!(searchctx.hints, searchctx.ssize)
        
        for i in eachindex(searchctx.hints)
            searchctx.hints[i] = rand(1:n)
        end
        unique!(searchctx.hints)
    end

    searchctx
end

# const BeamType = typeof((objID=Int32(0), dist=0.0))
### local search algorithm
function beam_init(bs::BeamSearch, index::SearchGraph, dist, q, res::KnnResult, searchctx)
    @inbounds for objID in searchctx.hints
        if getstate(searchctx.vstate, objID) == UNKNOWN
            setstate!(searchctx.vstate, objID, VISITED)
            d = dist(q, index.db[objID])
            push!(res, objID, d)
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
function search(bs::BeamSearch, index::SearchGraph, dist, q, res::KnnResult, searchctx::BeamSearchContext)
    n = length(index.db)
    n == 0 && return res

    vstate = searchctx.vstate
    beam_init(bs, index, dist, q, res, searchctx)
    beam = searchctx.beam
    prev_score = typemax(Float32)
    
    @inbounds while abs(prev_score - last(res).dist) > 0.0  # prepared to allow early stopping
        prev_score = last(res).dist
        nn = first(res)
        push!(beam, nn.objID, nn.dist)

        while length(beam) > 0
            prev = popfirst!(beam)
            getstate(vstate, prev.objID) == EXPLORED && continue
            setstate!(vstate, prev.objID, EXPLORED)

            for childID in index.links[prev.objID]
                if getstate(vstate, childID) == UNKNOWN
                    setstate!(vstate, childID, VISITED)
                    d = dist(q, index.db[childID])
                    push!(res, childID, d) && push!(beam, childID, d)
                    #d <= last(res) && push!(beam, childID, d)
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
