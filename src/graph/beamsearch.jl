# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random
export BeamSearch

"""
    BeamSearch(bsize::Integer=16, ssize=bsize; hints=Int32[], beam=KnnResult(bsize), vstate=VisitedVertices())

BeamSearch is an iteratively improving local search algorithm that explores the graph using blocks of `bsize` elements and neighborhoods at the time.
Multithreading applications must have copies of this object due to shared cache objects.

- `ssize`: The size of the first sampling.
- `hints`: An initial hint for the exploration (if it is not empty, then superseeds `ssize` and the initial sampling).
- `beam`: A cache object for reducing memory allocations
- `vstate`: A cache object for reducing memory allocations
"""
mutable struct BeamSearch <: LocalSearchAlgorithm
    hints::Vector{Int32}
    bsize::Int32  # size of the search beam
    ssize::Int32  # size of the first search beam (initial sampling)
    beam::KnnResult
    vstate::VisitedVertices
end

function BeamSearch(bsize::Integer=16, ssize=bsize; hints=Int32[], beam=KnnResult(bsize), vstate=VisitedVertices())
    BeamSearch(hints, bsize, ssize, beam, vstate)
end

Base.copy(bsearch::BeamSearch;
        hints=bsearch.hints,
        bsize=bsearch.bsize,
        ssize=bsearch.ssize,
        beam=KnnResult(Int(bsize)),
        vstate=VisitedVertices()
    ) = BeamSearch(hints, bsize, ssize, beam, vstate)

function Base.copy!(dst::BeamSearch, src::BeamSearch)
    dst.hints = src.hints
    dst.bsize = src.bsize
    dst.ssize = src.ssize
    dst.beam = src.beam
    dst.vstate = src.vstate
end

Base.string(s::BeamSearch) = """{BeamSearch: bsize=$(s.bsize), ssize=$(s.ssize), hints=$(length(s.hints))"""

# const BeamType = typeof((objID=Int32(0), dist=0.0))
### local search algorithm

function beamsearch_queue(index::SearchGraph, q, res::KnnResult, objID, vstate)
    if getstate(vstate, objID) === UNKNOWN
        setstate!(vstate, objID, VISITED)
        @inbounds d = evaluate(index.dist, q, index.db[objID])
        push!(res, objID, d)
    end
end

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)
    empty!(vstate)

    if length(hints) == 0
        _range = 1:length(index.db)
         @inbounds for i in 1:bs.ssize
            objID = rand(_range)
            beamsearch_queue(index, q, res, objID, vstate)
        end
    else
        for objID in hints
            beamsearch_queue(index, q, res, objID, vstate)
        end
    end
end

function beamsearch_inner(index::SearchGraph, q, res::KnnResult, beam::KnnResult, vstate)
    while length(beam) > 0
        prev_id, prev_dist = popfirst!(beam)
        getstate(vstate, prev_id) === EXPLORED && continue
        setstate!(vstate, prev_id, EXPLORED)
        @inbounds for childID in index.links[prev_id]
            if getstate(vstate, childID) === UNKNOWN
                setstate!(vstate, childID, VISITED)
                d = evaluate(index.dist, q, index.db[childID])
                push!(res, childID, d) && push!(beam, childID, d)
                #d <= 0.9 * farthest(res).dist && push!(beam, childID, d)
            end
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
function search(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints)
    n = length(index.db)
    n == 0 && return res

    empty!(bs.beam, bs.bsize)
    beamsearch_init(bs, index, q, res, hints, bs.vstate)
    prev_score = typemax(Float32)
    
    while abs(prev_score - maximum(res)) > 0.0  # prepared to allow early stopping
        prev_score = maximum(res)
        push!(bs.beam, first(res))
        beamsearch_inner(index, q, res, bs.beam, bs.vstate)
    end

    res
end

function opt_expand_neighborhood(fun, gsearch::BeamSearch, n::Integer, iter::Integer, probes::Integer)
    logn = ceil(Int, log(2, n+1))
    probes = probes == 0 ? logn : probes
    f(x) = max(2, x + rand(-logn:logn))
    for i in 1:probes
        copy(gsearch, bsize=f(gsearch.bsize), ssize=f(gsearch.ssize)) |> fun
    end
end
