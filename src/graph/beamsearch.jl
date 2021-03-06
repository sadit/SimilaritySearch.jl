# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Random
export BeamSearch

"""
    BeamSearch(bsize::Integer=16, hints=Int32[], beam=KnnResult(bsize), vstate=VisitedVertices())

BeamSearch is an iteratively improving local search algorithm that explores the graph using blocks of `bsize` elements and neighborhoods at the time.
Multithreading applications must have copies of this object due to shared cache objects.

- `hints`: An initial hint for the exploration (empty hints imply `bsize` random starting points).
- `bsize`: The size of the beam.
- `beam`: A cache object for reducing memory allocations
- `vstate`: A cache object for reducing memory allocations
"""
@with_kw mutable struct BeamSearch{BeamType<:KnnResult} <: LocalSearchAlgorithm
    hints::Vector{Int32} = Int32[]
    bsize::Int32 = 8  # size of the search beam
    beam::BeamType = KnnResult(8)
    vstate::VisitedVertices = VisitedVertices()
end


Base.copy(bsearch::BeamSearch;
        hints=bsearch.hints,
        bsize=bsearch.bsize,
        beam=KnnResult(Int(bsize)),
        vstate=VisitedVertices()
    ) = BeamSearch(; hints, bsize, beam, vstate)


function Base.copy!(dst::BeamSearch, src::BeamSearch)
    dst.hints = src.hints
    dst.bsize = src.bsize
    dst.beam = src.beam
    dst.vstate = src.vstate
end

Base.string(s::BeamSearch) = """{BeamSearch: bsize=$(s.bsize), hints=$(length(s.hints))"""

# const BeamType = typeof((objID=Int32(0), dist=0.0))
### local search algorithm

function beamsearch_queue(index::SearchGraph, q, res::KnnResult, objID, vstate)
    @inbounds if getstate(vstate, objID) === UNKNOWN
        setstate!(vstate, objID, VISITED)
        d = evaluate(index.dist, q, index[objID])
        push!(res, objID, d)
    end
end

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)
    empty!(vstate)

    if length(hints) == 0
        _range = 1:length(index)
         for i in 1:bs.bsize
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
        @inbounds for childID in keys(index.links[prev_id])
            if getstate(vstate, childID) === UNKNOWN
                setstate!(vstate, childID, VISITED)
                d = evaluate(index.dist, q, index[childID])
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
    if length(index) > 0
        empty!(bs.beam, bs.bsize)
        beamsearch_init(bs, index, q, res, hints, bs.vstate)
        push!(bs.beam, first(res))
        beamsearch_inner(index, q, res, bs.beam, bs.vstate)
    end

    res
end
