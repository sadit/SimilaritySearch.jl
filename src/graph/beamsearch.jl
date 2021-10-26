# This file is a part of SimilaritySearch.jl

using Random
export BeamSearch

const GlobalBeamKnnResult = [KnnResult(30)]  # see __init__ function
@inline getbeam() = @inbounds GlobalBeamKnnResult[Threads.threadid()]

"""
    BeamSearch(bsize::Integer=16, beam=KnnResult(bsize))

BeamSearch is an iteratively improving local search algorithm that explores the graph using blocks of `bsize` elements and neighborhoods at the time.
Multithreading applications must have copies of this object due to shared cache objects.

- `bsize`: The size of the beam.
"""
@with_kw mutable struct BeamSearch <: LocalSearchAlgorithm
    bsize::Int32 = 8  # size of the search beam
end

Base.copy(bsearch::BeamSearch;
        bsize=bsearch.bsize
    ) = BeamSearch(; bsize)


function Base.copy!(dst::BeamSearch, src::BeamSearch)
    dst.bsize = src.bsize
end

Base.string(s::BeamSearch) = """{BeamSearch: bsize=$(s.bsize)}"""

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
    for objID in hints
        beamsearch_queue(index, q, res, objID, vstate)
    end
    
    if length(vstate) == 0
        _range = 1:length(index)
        for i in 1:bs.bsize
           objID = rand(_range)
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
    search(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)

Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bs`: the parameters of `BeamSearch`
- `index`: the local search index
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
- `hints`: Starting points for searching, randomly selected when it is an empty collection
- `vstate`: A dictionary like object to store the visiting state of vertices
"""
function search(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)
    beamsearch_init(bs, index, q, res, hints, vstate)
    beam = getbeam()
    empty!(beam, bs.bsize)
    push!(beam, first(res))
    beamsearch_inner(index, q, res, beam, vstate)
    res
end
