# This file is a part of SimilaritySearch.jl

using Random
export BeamSearch

"""
    BeamSearch(bsize::Integer=16, Δ::Float32)

BeamSearch is an iteratively improving local search algorithm that explores the graph using blocks of `bsize` elements and neighborhoods at the time.

- `bsize`: The size of the beam.
- `Δ`: Soft margin for accepting elements into the beam
"""
@with_kw mutable struct BeamSearch <: LocalSearchAlgorithm
    bsize::Int32 = 8  # size of the search beam
    Δ::Float32 = 1.0  # soft-margin for accepting an element into the beam
end

Base.copy(bsearch::BeamSearch; bsize=bsearch.bsize, Δ=bsearch.Δ) =
    BeamSearch(; bsize, Δ)

const GlobalBeamKnnResult = [KnnResult(32)]  # see __init__ function

@inline function getbeam(bs::BeamSearch)
    @inbounds beam = GlobalBeamKnnResult[Threads.threadid()]
    empty!(beam, bs.bsize)
    beam
end

function __init__beamsearch()
    for i in 2:Threads.nthreads()
        push!(GlobalBeamKnnResult, KnnResult(32))
    end
end

### local search algorithm

function beamsearch_queue(index::SearchGraph, q, res::KnnResult, objID, vstate)
    @inbounds if !visited(vstate, objID)
        visit!(vstate, objID)
        d = evaluate(index.dist, q, index[objID])
        push!(res, objID, d)
    end
end

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)
    for objID in hints
        beamsearch_queue(index, q, res, objID, vstate)
    end
    
    if length(res) == 0
        _range = 1:length(index)
        for i in 1:bs.bsize
           objID = rand(_range)
           beamsearch_queue(index, q, res, objID, vstate)
       end
    end
end

function beamsearch_inner(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, beam::KnnResult, vstate)
    Δ = bs.Δ
    while length(beam) > 0
        prev_id, prev_dist = popfirst!(beam)
        # prev_dist > maximum(res) && break
        @inbounds for childID in keys(index.links[prev_id])
            if !visited(vstate, childID)
                visit!(vstate, childID)
                d = evaluate(index.dist, q, index[childID])
                push!(res, childID, d)
                if d <= Δ * maximum(res)
                    push!(beam, childID, d)
                    #satpush!(childID, d, beam, index)
                end
                # d <= 0.9 * farthest(res).dist && push!(beam, childID, d)
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
    beam = getbeam(bs)
    push!(beam, first(res)) 
    beamsearch_inner(bs, index, q, res, beam, vstate)
    res
end
