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
    maxvisits::Int64 = 1000_000 # maximum visits by search, useful for early stopping without convergence, very high by default
end

Base.copy(bsearch::BeamSearch; bsize=bsearch.bsize, Δ=bsearch.Δ, maxvisits=bsearch.maxvisits) =
    BeamSearch(; bsize, Δ, maxvisits)

const GlobalBeamKnnResult = [KnnResultShifted(32)]  # see __init__ function

@inline function getbeam(bsize::Integer)
    @inbounds beam = GlobalBeamKnnResult[Threads.threadid()]
    empty!(beam, bsize)
    beam
end

function __init__beamsearch()
    for i in 2:Threads.nthreads()
        push!(GlobalBeamKnnResult, KnnResultShifted(32))
    end
end

### local search algorithm

function beamsearch_queue(index::SearchGraph, q, res, k, objID, vstate, visited_)
    @inbounds if !visited(vstate, objID)
        visit!(vstate, objID)
        visited_ += 1
        d = evaluate(index.dist, q, index[objID])
        k = push!(res, k, objID, d)
    end

    k, visited_
end

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res, hints, vstate, bsize)
    visited_ = 0
    k = 0

    for objID in hints
        k, visited = beamsearch_queue(index, q, res, k, objID, vstate, visited_)
    end
    
    if k == 0
        _range = 1:length(index)
        for i in 1:bsize
           objID = rand(_range)
           k, visited_ = beamsearch_queue(index, q, res, k, objID, vstate, visited_)
       end
    end

    k, visited_
end

function beamsearch_inner(bs::BeamSearch, index::SearchGraph, q, res, k, beam, vstate, Δ, maxvisits, visited_)
    while length(beam) > 0
        prev_id, prev_dist = popfirst!(beam)
        @inbounds for childID in index.links[prev_id]
            if !visited(vstate, childID)
                visit!(vstate, childID)
                d = evaluate(index.dist, q, index[childID])
                k = push!(res, k, childID, d)
                visited_ += 1
                visited_ > maxvisits && return visited_
                if d <= Δ * getdist(res, k)
                    #if length(index.links[childID]) > 1
                    push!(beam, childID, d)
                    # length(beam) == maxlength(beam) && continue
                    # sat_should_push(keys(beam), index, q, childID, d) && push!(beam, childID, d)
                    #end
                end
            end
        end
    end

    k, visited_
end

"""
    search(bs::BeamSearch, index::SearchGraph, q, res, hints, vstate)

Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bs`: the parameters of `BeamSearch`
- `index`: the local search index
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
- `hints`: Starting points for searching, randomly selected when it is an empty collection
- `vstate`: A dictionary like object to store the visiting state of vertices

Optional arguments (defaults to values in `bs`)
- `bsize`: Beam size
- `Δ`: exploration expansion factor
- `maxvisits`: Maximum number of nodes to visit (distance evaluations)

"""
function search(bs::BeamSearch, index::SearchGraph, q, res, hints, vstate; bsize=bs.bsize, Δ=bs.Δ, maxvisits=bs.maxvisits)
    # k is the number of neighbors in res
    k, visited_ = beamsearch_init(bs, index, q, res, hints, vstate, bsize)
    beam = getbeam(bsize)
    push!(beam, res[1])
    beamsearch_inner(bs, index, q, res, k, beam, vstate, Δ, maxvisits, visited_)
end
