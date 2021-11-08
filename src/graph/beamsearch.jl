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
    maxvisits::Int = typemax(Int) # maximum visits by search, useful for early stopping without convergence
end

Base.copy(bsearch::BeamSearch; bsize=bsearch.bsize, Δ=bsearch.Δ, maxvisits=bsearch.maxvisits) =
    BeamSearch(; bsize, Δ, maxvisits)

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
    visited_ = 0
    @inbounds if !visited(vstate, objID)
        visit!(vstate, objID)
        visited_ += 1
        d = evaluate(index.dist, q, index[objID])
        push!(res, objID, d)
    end

    visited_
end

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate)
    visited_ = 0

    for objID in hints
        visited_ += beamsearch_queue(index, q, res, objID, vstate)
    end
    
    if length(res) == 0
        _range = 1:length(index)
        for i in 1:bs.bsize
           objID = rand(_range)
           visited_ += beamsearch_queue(index, q, res, objID, vstate)
       end
    end

    visited_
end

function beamsearch_inner(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, beam::KnnResult, vstate, visited_)
    Δ = bs.Δ
    maxvisits = bs.maxvisits
 
    while length(beam) > 0
        prev_id, prev_dist = popfirst!(beam)
        @inbounds for childID in keys(index.links[prev_id])
            if !visited(vstate, childID)
                visit!(vstate, childID)
                d = evaluate(index.dist, q, index[childID])
                push!(res, childID, d)
                visited_ += 1
                visited_ > maxvisits && return visited_
                if d <= Δ * maximum(res)
                    push!(beam, childID, d)
                    #satpush!(childID, d, beam, index)
                end
            end
        end
    end

    visited_
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
    visited_ = beamsearch_init(bs, index, q, res, hints, vstate)
    beam = getbeam(bs)
    push!(beam, first(res))
    visited_ = beamsearch_inner(bs, index, q, res, beam, vstate, visited_)
    res, visited_
end
