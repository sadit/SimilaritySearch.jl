# This file is a part of SimilaritySearch.jl

using Random

"""
    BeamSearch(bsize::Integer=16, Δ::Float32)

BeamSearch is an iteratively improving local search algorithm that explores the graph using blocks of `bsize` elements and neighborhoods at the time.

- `bsize`: The size of the beam.
- `Δ`: Soft margin for accepting elements into the beam
- `maxvisits`: MAximum visits while searching, useful for early stopping without convergence
"""
@with_kw mutable struct BeamSearch <: LocalSearchAlgorithm
    bsize::Int32 = 8  # size of the search beam
    Δ::Float32 = 1.0  # soft-margin for accepting an element into the beam
    maxvisits::Int64 = 1000_000 # maximum visits by search, useful for early stopping without convergence, very high by default
end

Base.copy(bsearch::BeamSearch; bsize=bsearch.bsize, Δ=bsearch.Δ, maxvisits=bsearch.maxvisits) =
    BeamSearch(; bsize, Δ, maxvisits)

### local search algorithm

@inline function beamsearch_queue(index::SearchGraph, q, res::KnnResult, objID, vstate)
    check_visited_and_visit!(vstate, convert(UInt64, objID)) && return 0
    @inbounds push_item!(res, objID, evaluate(distance(index), q, database(index, objID)))
    1
end

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate, bsize)
    visited_ = 0

    for objID in hints
        visited_ += beamsearch_queue(index, q, res, objID, vstate)
    end
    
    if length(res) == 0
        _range = 1:length(index)
        for _ in 1:bsize
           objID = rand(_range)
           visited_ += beamsearch_queue(index, q, res, objID, vstate)
       end
    end

    visited_
end

function beamsearch_inner(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, vstate, beam, Δ::Float32, maxvisits::Int64, visited_::Int64)
    push_item!(beam, res[1])
    sp = 1

    @inbounds while sp <= length(beam)
        prev_id = beam[sp].id
        #prev_dist > maximum(res) && break
        sp += 1
        for childID in neighbors(index.adj, prev_id)
            check_visited_and_visit!(vstate, convert(UInt64, childID)) && continue
            d = evaluate(distance(index), q, database(index, childID))
            push_item!(res, childID, d)
            visited_ += 1
            visited_ > maxvisits && @goto finish_search
            if d <= Δ * covradius(res) # maximum(res)
                push_item!(beam, IdWeight(childID, d), sp)
                # rand() < 1 / (sp-1) && continue
                # sat_should_push(keys(beam), index, q, childID, d) && push!(beam, childID, d)
            end
        end
    end

    @label finish_search
    SearchResult(res, visited_)
end

"""
    search(bs::BeamSearch, index::SearchGraph, context, q, res, hints; bsize=bs.bsize, Δ=bs.Δ, maxvisits=bs.maxvisits)

Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bs`: the parameters of `BeamSearch`
- `index`: the local search index
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
- `hints`: Starting points for searching, randomly selected when it is an empty collection
- `context`: A SearchGraphContext object with preallocated objects

Optional arguments (defaults to values in `bs`)
- `bsize`: Beam size
- `Δ`: exploration expansion factor
- `maxvisits`: Maximum number of nodes to visit (distance evaluations)

"""
function search(bs::BeamSearch, index::SearchGraph, context::SearchGraphContext, q, res, hints; bsize=bs.bsize, Δ=bs.Δ, maxvisits=bs.maxvisits, vstate=getvstate(length(index), context))
    # k is the number of neighbors in res
    visited_ = beamsearch_init(bs, index, q, res, hints, vstate, bsize)
    beam = getbeam(bsize, context)
    beamsearch_inner(bs, index, q, res, vstate, beam, Δ, maxvisits, visited_)
end
