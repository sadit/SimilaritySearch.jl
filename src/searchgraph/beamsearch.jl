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
    bsize::Int32 = 4  # size of the search beam
    Δ::Float32 = 1.0  # soft-margin for accepting an element into the beam
    maxvisits::Int64 = 1000_000 # maximum visits by search, useful for early stopping without convergence, very high by default
end

Base.copy(bsearch::BeamSearch; bsize=bsearch.bsize, Δ=bsearch.Δ, maxvisits=bsearch.maxvisits) =
    BeamSearch(; bsize, Δ, maxvisits)

### local search algorithm

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, hints, vstate, bsize, beam)
    visited_ = approx_by_hints(index, q, hints, res, vstate)
    
    if length(res) == 0
        _range = 1:length(index)
        for _ in 1:bsize
           objID = rand(_range)
           visited_ += enqueue_item!(index, q, database(index, objID), res, objID, vstate)
        end
    end

    visited_
end

function beamsearch_inner(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, vstate, beam, Δ::Float32, maxvisits::Int64, visited_::Int64)
    push_item!(beam, res[1])
    #sp = 1
    dist = distance(index)
    hops = 0
    #@inbounds while sp <= length(beam)
    @inbounds while 0 < length(beam)
        hops += 1
        #prev_id = beam[sp].id
        #prev_id = argmin(beam)
        prev_id = popfirst!(beam).id
        #sp += 1
        for childID in neighbors(index.adj, prev_id)
            check_visited_and_visit!(vstate, convert(UInt64, childID)) && continue
            d = evaluate(dist, q, database(index, childID))
            c = IdWeight(childID, d)
            push_item!(res, c)
            visited_ += 1
            visited_ > maxvisits && @goto finish_search 
            # covradius is the correct value but it uses a practical innecessary comparison (here we visited all hints)
            if neighbors_length(index.adj, childID) > 1 && d <= Δ * maximum(res)
            # if neighbors_length(index.adj, childID) > 1 && d <= Δ * covradius(res)
                #push_item!(beam, c, sp)
                push_item!(beam, c)
            end
        end
        
        # Δ *= 0.98f0
    end

    @label finish_search
    SearchResult(res, visited_, hops)
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
function search(bs::BeamSearch, index::SearchGraph, context::SearchGraphContext, q, res, hints; bsize::Int32=bs.bsize, Δ::Float32=bs.Δ, maxvisits::Int=bs.maxvisits, vstate::Vector{UInt64}=getvstate(length(index), context))
    # k is the number of neighbors in res
    vstate = PtrArray(vstate)
    beam = getbeam(bsize, context)
    visited_ = beamsearch_init(bs, index, q, res, hints, vstate, bsize, beam)
    beamsearch_inner(bs, index, q, res, vstate, beam, Δ, maxvisits, visited_)
end
