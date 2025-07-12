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

function beamsearch_init(::BeamSearch, index::SearchGraph, q, res::AbstractKnn, hints, vstate)
    res = approx_by_hints!(index, q, hints, res, vstate)
    if length(res) == 0
        n = length(index)
        for objID in 1:ceil(Int, log(2, 1+n))
           res = enqueue_item!(index, q, database(index, objID), res, objID, vstate)
        end
    end
    
    res
end

function beamsearch_inner_beam(::BeamSearch, index::SearchGraph, q, res::AbstractKnn, vstate, beam, Δ::Float32, maxvisits::Int64)
    res, _ = push_item!(beam, nearest(res))
    dist = distance(index)
    eblocks = 0
    cost = res.cost
    @inbounds while 0 < length(beam)
        eblocks += 1
        beam, prev = pop_min!(beam)
        for childID in neighbors(index.adj, prev.id)
            check_visited_and_visit!(vstate, convert(UInt64, childID)) && continue
            d = evaluate(dist, q, database(index, childID))
            c = IdWeight(childID, d)
            res, _ = push_item!(res, c)
            cost += one(cost)
            cost > maxvisits && @goto finish_search 
            # covradius is the correct value but it uses a practical innecessary comparison (here we visited all hints)
            if neighbors_length(index.adj, childID) > 1 && d <= Δ * maximum(res)
                beam, _ = push_item!(beam, c)
            end
        end
    end

    @label finish_search
    @reset res.eblocks = convert(typeof(res.eblocks), eblocks)
    @reset res.cost = convert(typeof(res.cost), cost)
    res
end

"""
    search(bs::BeamSearch, index::SearchGraph, ctx, q, res, hints; bsize=bs.bsize, Δ=bs.Δ, maxvisits=bs.maxvisits)

Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bs`: the parameters of `BeamSearch`
- `index`: the local search index
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
- `hints`: Starting points for searching, randomly selected when it is an empty collection
- `ctx`: A SearchGraphContext object with preallocated objects

Optional arguments (defaults to values in `bs`)
- `bsize`: Beam size
- `Δ`: exploration expansion factor
- `maxvisits`: Maximum number of nodes to visit (distance evaluations)

"""
function search(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractKnn, hints;
        bsize::Int32=bs.bsize,
        Δ::Float32=bs.Δ,
        maxvisits::Int=bs.maxvisits,
        vstate::Vector{UInt64}=getvstate(length(index), ctx)
        )
    # k is the number of neighbors in res
    # vstate = vstate
    beam = getbeam(bsize, ctx)
    res = beamsearch_init(bs, index, q, res, hints, vstate)
    beamsearch_inner_beam(bs, index, q, res, vstate, beam, Δ, maxvisits)
end
