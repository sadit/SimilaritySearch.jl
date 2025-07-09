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

function beamsearch_init(bs::BeamSearch, index::SearchGraph, q, res::AbstractKnn, hints, vstate)
    cost = approx_by_hints(index, q, hints, res, vstate)
    
    if length(res) == 0
        n = length(index)
        for objID in 1:ceil(Int, log(2, 1+n))
           #objID = rand(1:n)
           cost += enqueue_item!(index, q, database(index, objID), res, objID, vstate)
        end
    end
    
    cost
end

function beamsearch_inner_beam(bs::BeamSearch, index::SearchGraph, q, res::AbstractKnn, vstate, beam, Δ::Float32, maxvisits::Int64, cost::Int64)
    push_item!(beam, nearest(res))
    dist = distance(index)
    hops = 0
    @inbounds while 0 < length(beam)
        hops += 1
        prev_id = pop_min!(beam).id
        for childID in neighbors(index.adj, prev_id)
            check_visited_and_visit!(vstate, convert(UInt64, childID)) && continue
            d = evaluate(dist, q, database(index, childID))
            c = IdWeight(childID, d)
            push_item!(res, c)
            cost += 1
            cost > maxvisits && @goto finish_search 
            # covradius is the correct value but it uses a practical innecessary comparison (here we visited all hints)
            if neighbors_length(index.adj, childID) > 1 && d <= Δ * maximum(res)
                push_item!(beam, c)
            end
        end      
    end

    @label finish_search
    res.eblocks = hops
    res.cost = cost
    res
end

#=function beamsearch_inner_large_beam(bs::BeamSearch, index::SearchGraph, q, res::KnnResult, vstate, beam, Δ::Float32, maxvisits::Int64, cost::Int64)
    push_item!(beam, res[1])
    sp = 1
    dist = distance(index)
    hops = 0
    @inbounds while sp <= length(beam)
        hops += 1
        prev_id = beam[sp].id
        sp += 1
        for childID in neighbors(index.adj, prev_id)
            check_visited_and_visit!(vstate, convert(UInt64, childID)) && continue
            d = evaluate(dist, q, database(index, childID))
            c = IdWeight(childID, d)
            push_item!(res, c)
            cost += 1
            cost > maxvisits && @goto finish_search 
            # covradius is the correct value but it uses a practical innecessary comparison (here we visited all hints)
            if neighbors_length(index.adj, childID) > 1 && d <= Δ * maximum(res)
                push_item!(beam, c, sp)
            end
        end
    end

    @label finish_search
    SearchResult(res, cost, hops)
end
=#

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
function search(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractKnn, hints; bsize::Int32=bs.bsize, Δ::Float32=bs.Δ, maxvisits::Int=bs.maxvisits, vstate::Vector{UInt64}=getvstate(length(index), ctx))
    # k is the number of neighbors in res
    # vstate = vstate
    beam = getbeam(bsize, ctx)
    cost = beamsearch_init(bs, index, q, res, hints, vstate)
    #if bsize <= 12  # this could change with the running computer
    beamsearch_inner_beam(bs, index, q, res, vstate, beam, Δ, maxvisits, cost)
    #else
    #    beamsearch_inner_large_beam(bs, index, q, res, vstate, beam, Δ, maxvisits, cost)
    #end
end
