# This file is a part of SimilaritySearch.jl

using Random

function beamsearch_init(::BeamSearch, index::SearchGraph, q, res::AbstractKnn, hints, vstate)
    res = approx_by_hints!(index, q, hints, res, vstate)
    if length(res) == 0
        n = length(index)
        for objID in 1:ceil(Int, log(2, 1 + n))
            enqueue_item!(index, q, database(index, objID), res, objID, vstate)
        end
    end
end

function beamsearch_inner_beam(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractKnn, vstate)
    Δ, maxvisits = bs.Δ, bs.maxvisits
    beam = getbeam(bs.bsize, ctx)
    push_item!(beam, nearest(res))
    dist = distance(index)
    costdists, costblocks = 0, 0

    @inbounds while 0 < length(beam)
        costblocks += 1
        prev = pop_min!(beam)
        for childID in neighbors(index.adj, prev.id)
            check_visited_and_visit!(vstate, convert(UInt64, childID)) && continue
            d = evaluate(dist, q, database(index, childID))
            c = IdWeight(childID, d)
            push_item!(res, c)
            costdists += 1
            costdists > maxvisits && @goto finish_search
            # covradius is the correct value but it uses a practical innecessary comparison (here we visited all hints)
            if neighbors_length(index.adj, childID) > 1 && d <= Δ * maximum(res)
                #=if length(beam) > 0 && minimum(beam) ≈ c.weight
                    pop_min!(beam)  # REMOVE THIS AFTER TEST
                end=#
                push_item!(beam, c)
            end
        end
    end

    @label finish_search
    costdists, costblocks
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
    vstate::Vector{UInt64}=getvstate(length(index), ctx)
)
    # k is the number of neighbors in res
    # vstate = vstate
    n = length(index)
    if n == 0
        res
    elseif n < 64
        dist = distance(index)
        for i in 1:n
            d = evaluate(dist, q, database(index, i))
            push_item!(res, i, d)
        end

        add_distance_evaluations!(res, n)
    else
        beamsearch_init(bs, index, q, res, hints, vstate)
        costdists, costblocks = beamsearch_inner_beam(bs, index, ctx, q, res, vstate)
        add_distance_evaluations!(res, costdists)
        add_block_evaluations!(res, costblocks)
    end

    res
end
