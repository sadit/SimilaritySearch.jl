# This file is a part of SimilaritySearch.jl

using Random

# Below this many items `search` scans the whole database instead of running the beam: the
# answer is exact for at most 63 evaluations, where the beam spends ~50 for a recall of 0.94-0.98
# (ccnews subsets). From 128 items on, 1, 8, 16, 32 and 64 build the same graph within noise, so
# the value matters only to small indexes. See issue #125 and experiments/small-index-125.
const SMALLINDEX = 64


"""
    enqueue_item!(index::SearchGraph, ctx, q, obj, res, objID, vstate)

Internal function that evaluates the distance between a database object `obj` with id `objID` and the query `q`.
It helps to evaluate, mark as visited, and enqueue in the result set.
"""
@inline function enqueue_item!(index::SearchGraph, ctx, q, obj, res, objID, vstate)
    check_visited_and_visit!(vstate, convert(UInt64, objID)) && return res
    # a hint that is a near-duplicate member is not a node: nothing links to it, it links only
    # to its representative, and the search answers with representatives (see `Members`)
    isempty(index.members) || !ismember(index, objID) || return res
    d = evaluate(distance(index), q, obj)
    _pushres!(vstate, res, objID, d)
    add_distance_evaluations!(ctx, 1)
    res
end

"""
    beamsearch_init(bs::BeamSearch, index::SearchGraph, ctx, q, res::AbstractMetricQueue, hints, vstate)

Internal helper that seeds the search with the given `hints` (or, if that yields nothing,
with a small logarithmic sample of the database), enqueuing the corresponding items into `res`.
"""
function beamsearch_init(::BeamSearch, index::SearchGraph, ctx, q, res::AbstractMetricQueue, hints, vstate)
    res = approx_by_hints!(index, ctx, q, hints, res, vstate)
    if length(res) == 0
        n = length(index)
        for objID in 1:ceil(Int, log(2, 1 + n))
            enqueue_item!(index, ctx, q, database(index, objID), res, objID, vstate)
        end
    end
end

"""
    beamsearch_inner_beam(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractMetricQueue, vstate) -> (costdists, costblocks)

Internal helper that expands the beam (starting from the nearest item already found in `res`)
by visiting the neighborhoods of its elements, pushing newly evaluated items into `res` and back
into the beam when appropriate. Returns the number of distance evaluations (`costdists`) and
visited blocks/neighborhoods (`costblocks`) performed.
"""
function beamsearch_inner_beam(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractMetricQueue, vstate)
    Δ, maxvisits = bs.Δ, bs.maxvisits
    beam = getbeam(bs.bsize, ctx)
    push_item!(beam, nearest(res))
    dist = distance(index)
    costdists, costblocks = 0, 0

    @inbounds while length(beam) > 0
        prev = pop_min!(beam)
        N = neighbors(index.adj, prev.id)
        N === nothing && continue
        #prev.dist <= Δ * maximum(res) || continue
        costblocks += 1

        # Two passes over the neighbourhood. The first only reads the visited set and asks the
        # hardware to fetch the storage of every child that will be evaluated, so the cache
        # misses of a whole neighbourhood are in flight at once; the second evaluates them.
        # Measured outside the graph (bench-prefetch.jl, 384-d 8-bit codes, 1M random items): a
        # dependent miss per item costs 130 ns on one thread and 650 ns·thread on 64, batches of
        # 8-16 prefetched items 92 and 475.
        db = database(index)
        if prefetchable(db)
            for childID in N
                visited(vstate, convert(UInt64, childID)) || prefetch_item(db, childID)
            end
        end

        for childID in N
            check_visited_and_visit!(vstate, convert(UInt64, childID)) && continue
            d = evaluate(dist, q, database(index, childID))
            costdists += 1
            # a set that forgets can hand back a vertex already in `res`: it is not pushed again,
            # and not expanded again either (that is how a forgetting search would loop)
            _pushres!(vstate, res, childID, d) === :duplicate && continue
            costdists > maxvisits && @goto finish_search
            # covradius is the correct value but it uses a practical innecessary comparison (here we visited all hints)
            # The distance test goes first: most children fail it, and then their adjacency, two
            # dependent loads, is never read. This test had 14% of the compute threads' samples
            # in a ccnews construction profile; the swap gave 5-12% more QPS at recall 0.9 (#139).
            if d <= Δ * maximum(res) && neighbors_length(index.adj, childID) > 1
                push_item!(beam, childID, d)
            end
        end
    end

    @label finish_search
    costdists, costblocks
end

"""
    search(bs::BeamSearch, index::SearchGraph, ctx, q, res, hints, vstate)

Tries to reach the set of nearest neighbors specified in `res` for `q`.
- `bs`: the parameters of `BeamSearch`
- `index`: the local search index
- `ctx`: A SearchGraphContext object with preallocated objects
- `q`: the query
- `res`: The result object, it stores the results and also specifies the kind of query
- `hints`: Starting points for searching, randomly selected when it is an empty collection
- `vstate`: data structure to mark visited vertices. It is both an input and an output: what the
  caller marked is masked (`rebuild`'s `selfid`, the optimizer's clusters), and what the search
  evaluated stays marked (`allknn` skips the hints already covered). Below `SMALLINDEX` items the
  search scans everything through `enqueue_item!`, so the scan keeps the same contract, members
  included (issue #125).

"""
function search(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractMetricQueue, hints, vstate)
    n = length(index)
    if n == 0
        res
    elseif n < SMALLINDEX
        for i in 1:n
            enqueue_item!(index, ctx, q, database(index, i), res, i, vstate)
        end
    else
        beamsearch_init(bs, index, ctx, q, res, hints, vstate)
        costdists, costblocks = beamsearch_inner_beam(bs, index, ctx, q, res, vstate)
        add_distance_evaluations!(ctx, costdists)
        add_block_evaluations!(ctx, costblocks)
    end

    res
end
