# This file is a part of SimilaritySearch.jl

"""
    append_items!(
        index::SearchGraph,
        ctx::SearchGraphContext,
        db
    )

Appends all items in db to the index. It can be made in parallel or sequentially.

# Arguments:

- `index`: the search graph index
- `db`: the collection of objects to insert, an `AbstractDatabase` is the canonical input, but supports any iterable objects
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).

"""
function append_items!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    items::AbstractDatabase;
)
    append_items!(index.db, items)
    index!(index, ctx)
end

"""
    index!(index::SearchGraph, ctx::SearchGraphContext)

Indexes the already initialized database (e.g., given in the constructor method). It can be made in parallel or sequentially.
The arguments are the same than `append_items!` function but using the internal `index.db` as input.

# Arguments:

- `index`: The graph index
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).

"""
function index!(index::SearchGraph, ctx::SearchGraphContext)
    n = length(database(index))
    @assert n > 0

    if ctx.parallel_block == 1 || Threads.nthreads() == 1
        qcache = let s = neighborhoodsize(ctx.neighborhood, n), t = 2
            isodd(s) && (s+=1)
            zeros(IdDist, s, t)
        end
        _sequential_append_items_loop!(index, ctx, length(index) + 1, n, qcache)
    else
        qcache = let s = neighborhoodsize(ctx.neighborhood, n), t = 2 * Threads.maxthreadid()
            isodd(s) && (s+=1)
            zeros(IdDist, s, t)
        end
        _parallel_append_items_loop!(index, ctx, length(index) + 1, n, qcache)
    end

    index
end

function _sequential_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n, qcache)
    @inbounds while sp <= n
        ksearch = neighborhoodsize(ctx.neighborhood, sp)
        tmp = knnqueue(ctx, view(qcache, 1:ksearch, 1))
        neighbors = knnqueue(ctx, view(qcache, 1:ksearch, 2))

        push_item!(index, ctx, database(index, sp), tmp, neighbors, false)
        sp += 1
    end
end

function _parallel_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n, qcache)
    resize!(index.adj, n)
    
    while sp <= n
        ep = min(n, sp + ctx.parallel_block)
        minbatch = getminbatch(ctx, ep - sp + 1)
        # searching neighbors
        #@batch per=thread minbatch=minbatch for i in sp:ep
        Threads.@threads :static for objID in sp:ep
            item = database(index, objID)
            R = sp:objID-1
            ksearch = neighborhoodsize(ctx.neighborhood, ep)
            ti = 2 * Threads.threadid()
            tmp = knnqueue(ctx, view(qcache, 1:ksearch, ti-1))
            neighbors_ = knnqueue(ctx, view(qcache, 1:ksearch, ti))
            find_neighborhood!(neighbors_, index, ctx, item, tmp, R)
            add!(index.adj, objID, IdView(neighbors_))
        end

        LOG(ctx.logger, :add!, index, ctx, sp, ep)
        # connecting neighbors
        connect_reverse_links!(index.adj, sp, ep)
        index.len[] = ep

        # apply callbacks
        execute_callbacks!(index, ctx, sp, ep)
        sp = ep + 1
    end
end

"""
    push_item!(
        index::SearchGraph,
        ctx,
        item,
        qcache
        push_item
    )

Appends an object into the index. Internal function

Arguments:

- `index`: The search graph index where the insertion is going to happen.
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `tmp`: knnqueue to be used by the neighborhood computation
- `neighbors`: knnqueue to be used by the neighborhood computation
- `push_db`: if `false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed).

- Note: setting `callbacks` as `nothing` ignores the execution of any callback
"""
@inline function push_item!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    item,
    neighbors_,
    tmp,
    push_db::Bool
)
    push_db && push_item!(index.db, item)
    find_neighborhood!(neighbors_, index, ctx, item, tmp, 1:-1)
    n = Int32(index.len[] + 1)
    add!(index.adj, n, IdView(neighbors_))
    LOG(ctx.logger, :add!, index, ctx, n, n)
    if n > 1
        connect_reverse_links!(index.adj, n, neighbors(index.adj, n))
        execute_callbacks!(index, ctx)
    end
    index.len[] = n
    index
end
