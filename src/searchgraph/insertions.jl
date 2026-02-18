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
    index
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
    @assert length(database(index)) > 0

    if ctx.parallel_block == 1 || Threads.nthreads() == 1
        _sequential_append_items_loop!(index, ctx, length(index) + 1, length(database(index)))
    else
        _parallel_append_items_loop!(index, ctx, length(index) + 1, length(database(index)))
    end

    index
end

function _sequential_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n)
    @inbounds while sp <= n
        push_item!(index, ctx, database(index, sp), false)
        sp += 1
    end
end

function _parallel_append_items_loop!(index::SearchGraph, ctx::SearchGraphContext, sp, n)
    adj = index.adj
    resize!(adj, n)
    
    while sp <= n
        ep = min(n, sp + ctx.parallel_block)
        minbatch = getminbatch(ctx, ep - sp + 1)
        # searching neighbors
        #@batch per=thread minbatch=minbatch for i in sp:ep
        Threads.@threads :static for objID in sp:ep
            neighborhood = find_neighborhood(index, ctx, database(index, objID), sp:objID-1)
            #neighborhood = find_neighborhood(index, ctx, database(index, objID), sp:ep)
            if length(neighborhood) == 0
                adj.end_point[objID] = UInt32[]
            else
                adj.end_point[objID] = collect(UInt32, IdView(neighborhood))
            end
        end

        LOG(ctx.logger, :add_vertex!, index, ctx, sp, ep)
        # connecting neighbors
        connect_reverse_links(ctx.neighborhood, index.adj, sp, ep)
        index.len[] = ep

        # apply callbacks
        execute_callbacks(index, ctx, sp, ep)
        sp = ep + 1
    end
end

"""
    push_item!(
        index::SearchGraph,
        ctx,
        item;
        push_item=true
    )

Appends an object into the index.

Arguments:

- `index`: The search graph index where the insertion is going to happen
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `push_db`: if `push_db=false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed)

"""
@inline function push_item!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    item;
    push_db::Bool=true,
)

    push_item!(index, ctx, item, push_db)
end

"""
    push_item!(
        index::SearchGraph,
        ctx,
        item,
        push_item
    )

Appends an object into the index. Internal function

Arguments:

- `index`: The search graph index where the insertion is going to happen.
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `ctx`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `push_db`: if `false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed).

- Note: setting `callbacks` as `nothing` ignores the execution of any callback
"""
@inline function push_item!(
    index::SearchGraph,
    ctx::SearchGraphContext,
    item,
    push_db::Bool
)
    push_db && push_item!(index.db, item)
    neighbors = find_neighborhood(index, ctx, item) |> IdView |> collect
    add_vertex!(index.adj, neighbors)
    n = index.len[] = length(index.adj)
    LOG(ctx.logger, :add_vertex!, index, ctx, n, n)
    if n > 1
        connect_reverse_links(ctx.neighborhood, index.adj, n, neighbors)
        execute_callbacks(index, ctx)
    end

    index
end
