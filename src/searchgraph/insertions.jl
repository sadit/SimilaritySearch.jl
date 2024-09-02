# This file is a part of SimilaritySearch.jl

"""
    append_items!(
        index::SearchGraph,
        context::SearchGraphContext,
        db
    )

Appends all items in db to the index. It can be made in parallel or sequentially.

# Arguments:

- `index`: the search graph index
- `db`: the collection of objects to insert, an `AbstractDatabase` is the canonical input, but supports any iterable objects
- `context`: The context environment of the graph, see  [`SearchGraphContext`](@ref).

"""
function append_items!(
        index::SearchGraph,
        context::SearchGraphContext,
        db::AbstractDatabase;
    )
    db = convert(AbstractDatabase, db)
    append_items!(index.db, db)

    context.parallel_block == 1 && return _sequential_append_items_loop!(index, context)

    n = length(index) + length(db)
    m = 0

    parallel_first_block = min(context.parallel_first_block, n)

    @inbounds while length(index) < parallel_first_block
        m += 1
        push_item!(index, context, db[m], false)
    end

    sp = length(index) + 1
    sp > n && return index

    _parallel_append_items_loop!(index, context, sp, n)
    index
end

"""
    index!(index::SearchGraph, context::SearchGraphContext)

Indexes the already initialized database (e.g., given in the constructor method). It can be made in parallel or sequentially.
The arguments are the same than `append_items!` function but using the internal `index.db` as input.

# Arguments:

- `index`: The graph index
- `context`: The context environment of the graph, see  [`SearchGraphContext`](@ref).

"""
function index!(index::SearchGraph, context::SearchGraphContext)
    @assert length(index) == 0 && length(index.db) > 0
    context.parallel_block == 1 && return _sequential_append_items_loop!(index, context)

    m = 0
    db = database(index)
    n = length(db)

    parallel_first_block = min(context.parallel_first_block, n)
    @inbounds while length(index) < parallel_first_block
        m += 1
        push_item!(index, context, db[m], false)
    end

    sp = length(index) + 1
    sp > n && return index

    _parallel_append_items_loop!(index, context, sp, n)
    index
end

function _sequential_append_items_loop!(index::SearchGraph, context::SearchGraphContext)
    i = length(index)
    db = index.db
    n = length(db)
    @inbounds while i < n
        i += 1
        push_item!(index, context, db[i], false)
    end

    index
end

function _parallel_append_items_loop!(index::SearchGraph, context::SearchGraphContext, sp, n)
    adj = index.adj
    resize!(adj, n)

    while sp <= n
        ep = min(n, sp + context.parallel_block)
        # searching neighbors 
	      @batch minbatch=getminbatch(0, ep-sp+1) per=thread for i in sp:ep
            @inbounds adj.end_point[i] = find_neighborhood(index, context, database(index, i))
        end

        # connecting neighbors
        connect_reverse_links(context.neighborhood, index.adj, sp, ep)
        index.len[] = ep

        # apply callbacks
        execute_callbacks(index, context, sp, ep)
        context.logger !== nothing && LOG(context.logger, append_items!, index, sp, ep, n)
        sp = ep + 1
    end
end

"""
    push_item!(
        index::SearchGraph,
        context,
        item;
        push_item=true
    )

Appends an object into the index.

Arguments:

- `index`: The search graph index where the insertion is going to happen
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `context`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `push_db`: if `push_db=false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed)

"""
@inline function push_item!(
        index::SearchGraph,
        context::SearchGraphContext,
        item;
        push_db=true,
    )

    push_item!(index, context, item, push_db)
end

"""
    push_item!(
        index::SearchGraph,
        context,
        item,
        push_item
    )

Appends an object into the index. Internal function

Arguments:

- `index`: The search graph index where the insertion is going to happen.
- `item`: The object to be inserted, it should be in the same space than other objects in the index and understood by the distance metric.
- `context`: The context environment of the graph, see  [`SearchGraphContext`](@ref).
- `push_db`: if `false` is an internal option, used by `append!` and `index!` (it avoids to insert `item` into the database since it is already inserted but not indexed).

- Note: setting `callbacks` as `nothing` ignores the execution of any callback
"""
@inline function push_item!(
    index::SearchGraph,
    context::SearchGraphContext,
    item,
    push_db::Bool
)
    push_db && push_item!(index.db, item)
    neighbors = find_neighborhood(index, context, item)
    add_vertex!(index.adj, neighbors)
    n = index.len[] = length(index.adj)
    if n > 1 
        connect_reverse_links(context.neighborhood, index.adj, n, neighbors)
        execute_callbacks(index, context)
    end
    
    context.logger !== nothing && LOG(context.logger, push_item!, index, n)
    index
end
