# This file is a part of SimilaritySearch.jl


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
            isodd(s) && (s += 1)
            zeros(IdDist, s, t)
        end
        _sequential_append_items_loop!(index, ctx, length(index) + 1, n, qcache)
    else
        qcache = let s = neighborhoodsize(ctx.neighborhood, n), t = 2 * Threads.maxthreadid()
            isodd(s) && (s += 1)
            zeros(IdDist, s, t)
        end
        _parallel_append_items_loop!(index, ctx, length(index) + 1, n, qcache)
    end

    index
end

function index!(idx::SearchGraph, ctx::SearchGraphContext, kind::Symbol; kwargs...)
    index!(idx, ctx, Val(kind); kwargs...)
end

include("staticindexing-prefixes.jl")
