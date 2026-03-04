# This file is a part of SimilaritySearch.jl

export rebuild

"""
    rebuild(g::SearchGraph; context=SearchGraphContext())

Rebuilds the `SearchGraph` index but seeing the whole dataset for the incremental construction, i.e.,
it can connect the i-th vertex to its knn in the 1..n possible vertices instead of its knn among 1..(i-1) as in the original algorithm.

# Arguments

- `g`: The search index to be rebuild.
- `context`: The context to run the procedure, it can differ from the original one.

"""
function rebuild(g::SearchGraph, ctx::SearchGraphContext;
    progress=Progress(length(g); desc="rebuild", dt=2.0)
)
    n = length(g)
    ksearch = neighborhoodsize(ctx.neighborhood, n)
    @assert n > 0
    direct = Vector{Vector{UInt32}}(undef, n)  # this separated links version needs has easier multithreading/locking needs
    minbatch = getminbatch(ctx, n)
    qcache = zeros(IdDist, neighborhoodsize(ctx.neighborhood, n), 2 * Threads.maxthreadid())

    Threads.@threads :static for j in 1:minbatch:n
        n_ = min(n, j + minbatch - 1)
        @inbounds for objID in j:n_
            tid = 2Threads.threadid()
            tmp = knnqueue(ctx, view(qcache, 1:ksearch, tid-1))
            N = knnqueue(ctx, view(qcache, 1:ksearch, tid))
            find_neighborhood!(N, g, ctx, database(g, objID), tmp, 1:-1; hints=first(neighbors(g.adj, objID)))
            direct[objID] = collect(IdView(N))
            # @info length(direct[objID]) neighbors_length(g.adj, objID) 
        end

        progress !== nothing && next!(progress)
    end

    adj = AdjList(direct)
    Threads.@threads :static for nodeID in eachindex(direct)
        connect_reverse_links!(adj, nodeID, neighbors(adj, nodeID)) do relID
            relID != nodeID
        end
    end

    G = SearchGraph(distance(g), database(g), adj, copy(g.hints), Ref(g.algo[]), Ref(length(g)))

    execute_callbacks!(G, ctx, force=true)
   
    G
end
