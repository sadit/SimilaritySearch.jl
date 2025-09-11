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
function rebuild(g::SearchGraph, ctx::SearchGraphContext)
    n = length(g)
    @assert n > 0
    direct = Vector{Vector{UInt32}}(undef, n)  # this separated links version needs has easier multithreading/locking needs
    reverse = Vector{Vector{UInt32}}(undef, n)
    minbatch = ctx.minbatch < 0 ? n : getminbatch(ctx, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        neighborhood = find_neighborhood(g, ctx, database(g, i); hints=first(neighbors(g.adj, i))) 
        @inbounds direct[i] = collect(IdView(neighborhood))
        # @info length(direct[i]) neighbors_length(g.adj, i) 
        reverse[i] = UInt32[]
    end

    rebuild_connect_reverse_links!(ctx.neighborhood, direct, reverse, g.adj.locks, 1, length(g), minbatch)
    G = copy(g; adj=AdjacencyList(direct), hints=copy(g.hints), algo=copy(g.algo))
    execute_callbacks(G, ctx, force=true)
    G
end

function rebuild_connect_reverse_links!(N, direct, reverse, locks, sp, ep, minbatch)
    @batch minbatch=minbatch per=thread for i in sp:ep
        j = 0
        D = direct[i]
        # p = 1f0
        @inbounds while j < length(D)
            j += 1
            id = D[j]
            if i == id
                D[j] = D[end]
                pop!(D)
                j -= 1
                continue
            end

            lock(locks[id])
            try
                push!(reverse[id], i)
            finally
                unlock(locks[id])
            end
        end
    end

    @batch minbatch=minbatch per=thread for i in sp:ep
        @inbounds begin
            if length(direct[i]) >= length(reverse[i])
                append!(direct[i], reverse[i])
            else
                append!(reverse[i], direct[i])
                direct[i] = reverse[i]
            end
        end
    end
end
