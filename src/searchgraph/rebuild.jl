# This file is a part of SimilaritySearch.jl

export rebuild

"""
    rebuild(g::SearchGraph; context=SearchGraphContext())

Rebuilds the `SearchGraph` index but seeing the whole dataset for the incremental construction, i.e.,
it can connect the i-th vertex to its knn in the 1..n possible vertices instead of its knn among 1..(i-1) as in the original algorithm.

# Arguments

- `g`: The search index to be rebuild.
- `context`: The context to run the procedure, it can differ from the original one.
- `minbatch`: controls how the multithreading is made, see [`getminbatch`](@ref)

"""
function rebuild(g::SearchGraph, context::SearchGraphContext)
    n = length(g)
    @assert n > 0
    direct = Vector{Vector{UInt32}}(undef, n)  # this separated links version needs has easier multithreading/locking needs
    reverse = Vector{Vector{UInt32}}(undef, n)
    minbatch = context.minbatch < 0 ? n : getminbatch(context.minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        @inbounds direct[i] = find_neighborhood(g, context, database(g, i), hints=first(neighbors(g.adj, i)))
        reverse[i] = Vector{UInt32}(undef, 0)
    end

    rebuild_connect_reverse_links!(context.neighborhood, direct, reverse, g.adj.locks, 1, length(g), minbatch)
    G = copy(g; adj=AdjacencyList(direct), hints=copy(g.hints), search_algo=copy(g.search_algo))
    execute_callbacks(G, context, force=true)
    G
end

function rebuild_connect_reverse_links!(N, direct, reverse, locks, sp, ep, minbatch)
    @batch minbatch=minbatch per=thread for i in sp:ep
        j = 0
        D = direct[i]
        p = 1f0
        @inbounds while j < length(D)
            j += 1
            id = D[j]
            if i == id
                D[j] = D[end]
                pop!(D)
                j -= 1
                continue
            end

            if rand(Float32) < p
                lock(locks[id])
                try
                    push!(reverse[id], i)
                finally
                    unlock(locks[id])
                end
                p *= N.connect_reverse_links_factor
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
