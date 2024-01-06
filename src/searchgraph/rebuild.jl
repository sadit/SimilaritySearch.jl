# This file is a part of SimilaritySearch.jl

export rebuild

"""
    rebuild(g::SearchGraph; setup=SearchGraphSetup(), minbatch=0, pools=getpools(g))

Rebuilds the `SearchGraph` index but seeing the whole dataset for the incremental construction, i.e.,
it can connect the i-th vertex to its knn in the 1..n possible vertices instead of its knn among 1..(i-1) as in the original algorithm.

# Arguments

- `g`: The search index to be rebuild.
- `neighborhood`: The neighborhood strategy to follow in the rebuild, it can differ from the original one.
- `callbacks`: The set of callbacks
- `minbatch`: controls how the multithreading is made, see [`getminbatch`](@ref)
- `pools`: The set of caches for the indexes

"""
function rebuild(g::SearchGraph; setup=SearchGraphSetup(), minbatch=0, pools=getpools(g))
    n = length(g)
    @assert n > 0
    direct = Vector{Vector{UInt32}}(undef, n)  # this separated links version needs has easier multithreading/locking needs
    reverse = Vector{Vector{UInt32}}(undef, n)
    minbatch = minbatch < 0 ? n : getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        @inbounds direct[i] = find_neighborhood(g, database(g, i), setup.neighborhood, pools, hints=first(neighbors(g.adj, i)))
        reverse[i] = Vector{UInt32}(undef, 0)
    end

    rebuild_connect_reverse_links!(direct, reverse, g.adj.locks, 1, length(g), minbatch)
    G = copy(g; adj=AdjacencyList(direct), hints=copy(g.hints), search_algo=copy(g.search_algo))
    execute_callbacks(setup, G, force=true)
    G
end

function rebuild_connect_reverse_links!(direct, reverse, locks, sp, ep, minbatch)
    @batch minbatch=minbatch per=thread for i in sp:ep
        j = 0
        D = direct[i]
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
