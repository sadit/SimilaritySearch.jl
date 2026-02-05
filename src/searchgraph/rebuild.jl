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
    progress=Progress(length(g); desc="rebuild", dt=4)
)
    n = length(g)
    @assert n > 0
    direct = Vector{Vector{UInt32}}(undef, n)  # this separated links version needs has easier multithreading/locking needs
    reverse = Vector{Vector{UInt32}}(undef, n)
    minbatch = getminbatch(ctx, n)

    let progress = progress
        Threads.@threads :static for j in 1:minbatch:n
            @inbounds for i in j:min(n, j + minbatch - 1)
                neighborhood = find_neighborhood(g, ctx, database(g, i); hints=first(neighbors(g.adj, i)))
                progress !== nothing && next!(progress)
                direct[i] = collect(IdView(neighborhood))
                # @info length(direct[i]) neighbors_length(g.adj, i) 
                reverse[i] = UInt32[]
            end
        end
    end

    rebuild_connect_reverse_links!(ctx.neighborhood, direct, reverse, g.adj.locks, 1, length(g), minbatch)
    G = SearchGraph(distance(g), database(g), AdjacencyList(direct), copy(g.hints), Ref(g.algo[]), Ref(length(g)))
    execute_callbacks(G, ctx, force=true)
    G
end

function rebuild_connect_reverse_links!(N, direct, reverse, locks, sp, ep, minbatch)
    Threads.@threads :static for jj in sp:minbatch:ep
        for i in jj:min(ep, jj + minbatch - 1)
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
    end

    Threads.@threads :static for jj in sp:minbatch:ep
        @inbounds for i in jj:min(ep, jj + minbatch - 1)
            if length(direct[i]) >= length(reverse[i])
                append!(direct[i], reverse[i])
            else
                append!(reverse[i], direct[i])
                direct[i] = reverse[i]
            end
        end
    end
end
