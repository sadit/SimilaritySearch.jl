# This file is a part of SimilaritySearch.jl

export rebuild


function rebuild(g::SearchGraph)
    n = length(g)
    @assert n > 0
    direct = Vector{Vector{Int32}}(undef, n)  # this separated links version needs has easier multithreading/locking needs
    reverse = Vector{Vector{Int32}}(undef, n)

    Threads.@threads for i in 1:n
        @inbounds direct[i] = find_neighborhood(g, g[i], hints=g.links[i][1])
        reverse[i] = Vector{Int32}(undef, 0)
    end
    
    G = copy(g; links=direct, neighborhood=copy(g.neighborhood), locks=copy(g.locks), hints=copy(g.hints), callbacks=copy(g.callbacks), search_algo=copy(g.search_algo))
    _connect_reverse_links_neg(G.links, reverse, G.locks, 1, length(G))
    callbacks(G, force=true)
    G
end

function _connect_reverse_links_neg(direct, reverse, locks, sp, ep)
    Threads.@threads for i in sp:ep
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

    Threads.@threads for i in sp:ep
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

