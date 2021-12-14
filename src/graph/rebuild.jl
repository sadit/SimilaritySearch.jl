# This file is a part of SimilaritySearch.jl

export rebuild


function rebuild(g::SearchGraph)
    n = length(g)
    @assert n > 0
    links = Vector{Vector{Int32}}(undef, n)

    Threads.@threads for i in 1:n
        @inbounds links[i] = find_neighborhood(g, g[i], hints=g.links[i][1], self_link=i)
    end
    
    G = copy(g; links=links, neighborhood=copy(g.neighborhood), locks=copy(g.locks), hints=copy(g.hints), callbacks=copy(g.callbacks))
    _connect_links__(G, 1, length(G))
    callbacks(G, force=true)
    G
end

function _connect_links__(index, sp, ep)
    Threads.@threads for i in sp:ep
        @inbounds for id in index.links[i]
            id < 0 && continue
            lock(index.locks[id])
            try
                push!(index.links[id], -i)
                # sat_should_push(index.links[id], index, index[i], i, -1.0) && push!(index.links[id], i)
            finally
                unlock(index.locks[id])
            end
        end
    end

    Threads.@threads for i in sp:ep
        @inbounds L = index.links[i]
        @inbounds for j in eachindex(L)
            L[j] = abs(L[j])
        end
    end
end

