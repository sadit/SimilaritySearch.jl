# This file is a part of SimilaritySearch.jl

export allknn

function allknn(g::AbstractSearchContext, k::Integer; parallel=false)
    n = length(g)
    knns = zeros(Int32, k, n)
    dists = Matrix{Float32}(undef, k, n)

    allknn(g, knns, dists; parallel)
end

function _allknn_loop(g::SearchGraph, i, knns, dists)
    k = size(knns, 1) + 1
    res = getknnresult(k)
    @inbounds search(g, g[i], res, hints=g.links[i][1])
    _allknn_inner_loop(res, i, knns, dists)
end

function _allknn_loop(g, i, knns, dists)
    k = size(knns, 1) + 1
    res = getknnresult(k)
    @inbounds search(g, g[i], res)
    _allknn_inner_loop(res, i, knns, dists)
end

@inline function _allknn_inner_loop(res, i,  knns, dists)
    j = 0
    @inbounds for (id, dist) in res
        i == id && continue
        j += 1
        knns[j, i] = id
        dists[j, i] = dist
    end
end

function allknn(g::AbstractSearchContext, knns::AbstractMatrix{Int32}, dists::AbstractMatrix{Float32}; parallel=false)
    n = length(g)
    @assert n > 0

    if parallel
        Threads.@threads for i in 1:n
            _allknn_loop(g, i, knns, dists)
        end
    else
        for i in 1:n
            _allknn_loop(g, i, knns, dists)
        end
    end
    
    knns, dists
end
