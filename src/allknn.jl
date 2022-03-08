# This file is a part of SimilaritySearch.jl

export allknn

"""
    allknn(g::AbstractSearchContext, k::Integer; parallel=false, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

Parameters:

- `g`: the index
- `k`: the number of neighbors to retrieve
- `parallel`: If true, the construction will use all threads available threads
- `pools`: A pools object, dependent of `g`

Returns:

- `knns` a (k, n) matrix of identifiers; the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
- `dists` a (k, n) matrix of distances; the i-th column corresponds to the i-th object in the dataset.
    Zero values in `knns` should be ignored in `dists`

"""
function allknn(g::AbstractSearchContext, k::Integer; parallel=false, pools=getpools(g))
    n = length(g)
    knns = zeros(Int32, k, n)
    dists = Matrix{Float32}(undef, k, n)
    allknn(g, knns, dists; parallel, pools)
end

function _allknn_loop(g::SearchGraph, i, knns, dists, pools)
    k = size(knns, 1) + 1
    res = getknnresult(k, pools)
    vstate = getvstate(length(g), pools)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    for h in g.links[i] # hints
        visited(vstate, h) && continue
        @inbounds search(g.search_algo, g, g[i], res, h, pools; vstate)
        length(res) == k && break
    end

    # again for the same issue
    for j in 1:2k
        h = rand(1:length(g))
        visited(vstate, h) && continue
        @inbounds search(g.search_algo, g, g[i], res, h, pools; vstate)
        length(res) == k && break
    end

    _allknn_inner_loop(res, i, knns, dists)
end

function _allknn_loop(g, i, knns, dists, pools)
    k = size(knns, 1) + 1
    res = getknnresult(k, pools)
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

"""
allknn(g, knns, dists; parallel=false, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

Arguments:

- `g`: the index
- `knns`: an uninitialized integer matrix of (k, n) size for storing the `k` nearest neighbors of the `n` elements
- `dists`: an uninitialized floating point matrix of (k, n) size for storing the `k` nearest distances of the `n` elements
- `parallel`: If true, the construction will use all threads available threads
- `pools`: A pools object, dependent of `g`

Results:

- `knns` and `dists` are returned. Note that the index can retrieve less than `k` objects, and these are represented as zeros at the end of each column (can happen)
"""
function allknn(g::AbstractSearchContext, knns::AbstractMatrix{Int32}, dists::AbstractMatrix{Float32}; parallel=false, pools=getpools(g))
    n = length(g)
    @assert n > 0

    if parallel
        Threads.@threads for i in 1:n
            _allknn_loop(g, i, knns, dists, pools)
        end
    else
        for i in 1:n
            _allknn_loop(g, i, knns, dists, pools)
        end
    end
    
    knns, dists
end
