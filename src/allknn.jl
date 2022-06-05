# This file is a part of SimilaritySearch.jl

export allknn
using Polyester
"""
    allknn(g::AbstractSearchContext, k::Integer; parallel_block=32, pools=getpools(g)) -> knns, dists

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
function allknn(g::AbstractSearchContext, k::Integer; parallel_block=32, pools=getpools(g))
    allknn(g, KnnResultSet(k, length(g)); parallel_block, pools)
end

function _allknn_loop(g::SearchGraph, i, R, pools)
    k = size(R, 1)
    res = reuse!(R, i)
    vstate = getvstate(length(g), pools)
    visit!(vstate, i)
    c = g[i]
    #@inbounds search(g.search_algo, g, g[i], res, g.links[i], pools; vstate)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    for h in g.links[i] # hints
        visited(vstate, h) && continue
        search(g.search_algo, g, c, res, h, pools; vstate)
        length(res) == k && break
    end

    # again for the same issue
    if length(res) < k
        for _ in 1:k
            h = rand(1:length(g))
            visited(vstate, h) && continue
            @inbounds search(g.search_algo, g, c, res, h, pools; vstate)
            length(res) == k && break
        end
    end
    R
end

function _allknn_loop(g, i, R, pools)
    k = size(R, 1) + 1
    res = getknnresult(k, pools)
    @inbounds search(g, g[i], res)
    _allknn_inner_loop(res, i, R.id, R.dist)
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
allknn(g, knns, dists; parallel_block=32, pools=getpools(g)) -> knns, dists

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
function allknn(g::AbstractSearchContext, R; parallel_block=32, pools=getpools(g))
    n = length(g)
    @assert n > 0

    if parallel_block > 1
        @batch minbatch=512 per=thread for i in 1:n
            _allknn_loop(g, i, R, pools)
        end
    else
        for i in 1:n
            _allknn_loop(g, i, R, pools)
        end
    end
    
    R.id, R.dist
end
