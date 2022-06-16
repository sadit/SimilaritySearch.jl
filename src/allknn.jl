# This file is a part of SimilaritySearch.jl

export allknn

"""
    allknn(g::AbstractSearchContext, k::Integer; minbatch=0, pools=getpools(g)) -> knns, dists
    allknn(g, knns, dists; minbatch=0, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

# Parameters:

- `g`: the index
- Query specification and result:
   - `k`: the number of neighbors to retrieve
   - `knns`: an uninitialized integer matrix of (k, n) size for storing the `k` nearest neighbors of the `n` elements
   - `dists`: an uninitialized floating point matrix of (k, n) size for storing the `k` nearest distances of the `n` elements

- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `pools`: A pools object, dependent of `g`

# Returns:

- `knns` a (k, n) matrix of identifiers; the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
- `dists` a (k, n) matrix of distances; the i-th column corresponds to the i-th object in the dataset.
    Zero values in `knns` should be ignored in `dists`

# Keyword arguments
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `pools`: `pools`: A pools object, dependent of `g`
 
# Note:
This function was introduced in `v0.8` series, and removes self references automatically.
In `v0.9` the self reference is kept since removing from the algorithm introduces a considerable overhead.    
"""
function allknn(g::AbstractSearchContext, k::Integer; minbatch=0, pools=getpools(g))
    n = length(g)
    knns = zeros(Int32, k, n)
    dists = Matrix{Float32}(undef, k, n)
    allknn(g, knns, dists; minbatch, pools)
end

function allknn(g::AbstractSearchContext, knns::AbstractMatrix{Int32}, dists::AbstractMatrix{Float32}; minbatch=0, pools=getpools(g))
    n = length(g)
    @assert n > 0
    minbatch = getminbatch(minbatch, n)

    if minbatch > 0
        @batch minbatch=minbatch per=thread for i in 1:n
            _allknn_loop(g, i, knns, dists, pools)
        end
    else
        for i in 1:n
            _allknn_loop(g, i, knns, dists, pools)
        end
    end
    
    knns, dists
end

function _allknn_loop(g::SearchGraph, i, knns, dists, pools)
    k = size(knns, 1)
    res = getknnresult(k, pools)
    vstate = getvstate(length(g), pools)
    c = g[i]
    # visit!(vstate, i)
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
            search(g.search_algo, g, c, res, h, pools; vstate)
            length(res) == k && break
        end
    end

    _k = length(res)
    knns[1:_k, i] .= res.id
    dists[1:_k, i] .= res.dist
end

function _allknn_loop(g, i, knns, dists, pools)
    k = size(knns, 1)
    res = getknnresult(k, pools)
    @inbounds search(g, g[i], res; pools)

    _k = length(res)
    knns[1:_k, i] .= res.id
    dists[1:_k, i] .= res.dist
end
