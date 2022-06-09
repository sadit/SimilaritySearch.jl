# This file is a part of SimilaritySearch.jl

export allknn
using Polyester
"""
    allknn(g::AbstractSearchContext, k::Integer; minbatch=0, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

Parameters:

- `g`: the index
- `k`: the number of neighbors to retrieve
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `pools`: A pools object, dependent of `g`

Returns:

- `knns` a (k, n) matrix of identifiers; the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
- `dists` a (k, n) matrix of distances; the i-th column corresponds to the i-th object in the dataset.
    Zero values in `knns` should be ignored in `dists`

"""
function allknn(g::AbstractSearchContext, k::Integer; minbatch=0, pools=getpools(g))
    allknn(g, KnnResultSet(k, length(g)); minbatch, pools)
end

function _allknn_loop(g::SearchGraph, i, R, pools)
    k = size(R, 1)
    res = reuse!(R, i)
    vstate = getvstate(length(g), pools)
    c = g[i]
    visit!(vstate, i)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    ##prev = typemax(Float32)
    for h in g.links[i] # hints
        visited(vstate, h) && continue
        search(g.search_algo, g, c, res, h, pools; vstate)
        ## curr = maximum(res)
        length(res) == k && break
        ##curr == prev && break
        ## prev = curr
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
    R
end

function _allknn_loop(g, i, R, pools)
    k = size(R, 1) + 1
    res = getknnresult(k, pools)
    @inbounds search(g, g[i], res)
    _allknn_inner_loop(res, i, R.id, R.dist)
end

@inline function _allknn_inner_loop(res, i, knns, dists)
    j = 0
    @inbounds for (id, dist) in res
        i == id && continue
        j += 1
        knns[j, i] = id
        dists[j, i] = dist
    end
end

"""
allknn(index, R::KnnResultSet; minbatch=0, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

Arguments:

- `g`: the index
- `knns`: an uninitialized integer matrix of (k, n) size for storing the `k` nearest neighbors of the `n` elements
- `dists`: an uninitialized floating point matrix of (k, n) size for storing the `k` nearest distances of the `n` elements
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `pools`: A pools object, dependent of `g`

Results:

- `knns` and `dists` are returned. Note that the index can retrieve less than `k` objects, and these are represented as zeros at the end of each column (can happen)
"""
function allknn(g::AbstractSearchContext, R::KnnResultSet; minbatch=0, pools=getpools(g))
    n = length(g)
    @assert n > 0
    minbatch = getminibatch(minbatch, n)

    if minbatch > 0
        @batch minbatch=minbatch per=thread for i in 1:n
            _allknn_loop(g, i, R, pools)
        end
        #=B = floor(Int, n / minbatch)
        Threads.@threads for b in 1:B
            ep = b * minbatch
            sp = ep - minbatch + 1
            ep = min(ep, n)

            for i in sp:ep
            
                _allknn_loop(g, i, knns, dists, pools)
            end
        end
        =#
    else
        for i in 1:n
            _allknn_loop(g, i, R, pools)
        end
    end
    
    R.id, R.dist
end
