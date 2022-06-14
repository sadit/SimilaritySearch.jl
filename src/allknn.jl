# This file is a part of SimilaritySearch.jl

export allknn

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
    n = length(g)
    R = KnnResultSet(k, n)
    allknn(g, R; minbatch, pools)
end

function _allknn_loop(g::SearchGraph, i::Integer, res::KnnResult, pools)
    vstate = getvstate(length(g), pools)
    # visit!(vstate, i)
    @inbounds _allknn_loop_barrier(g, i, g[i], res, g.links[i], pools, vstate)
end

function _allknn_loop_barrier(g::SearchGraph, i, c, res, hints, pools, vstate)
    k = maxlength(res)

    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    for h in hints # hints
        visited(vstate, h) && continue
        search(g.search_algo, g, c, res, h, pools; vstate)
        length(res) == k && break
    end
    # search(g.search_algo, g, c, res, hints, pools; vstate)

    #=if length(res) < k
        for _ in 1:k
            h = rand(1:length(g))
            visited(vstate, h) && continue
            search(g.search_algo, g, c, res, h, pools; vstate)
            length(res) == k && break
        end
    end=#
end

function _allknn_loop(index, i::Integer, dst::KnnResult, pools)
    @inbounds search(index, index[i], dst; pools)
end

"""
allknn(g, knns, dists; parallel_block=512, pools=getpools(g)) -> knns, dists

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
    minbatch = getminbatch(minbatch, n)

    if minbatch > 0
        @batch minbatch=minbatch per=thread for i in 1:n
            _allknn_loop(g, i, R[i], pools)
        end
    else
        for i in 1:n
            _allknn_loop(g, i, R[i], pools)
        end
    end
    
    R.id, R.dist
end
