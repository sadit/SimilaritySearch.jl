# This file is a part of SimilaritySearch.jl

export allknn

"""
    allknn(g::AbstractSearchContext, k::Integer; minbatch=0, pools=getpools(g)) -> knns, dists
    allknn(g, knns::Matrix{Int32}, dists::Matrix{Float32}; minbatch=0, pools=getpools(g)) -> knns, dists
    allknn(g, R::KnnResultSet; minbatch=0, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

# Parameters:

- `g`: the index
- Query specification and result:
    - `k`: the number of neighbors to retrieve
    - `knns`: an uninitialized integer matrix of (k, n) size for storing the `k` nearest neighbors of the `n` elements
    - `dists`: an uninitialized floating point matrix of (k, n) size for storing the `k` nearest distances of the `n` elements
    - `R`: an uninitialized `KnnResultSet` (contains `knns` and `dists` internally, along with the lenghts of the results)
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `pools`: A pools object, dependent of `g`

# Returns:

- `knns` a (k, n) matrix of identifiers; the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
- `dists` a (k, n) matrix of distances; the i-th column corresponds to the i-th object in the dataset.
    Zero values in `knns` should be ignored in `dists`

These are the same buffers passed as arguments in some of the function methods.

# Note:
This function was introduced in `v0.8` series, and removes self references automatically.
In `v0.9` the self reference is kept since removing from the algorithm introduces a considerable overhead.
"""
allknn(g::AbstractSearchContext, k::Integer; minbatch=0, pools=getpools(g)) = allknn(g, KnnResultSet(k, length(g)); minbatch, pools)
allknn(g::AbstractSearchContext, knns::Matrix{Int32}, dists::Matrix{Float32}; minbatch=0, pools=getpools(g)) = allknn(g, KnnResultSet(knns, dists); minbatch, pools)

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

function _allknn_loop(g::SearchGraph, i::Integer, res::KnnResult, pools)
    vstate = getvstate(length(g), pools)
    @inbounds _allknn_loop_barrier(g, g[i], res, g.links[i], pools, vstate)
    # _allknn_fix_self(i, res)
end

function _allknn_loop_barrier(g::SearchGraph, c, res, hints, pools, vstate)
    k = maxlength(res)

    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    for h in hints # hints
        visited(vstate, h) && continue
        search(g.search_algo, g, c, res, h, pools; vstate)
        length(res) == k && break
    end

    # search(g.search_algo, g, c, res, rand(hints), pools; vstate)
    # search(g.search_algo, g, c, res, hints, pools; vstate)

    if length(res) < k
        for _ in 1:k
            h = rand(1:length(g))
            visited(vstate, h) && continue
            search(g.search_algo, g, c, res, h, pools; vstate)
            length(res) == k && break
        end
    end
end

function _allknn_loop(index, i::Integer, res::KnnResult, pools)
    @inbounds search(index, index[i], res; pools)
    # _allknn_fix_self(i, res)
end

"""
    _allknn_fix_self(res::KnnResult)

Set self references to the first position on the result set.
Near duplicates and floating point arithmetic could make that self references goes beyond the first entry.
"""
function _allknn_fix_self(selfID::Integer, res::KnnResult)
    selfID = convert(Int32, selfID)
    I = idview(res)
    @assert length(I) == length(Set(I))
    @inbounds for (i, objID) in enumerate(I)
        if selfID === objID
            if i > 1
                I[1], I[i] = I[i], I[1]
            end

            return
        end
    end
end
