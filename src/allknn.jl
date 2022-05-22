# This file is a part of SimilaritySearch.jl

export allknn

"""
    allknn(g::AbstractSearchIndex, k::Integer; parallel=false, pools=getpools(g)) -> knns, dists

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
function allknn(g::AbstractSearchIndex, k::Integer; parallel=false, pools=getpools(g))
    n = length(g)
    R = KnnResultSet(k, n)
    knns = allknn(g, R; parallel, pools)
    knns.id, knns.dist
end

function _allknn_loop(g::SearchGraph, k, i, pools)
    k += 1
    res = getknnresult(k, pools)
    vstate = getvstate(length(g), pools)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    for h in g.links[i] # hints
        visited(vstate, h) && continue
        @inbounds search(g.search_algo, g, g[i], res, h, pools; vstate)
        length(res) == k && break
    end

    # again for the same issue
    if length(res) < k
        for j in 1:k
            h = rand(1:length(g))
            visited(vstate, h) && continue
            @inbounds search(g.search_algo, g, g[i], res, h, pools; vstate)
            length(res) == k && break
        end
    end

    res
end

function _allknn_loop(g::AbstractSearchIndex, k, i, pools)
    k += 1
    res = getknnresult(k, pools)
    @inbounds search(g, g[i], res)
    res
end

@inline function _allknn_inner_loop(knns::KnnResultSet, id, res::KnnResultSingle)
    i = 1
    j = 0
    k = size(knns, 1)
    
    @inbounds while j < k
        id_, dist_ = res[i]
        i += 1
        id == id_ && continue
        j += 1
        knns.id[j, id] = id_
        knns.dist[j, id] = dist_
    end
end

"""
allknn(g, knns; parallel=false, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

Arguments:

- `g`: the index
- `knns`: a knn result set of size ``(k, numqueries)``
- `parallel`: If true, the construction will use all threads available threads
- `pools`: A pools object, dependent of `g`

Results:

- `knns` and `dists` are returned. Note that the index can retrieve less than `k` objects, and these are represented as zeros at the end of each column (can happen)
"""
function allknn(g::AbstractSearchIndex, knns::KnnResultSet; parallel=false, pools=getpools(g))
    n = length(g)
    k = size(knns, 1)
    @assert n > 0

    if parallel
        Threads.@threads for i in 1:n
            res = _allknn_loop(g, k, i, pools)
            _allknn_inner_loop(knns, i, res)
        end
    else
        for i in 1:n
            res = _allknn_loop(g, k, i, pools)
            _allknn_inner_loop(knns, i, res)
        end
    end
    
    knns
end
