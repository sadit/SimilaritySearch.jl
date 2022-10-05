# This file is a part of SimilaritySearch.jl

export allknn

"""
    allknn(g::AbstractSearchIndex, k::Integer; minbatch=0, pools=getpools(g)) -> knns, dists
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
function allknn(g::AbstractSearchIndex, k::Integer; minbatch=0, pools=getpools(g))
    n = length(g)
    knns = Matrix{Int32}(undef, k, n)
    dists = Matrix{Float32}(undef, k, n)
    allknn(g, knns, dists; minbatch, pools)
end

function allknn(g::AbstractSearchIndex, knns::AbstractMatrix{Int32}, dists::AbstractMatrix{Float32}; minbatch=0, pools=getpools(g))
    k, n = size(knns)
    # @assert n > 0 && k > 0 && n == length(g)
    #knns_ = pointer(knns)
    #dists_ = pointer(dists)
    knns_ = PtrArray(knns)
    dists_ = PtrArray(dists)
    if minbatch < 0
        for i in 1:n
            res = _allknn_loop(g, i, k, pools)
            _k = length(res)
            knns_[1:_k, i] .= res.id
            _k < k && (knns_[_k+1:k] .= zero(Int32))
            dists_[1:_k, i] .= res.dist 
        end
    else
        minbatch = getminbatch(minbatch, n)

        @batch minbatch=minbatch per=thread for i in 1:n
            res = _allknn_loop(g, i, k, pools)
            _k = length(res)
            #unsafe_copyto_knns_and_dists!(knns_, pointer(res.id), dists_, pointer(res.dist), i, _k, k)
            knns_[1:_k, i] .= res.id
            _k < k && (knns_[_k+1:k, i] .= zero(Int32))
            dists_[1:_k, i] .= res.dist
        end
    end
    
    knns, dists
end

#=
@inline function unsafe_copyto_knns_and_dists!(knns, id, dists, dist, i, _k, k)
    sp = 4*k*(i-1) + 1
    knns = knns + sp
    unsafe_copyto!(knns, id, _k)
    for i in _k+1:k
        unsafe_store!(knns, zero(Int32),  i)
    end

    dists = dists + sp
    unsafe_copyto!(dists, dist, _k)
end=#


function _allknn_loop(g::SearchGraph, i, k, pools)
    res = getknnresult(k, pools)
    vstate = getvstate(length(g), pools)
    q = g[i]
    # visit!(vstate, i)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    
    for h in g.links[i] # hints
        visited(vstate, convert(UInt64, h)) && continue
        search(g.search_algo, g, q, res, h, pools; vstate)
        # length(res) == k && break
    end

    #=
    Δ = 1.9f0
    maxvisits = bs.maxvisits ÷ 2
    bsize = 1 # ceil(Int, bs.bsize / 2)
    search(bs, g, c, res, i, pools; vstate, Δ, maxvisits, bsize)
    =#

    # again for the same issue
    #=if length(res) < k
        for _ in 1:k
            h = rand(1:length(g))
            visited(vstate, convert(UInt64, h)) && continue
            search(g.search_algo, g, q, res, h, pools; vstate)
            length(res) == k && break
        end
    end=#

    res
end

function _allknn_loop(g, i, k, pools)
    res = getknnresult(k, pools)
    @inbounds search(g, database(g, i), res; pools)
    res
end
