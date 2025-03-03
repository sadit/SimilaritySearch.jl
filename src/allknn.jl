# This file is a part of SimilaritySearch.jl

export allknn
using ProgressMeter

"""
    allknn(g::AbstractSearchIndex, context, k::Integer; minbatch=0, pools=getpools(g)) -> knns, dists
    allknn(g, context, knns, dists; minbatch=0, pools=getpools(g)) -> knns, dists

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

# Parameters:

- `g`: the index
- `context`: the index's context (caches, hyperparameters, logger, etc)
- Query specification and result:
   - `k`: the number of neighbors to retrieve
   - `knns`: an uninitialized integer matrix of (k, n) size for storing the `k` nearest neighbors of the `n` elements
   - `dists`: an uninitialized floating point matrix of (k, n) size for storing the `k` nearest distances of the `n` elements

- `context`: caches, hyperparameters and meta specifications, e.g., see [`SearchGraphContext`](@ref)

# Returns:

- `knns` a (k, n) matrix of identifiers; the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
- `dists` a (k, n) matrix of distances; the i-th column corresponds to the i-th object in the dataset.
    Zero values in `knns` should be ignored in `dists`
 
# Note:
This function was introduced in `v0.8` series, and removes self references automatically.
In `v0.9` the self reference is kept since removing from the algorithm introduces a considerable overhead.    
"""
function allknn(g::AbstractSearchIndex, context::AbstractContext, k::Integer)
    n = length(g)
    knns = Matrix{Int32}(undef, k, n)
    dists = Matrix{Float32}(undef, k, n)
    allknn(g, context, knns, dists)
end

function allknn(g::AbstractSearchIndex, context::AbstractContext, knns::AbstractMatrix{Int32}, dists::AbstractMatrix{Float32})
    k, n = size(knns, 1), length(g)  # don't use n from knns, use directly length(g), i.e., allows to reuse knns
    # @assert n > 0 && k > 0 && n == length(g)
    #knns_ = pointer(knns)
    #dists_ = pointer(dists)
    knns_ = PtrArray(knns)
    dists_ = PtrArray(dists)
    if context.minbatch < 0
        for i in 1:n
            res = getknnresult(k, context)
            allknn_single_search(g, context, i, res)
            _k = length(res)
            knns_[1:_k, i] .= res.id
            _k < k && (knns_[_k+1:k] .= zero(Int32))
            dists_[1:_k, i] .= res.dist 
        end
    else
        minbatch = getminbatch(context.minbatch, n)

        #@batch minbatch=minbatch per=thread for i in 1:n
        P = Iterators.partition(1:n, minbatch) |> collect
        @showprogress desc="allknn" dt=4 Threads.@threads :static for R in P
            for i in R
                res = getknnresult(k, context)
                allknn_single_search(g, context, i, res)
                _k = length(res)
                #unsafe_copyto_knns_and_dists!(knns_, pointer(res.id), dists_, pointer(res.dist), i, _k, k)
                @inbounds for j in 1:_k
                    u = res.items[j]
                    knns_[j, i] = u.id
                    dists_[j, i] = u.weight
                end
            
                for j in _k+1:k
                    knns_[j, i] = zero(Int32)
                end
            end
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


function allknn_single_search(g::SearchGraph, context::SearchGraphContext, i::Integer, res::KnnResult)
    cost = 0
    vstate = getvstate(length(g), context)
    q = database(g, i)
    # visit!(vstate, i)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    
    for h in neighbors(g.adj, i) # hints
        visited(vstate, convert(UInt64, h)) && continue
        cost += search(g.algo, g, context, q, res, h; vstate).cost
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
            search(g.algo, g, q, res, h, pools; vstate)
            length(res) == k && break
        end
    end=#

    (; res, cost)
end

function allknn_single_search(g::AbstractSearchIndex, context::AbstractContext, i::Integer, res::KnnResult)
    search(g, context, database(g, i), res,)
end
