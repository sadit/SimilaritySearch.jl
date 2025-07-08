# This file is a part of SimilaritySearch.jl

export allknn
using ProgressMeter

"""
    allknn(g::AbstractSearchIndex, ctx, k::Integer; minbatch=0, pools=getpools(g)) -> knns
    allknn(g, ctx, knns; minbatch=0, pools=getpools(g)) -> knns

Computes all the k nearest neighbors (all vs all) using the index `g`. It removes self references.

# Parameters:

- `g`: the index
- `ctx`: the index's ctx (caches, hyperparameters, logger, etc)
- Query specification and result:
   - `k`: the number of neighbors to retrieve
   - `knns`: an uninitialized IdWeight matrix of (k, n) size for storing the `k` nearest neighbors and sitances of the `n` elements

- `ctx`: caches, hyperparameters and meta specifications, e.g., see [`SearchGraphContext`](@ref)

# Returns:

- `knns` a (k, n) matrix of IdWeight elements (integers, floats); the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
 
# Note:
This function was introduced in `v0.8` series, and removes self references automatically.
In `v0.9` the self reference is kept since removing from the algorithm introduces a considerable overhead.    
"""
function allknn(g::AbstractSearchIndex, ctx::AbstractContext, k::Integer; sort=true)
    n = length(g)
    knns = Matrix{IdWeight}(undef, k, n)
    allknn(g, ctx, knns; sort)
end

function allknn(g::AbstractSearchIndex, ctx::AbstractContext, knns::AbstractMatrix; sort::Bool=true)
    m = length(g)  # don't use n from knns, use directly length(g), i.e., allows to reuse knns
    k, n = size(knns)
    @assert n > 0 && n == m
    @assert 0 < k <= n
    #knns_ = pointer(knns)
    #dists_ = pointer(dists)
    if ctx.minbatch < 0
        for i in 1:n
            res = knn(@view knns[:, i])
            allknn_single_search(g, ctx, i, res)
            sort && sortitems!(res)
        end
    else
        minbatch = getminbatch(ctx.minbatch, n)

        #@batch minbatch=minbatch per=thread for i in 1:n
        P = Iterators.partition(1:n, minbatch) |> collect
        @showprogress desc="allknn" dt=4 Threads.@threads :static for R in P
            for i in R
                res = knn(@view knns[:, i])
                allknn_single_search(g, ctx, i, res)
                sort && sortitems!(res)
            end
        end
    end
    
    knns
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


function allknn_single_search(g::SearchGraph, ctx::SearchGraphContext, i::Integer, res)
    cost = 0
    vstate = getvstate(length(g), ctx)
    q = database(g, i)
    # visit!(vstate, i)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    
    for h in neighbors(g.adj, i) # hints
        visited(vstate, convert(UInt64, h)) && continue
        cost += search(g.algo, g, ctx, q, res, h; vstate).cost
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

    res.cost = cost
    res
end

function allknn_single_search(g::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    search(g, ctx, database(g, i), res)
end
