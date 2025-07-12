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

- `knns` a (k, n) matrix of `IdWeight` elements, i.e., `zeros(IdWeight, k, n)`; the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
 
"""
function allknn(g::AbstractSearchIndex, ctx::AbstractContext, k::Integer; sort=true)
    n = length(g)
    knns = zeros(IdWeight, k, n)
    allknn(g, ctx, knns; sort)
end

function allknn(g::AbstractSearchIndex, ctx::AbstractContext, knns::AbstractMatrix; sort::Bool=true)
    m = length(g)  # don't use n from knns, use directly length(g), i.e., allows to reuse knns
    k, n = size(knns)
    @assert n > 0 && n == m
    @assert 0 < k <= n
    if ctx.minbatch < 0
        for i in 1:n
            res = knn(@view knns[:, i])
            res = allknn_single_search!(g, ctx, i, res)
            sort && sortitems!(res)
        end
    else
        minbatch = getminbatch(ctx.minbatch, n)

        #@batch minbatch=minbatch per=thread for i in 1:n
        P = Iterators.partition(1:n, minbatch) |> collect
        @showprogress desc="allknn" dt=4 Threads.@threads :static for R in P
            for i in R
                res = knn(@view knns[:, i])
                res = allknn_single_search!(g, ctx, i, res)
                sort && sortitems!(res)
            end
        end
    end
    
    knns
end

function allknn_single_search!(g::SearchGraph, ctx::SearchGraphContext, i::Integer, res)
    cost = 0
    vstate = getvstate(length(g), ctx)
    q = database(g, i)
    # visit!(vstate, i)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)
    
    for h in neighbors(g.adj, i) # hints
        visited(vstate, convert(UInt64, h)) && continue
        res = search(g.algo, g, ctx, q, res, h; vstate)
        cost += res.cost
        # length(res) == k && break
    end

    @reset res.cost = convert(typeof(res.cost), cost)
    res
end

function allknn_single_search!(g::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    search(g, ctx, database(g, i), res)
end
