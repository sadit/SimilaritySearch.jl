# This file is a part of SimilaritySearch.jl

export allknn
using ProgressMeter

"""
    allknn(index, ctx, k::Integer; minbatch=0, sort=true, show_progress=true) -> knns

Computes all the k nearest neighbors (all vs all) using the given index. User must remove self references

# Parameters:
- `index`: the index
- `ctx`: the index's ctx (caches, hyperparameters, logger, etc)
- Query specification and result:
   - `k`: the number of neighbors to retrieve
   - `knns`: an uninitialized IdWeight matrix of (k, n) size for storing the `k` nearest neighbors and sitances of the `n` elements

- `sort`: ensures that result set is presented in ascending order by distance
- `show_progress`: enables or disables progress bar

# Returns:

- `knns` a (k, n) matrix of `IdWeight` elements, i.e., `zeros(IdWeight, k, n)`; the i-th column corresponds to the i-th object in the dataset.
    Zeros can happen to the end of each column meaning that the retrieval was less than the desired `k`
 
"""
function allknn(g::AbstractSearchIndex, ctx::AbstractContext, k::Integer;
    sort::Bool=true,
    progress=Progress(length(g); desc="allknn", dt=4),
    minbatch::Int=0,
)
    n = length(g)
    knns = zeros(IdWeight, k, n)
    allknn(g, ctx, knns; sort, progress)
end


function allknn(g::AbstractSearchIndex, ctx::AbstractContext, knns::AbstractMatrix;
    sort::Bool=true,
    progress=nothing,
    minbatch::Int=0
)
    m = length(g)  # don't use n from knns, use directly length(g), i.e., allows to reuse knns
    k, n = size(knns)
    @assert n > 0 "invalid assertion n > 0"
    @assert n == m "invalid assertion n == m"
    @assert 0 < k <= n
    minbatch = getminbatch(ctx, n)
    #progress = Progress(n, desc="allknn", dt=4, enabled=show_progress)
    let progress = progress
        #Threads.@threads :static for j in 1:minbatch:n
        @batch per=thread minbatch=4 for j in 1:minbatch:n
            m_ = min(m, j + minbatch - 1)
            res = knnqueue(ctx, view(knns, :, j))
            allknn_single_search!(g, ctx, j, res)
            sort && sortitems!(res)
            progress !== nothing && next!(progress)
            i = j + 1
            @inbounds while i <= m_
                reuse!(res, view(knns, :, i))
                allknn_single_search!(g, ctx, i, res)
                sort && sortitems!(res)
                progress !== nothing && next!(progress)
                i += 1
            end
        end
    end
    #=
        progress = Progress(n, desc="allknn", dt=4)
        @batch per = thread minbatch = minbatch for i in 1:n
            res = knnqueue(ctx, @view knns[:, i])
            res = allknn_single_search!(g, ctx, i, res)
            sort && sortitems!(res)
            next!(progress)
        end
    end=#

    knns
end

function allknn_single_search!(g::SearchGraph, ctx::SearchGraphContext, i::Integer, res)
    vstate = getvstate(length(g), ctx)
    q = database(g, i)
    # visit!(vstate, i)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)

    for h in neighbors(g.adj, i) # hints
        visited(vstate, convert(UInt64, h)) && continue
        search(g.algo[], g, ctx, q, res, h; vstate)
        # length(res) == k && break
    end

    res
end

function allknn_single_search!(g::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    search(g, ctx, database(g, i), res)
end
