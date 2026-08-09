# This file is a part of SimilaritySearch.jl

export allknn, allknn!
using ProgressMeter

"""
    allknn(index::AbstractSearchIndex, ctx::AbstractContext, k::Integer; sort::Bool=true, progress=Progress(length(index); desc="allknn", dt=4)) -> (ids, dists)

Computes all the k nearest neighbors (all vs all) using the given index. Note that each object is its own
nearest neighbor, so the user is responsible for removing these self references from the output if needed.

# Arguments
- `index`: the index
- `ctx`: the index's context (caches, hyperparameters, logger, etc)
- `k`: the number of neighbors to retrieve for each object indexed by `index`

# Keyword Arguments
- `sort`: ensures that each result set is presented in ascending order by distance
- `progress`: a `ProgressMeter.Progress` object used to report the progress of the computation, or `nothing` to disable it

# Returns
A tuple `(ids, dists)` where both are `(k, n)` matrices. The `i`-th column corresponds to
the `i`-th object in the dataset. Trailing zeros in `ids` (and `Inf32` in `dists`) mean
that the retrieval found fewer than the desired `k` neighbors for that object.

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 8, 10^3))
G = SearchGraph(; dist=Dist.SqL2(), db=X)
ctx = SearchGraphContext()
index!(G, ctx)

ids, dists = allknn(G, ctx, 8)  # both are (8, 10^3) matrices
```
"""
function allknn(g::AbstractSearchIndex, ctx::AbstractContext, k::Integer;
    sort::Bool=true,
    progress=Progress(length(g); desc="allknn", dt=4)
)
    n = length(g)
    ids   = zeros(UInt32,  k, n)
    dists = fill(typemax(Float32), k, n)
    allknn!(g, ctx, ids, dists; sort, progress)
end

"""
    allknn!(index::AbstractSearchIndex, ctx::AbstractContext, ids, dists; sort::Bool=true, progress=nothing) -> (ids, dists)

In-place variant of [`allknn`](@ref) that receives preallocated `(k, n)` matrices as output,
where `n == length(index)`. Useful to reuse memory across calls or to resume/replace previously
computed results.

# Arguments
- `index`: the index
- `ctx`: the index's context (caches, hyperparameters, logger, etc)
- `ids`: an output `(k, n)` matrix of `UInt32`, `n == length(index)`
- `dists`: an output `(k, n)` matrix of `Float32`, parallel to `ids`

# Keyword Arguments
- `sort`: ensures that each result set is presented in ascending order by distance
- `progress`: a `ProgressMeter.Progress` object used to report the progress of the computation, or `nothing` to disable it
"""
function allknn!(g::AbstractSearchIndex, ctx::AbstractContext,
                 ids::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32};
    sort::Bool=true,
    progress=nothing
)
    m = length(g)  # don't use n from knns, use directly length(g), i.e., allows to reuse knns
    k, n = size(ids)
    @assert n > 0 "invalid assertion n > 0"
    @assert n == m "invalid assertion n == m"
    @assert 0 < k <= n
    minbatch = getminbatch(ctx, n)
    let progress = progress
        @BATCHES minbatch begin
        @BEGINBATCH
            bctx = @set ctx.batchid = @batchid()
        @LOOP for j in 1:n
            res = knnqueue(bctx, view(ids, :, j), view(dists, :, j))
            allknn_single_search!(g, bctx, j, res)
            sort && sortitems!(res)
            progress !== nothing && next!(progress)
        end
        end
    end

    ids, dists
end

function allknn_single_search!(g::SearchGraph, ctx::SearchGraphContext, i::Integer, res)
    vstate = getvstate(length(g), ctx)
    q = database(g, i)
    # visit!(vstate, i)
    # the loop helps to overcome when the current nn is in a small clique (smaller the the desired k)

    for h in neighbors(g.adj, i) # hints
        visited(vstate, convert(UInt64, h)) && continue
        search(g.algo[], g, ctx, q, res, h, vstate)
        # length(res) == k && break
    end

    res
end

function allknn_single_search!(g::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    search(g, ctx, database(g, i), res)
end
