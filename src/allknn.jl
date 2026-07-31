# This file is a part of SimilaritySearch.jl

export allknn
using ProgressMeter

"""
    allknn(index::AbstractSearchIndex, ctx::AbstractContext, k::Integer; sort::Bool=true, progress=Progress(length(index); desc="allknn", dt=4)) -> knns

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
- `knns`: a `(k, n)` matrix of `IdDist` elements (as created with `zeros(IdDist, k, n)`); the `i`-th column
  corresponds to the `i`-th object in the dataset. Trailing zeros at the end of a column mean that the
  retrieval found fewer than the desired `k` neighbors for that object.

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 8, 10^3))
G = SearchGraph(; dist=Dist.SqL2(), db=X)
ctx = getcontext(G)
index!(G, ctx)

knns = allknn(G, ctx, 8)  # (8, 10^3) matrix of `IdDist`
```
"""
function allknn(g::AbstractSearchIndex, ctx::AbstractContext, k::Integer;
    sort::Bool=true,
    progress=Progress(length(g); desc="allknn", dt=4)
)
    n = length(g)
    knns = zeros(IdDist, k, n)
    allknn(g, ctx, knns; sort, progress)
end

"""
    allknn(index::AbstractSearchIndex, ctx::AbstractContext, knns::AbstractMatrix; sort::Bool=true, progress=nothing) -> knns

In-place variant of [`allknn`](@ref) that receives a preallocated `(k, n)` matrix of `IdDist` elements
(e.g., `zeros(IdDist, k, n)`) as output, where `n == length(index)`. Useful to reuse memory across calls
or to resume/replace previously computed results.

# Arguments
- `index`: the index
- `ctx`: the index's context (caches, hyperparameters, logger, etc)
- `knns`: an output `(k, n)` matrix of `IdDist` elements, `n == length(index)`

# Keyword Arguments
- `sort`: ensures that each result set is presented in ascending order by distance
- `progress`: a `ProgressMeter.Progress` object used to report the progress of the computation, or `nothing` to disable it
"""
function allknn(g::AbstractSearchIndex, ctx::AbstractContext, knns::AbstractMatrix;
    sort::Bool=true,
    progress=nothing
)
    m = length(g)  # don't use n from knns, use directly length(g), i.e., allows to reuse knns
    k, n = size(knns)
    @assert n > 0 "invalid assertion n > 0"
    @assert n == m "invalid assertion n == m"
    @assert 0 < k <= n
    minbatch = getminbatch(n)
    #progress = Progress(n, desc="allknn", dt=4, enabled=show_progress)
    let progress = progress
        @BATCHES minbatch for j in 1:n
            res = knnqueue(ctx, view(knns, :, j))
            allknn_single_search!(g, ctx, j, res)
            sort && sortitems!(res)
            progress !== nothing && next!(progress)
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
        search(g.algo[], g, ctx, q, res, h, vstate)
        # length(res) == k && break
    end

    res
end

function allknn_single_search!(g::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    search(g, ctx, database(g, i), res)
end
