# This file is a part of SimilaritySearch.jl

export closestpair

"""
    closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; min_k::Int=8) -> (i, j, dist)

Finds the closest pair among all elements indexed by `idx`. If `idx` is an approximate index then the
resulting pair may also be an approximation of the true closest pair. Dispatches to a parallel or a
sequential implementation depending on `Threads.nthreads()`.

# Arguments
- `idx`: the search structure that indexes the set of points
- `ctx`: the search context (caches, hyperparameters, etc)

# Keyword Arguments
- `min_k`: instead of looking for `k=1` some approximate methods can take advantage of a larger `k`

# Returns
A tuple `(i, j, dist)` with the identifiers `i` and `j` of the closest pair found and their distance `dist`.

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 2, 10^3))
G = SearchGraph(; dist, db=X)
ctx = getcontext(G)
index!(G, ctx)

i, j, d = closestpair(G, ctx)
```
"""
function closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; min_k::Int=8)
    if Threads.nthreads() == 1
        sequential_closestpair(idx, ctx, min_k)
    else
        parallel_closestpair(idx, ctx, min_k)
    end
end

function search_hint(idx::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    res = search(idx, ctx, database(idx, i), res)
    if argmin(res) == i
        pop_min!(res)
    end

    nearest(res)
end

function search_hint(G::SearchGraph, ctx::SearchGraphContext, i::Integer, res)
    vstate = getvstate(length(G), ctx)
    visit!(vstate, convert(UInt64, i))
    res = search(G.algo[], G, ctx, database(G, i), res, rand(neighbors(G.adj, i)), vstate)
    if argmin(res) == i
        pop_min!(res)
    end
    
    nearest(res)
end

function parallel_closestpair(idx::AbstractSearchIndex, ctx::AbstractContext, min_k)::Tuple{Int32,Int32,Float32}
    n = length(idx)
    minbatch = getminbatch(ctx, n)
    local best

    @BATCHES minbatch begin
    @BEGIN
        # one column/slot per batch -- @batchid()-indexed, so race-free regardless of scheduler
        knns_ids = zeros(UInt32, min_k, @nbatches())
        knns_dists = zeros(Float32, min_k, @nbatches())
        B = Vector{Tuple{Int32,Int32,Float32}}(undef, @nbatches())
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
        r = knnqueue(KnnSorted, view(knns_ids, :, @batchid()), view(knns_dists, :, @batchid())) # requires KnnSorted to support pop_min!
        b = (zero(Int32), zero(Int32), typemax(Float32))
    @LOOP for objID in 1:n
        reuse!(r)
        p = search_hint(idx, bctx, objID, r)
        if p.dist < last(b)
            b = (Int32(objID), p.id, p.dist)
        end
    end
    @ENDBATCH
        B[@batchid()] = b
    @END
        _, i = findmin(last, B)
        best = B[i]
    end

    best
end

function sequential_closestpair(idx::AbstractSearchIndex, ctx::AbstractContext, min_k)::Tuple{Int32,Int32,Float32}
    mindist = typemax(Float32)
    I = J = zero(Int32)
    res = knnqueue(KnnSorted, min_k) # requires KnnSorted to support pop_min!
    for i in eachindex(idx)
        p = search_hint(idx, ctx, i, reuse!(res))
        if p.dist < mindist
            I, J, mindist = Int32(i), p.id, p.dist
        end
    end

    (I, J, mindist)
end
