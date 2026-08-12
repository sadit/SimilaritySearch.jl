# This file is a part of SimilaritySearch.jl

"""
    bichromatic_closestpair(idxA::AbstractSearchIndex, ctxA::AbstractContext,
                             idxB::AbstractSearchIndex, ctxB::AbstractContext; min_k::Int=8) -> (i, j, dist)

Finds the closest pair `(a, b)` with `a` an identifier of `idxA` and `b` an identifier of `idxB`, i.e.,
the closest pair between the two (already built) datasets indexed by `idxA` and `idxB`. If either index
is approximate then the resulting pair may also be an approximation of the true closest pair.

If `database(idxA) === database(idxB)` (the two indices share the very same underlying database --
including the case where `idxA` and `idxB` are literally the same index, as [`closestpair`](@ref) uses),
self-matches (an element paired with itself) are excluded from the result; otherwise `idxA` and `idxB`
are assumed to index disjoint datasets and every candidate pair is eligible.

This first implementation always iterates over `idxA`'s elements querying into `idxB`; it does not (yet)
pick the smaller side to minimize the number of queries.

Dispatches to a parallel or a sequential implementation depending on `Threads.nthreads()`.

# Arguments
- `idxA`, `idxB`: the search structures that index the two sets of points
- `ctxA`, `ctxB`: their respective search contexts (caches, hyperparameters, etc)

# Keyword Arguments
- `min_k`: instead of looking for `k=1` some approximate methods can take advantage of a larger `k`

# Returns
A tuple `(i, j, dist)` with the identifier `i` of `idxA`, the identifier `j` of `idxB`, and their
distance `dist`.

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
A = MatrixDatabase(rand(Float32, 2, 10^3))
B = MatrixDatabase(rand(Float32, 2, 10^3))
GA, GB = SearchGraph(dist, A), SearchGraph(dist, B)
ctxA, ctxB = SearchGraphContext(), SearchGraphContext()
index!(GA, ctxA); index!(GB, ctxB)

i, j, d = bichromatic_closestpair(GA, ctxA, GB, ctxB)
```
"""
function bichromatic_closestpair(idxA::AbstractSearchIndex, ctxA::AbstractContext, idxB::AbstractSearchIndex, ctxB::AbstractContext; min_k::Int=8)
    if Threads.nthreads() == 1
        sequential_bichromatic_closestpair(idxA, ctxA, idxB, ctxB, min_k)
    else
        parallel_bichromatic_closestpair(idxA, ctxA, idxB, ctxB, min_k)
    end
end

"""
    bichromatic_search_hint(idxA, i, idxB, ctxB, res, exclude_self::Bool)

Queries `idxB` (via `ctxB`) with the `i`-th element of `idxA`, reusing `res` as the output buffer.
`exclude_self` controls whether a match with the very same identifier `i` is dropped -- needed when
`idxA` and `idxB` share their underlying database (see [`bichromatic_closestpair`](@ref)).
"""
function bichromatic_search_hint(idxA::AbstractSearchIndex, i::Integer, idxB::AbstractSearchIndex, ctxB::AbstractContext, res, exclude_self::Bool)
    res = search(idxB, ctxB, database(idxA, i), res)
    if exclude_self && argmin(res) == i
        pop_min!(res)
    end

    nearest(res)
end

function sequential_bichromatic_closestpair(idxA::AbstractSearchIndex, ctxA::AbstractContext, idxB::AbstractSearchIndex, ctxB::AbstractContext, min_k)::Tuple{Int32,Int32,Float32}
    sameidx = idxA === idxB
    samedata = database(idxA) === database(idxB)
    mindist = typemax(Float32)
    I = J = zero(Int32)
    res = knnqueue(KnnSorted, min_k) # requires KnnSorted to support pop_min!

    for i in eachindex(idxA)
        p = sameidx ? search_hint(idxA, ctxA, i, reuse!(res)) : bichromatic_search_hint(idxA, i, idxB, ctxB, reuse!(res), samedata)
        if p.dist < mindist
            I, J, mindist = Int32(i), p.id, p.dist
        end
    end

    (I, J, mindist)
end

function parallel_bichromatic_closestpair(idxA::AbstractSearchIndex, ctxA::AbstractContext, idxB::AbstractSearchIndex, ctxB::AbstractContext, min_k)::Tuple{Int32,Int32,Float32}
    sameidx = idxA === idxB
    samedata = database(idxA) === database(idxB)
    n = length(idxA)
    minbatch = getminbatch(ctxA, n)
    local best

    @BATCHES minbatch scheduler=ctxA.scheduler begin
    @BEGIN
        # one column/slot per batch -- @batchid()-indexed, so race-free regardless of scheduler
        knns_ids = zeros(UInt32, min_k, @nbatches())
        knns_dists = zeros(Float32, min_k, @nbatches())
        B = Vector{Tuple{Int32,Int32,Float32}}(undef, @nbatches())
    @BEGINBATCH
        bctxA = @set ctxA.batchid = @batchid()
        bctxB = sameidx ? bctxA : (@set ctxB.batchid = @batchid())
        r = knnqueue(KnnSorted, view(knns_ids, :, @batchid()), view(knns_dists, :, @batchid())) # requires KnnSorted to support pop_min!
        b = (zero(Int32), zero(Int32), typemax(Float32))
    @LOOP for objID in 1:n
        p = sameidx ? search_hint(idxA, bctxA, objID, reuse!(r)) : bichromatic_search_hint(idxA, objID, idxB, bctxB, reuse!(r), samedata)
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
