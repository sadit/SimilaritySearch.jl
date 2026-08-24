# This file is a part of SimilaritySearch.jl

export multirandsel

# `log2(0)` is `-Inf` and `log2(1)` is `0`; neither survives `ceil(Int, ...)` as a block size,
# and a keyword's default is evaluated before the body that would have caught an empty database
_default_m(n::Integer) = n <= 2 ? 1 : ceil(Int, log2(n))

"""
    multirandsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; m::Int=_default_m(length(X)), start::Int=0, scheduler::Symbol=get_batch_scheduler())

Selects `k` centers iteratively. Starts with a random point (or `start` if `start > 0`). In each step, it selects `m` random
candidates from `X` and adds the one with the largest total distance to all currently selected centers (i.e. farthest from
all of them at once). Once `k` centers are selected, it computes for the entire database the same properties
[`fft`](@ref) and [`randsel`](@ref) do, returning the same [`CenterSelection`](@ref).

# Arguments
- `dist`: distance function
- `X`: the objects to be computed
- `k`: number of centers to be computed

# Keyword Arguments
- `m`: number of candidates to evaluate per step (default is `ceil(Int, log2(length(X)))`, and
  `1` for a database of two objects or fewer, where that formula gives `0` or `-Inf`);
  internally capped so at least `k - 1` rounds are always possible
- `start`: index of the first center. If 0, a random center is chosen.
- `scheduler`: the [`@BATCHES`](@ref) scheduler used for the per-step candidate evaluation
  and stored in the internal `GenericContext` used for the final nearest-center pass
  (`:default`, `:static`, `:greedy`, or `:sequential` to disable threading entirely).
  Defaults to [`get_batch_scheduler`](@ref).

# Returns
A [`CenterSelection`](@ref). Its `separation` costs nothing extra: each round already
evaluates every candidate against every selected center, so the winner's distances to all of
them are sitting in the matrix that chose it.
"""
function multirandsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; m::Int=_default_m(length(X)), start::Int=0, scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    costdists = 0

    N == 0 && return empty_selection()
    k >= 1 || throw(ArgumentError("multirandsel needs k >= 1, got $k"))
    k = min(N, k)

    # a single shuffle of 1:N drives both the first center and every later round's block of
    # candidates. `first_center` is removed from `perm` up front (swapped to the end for O(1)
    # removal when `start` names a specific element), so every later block -- a disjoint
    # chunk of the remaining, already-shuffled `perm` -- can never contain an
    # already-selected center; no per-candidate membership check against `centers` is needed
    perm = shuffle!(collect(1:N))
    first_center = if start > 0
        idx = findfirst(==(start), perm)
        perm[idx], perm[end] = perm[end], perm[idx]
        pop!(perm)
    else
        pop!(perm)
    end
    centers = UInt32[first_center]
    sizehint!(centers, k)
    separation = typemax(Float32)  # no pair of centers exists yet

    # bound `m` so partitioning the remaining `perm` always yields at least `k - 1` blocks --
    # one center is picked per block, and the pool is never reshuffled/topped up, so a
    # too-large `m` could otherwise run out of blocks before reaching `k` centers. A smaller
    # `m` only ever yields *more* blocks than needed, which is fine: the
    # `length(centers) >= k && break` check below still stops the loop at exactly `k`
    m = min(m, max(1, length(perm) ÷ max(k - 1, 1)))

    # guard the partition construction itself: m can be 0 when N == 1 (ceil(Int, log2(1))
    # == 0), which Iterators.partition rejects outright -- but then k <= 1 already and no
    # rounds are needed at all
    if length(centers) < k
        for block in Iterators.partition(perm, m)
            length(centers) >= k && break

            ncenters = length(centers)
            npairs = length(block) * ncenters

            # D[ci, j] = distance from block[j] to centers[ci], laid out (ncenters, |block|)
            # so consecutive linear pair indices `p` decode to consecutive `ci` for a fixed
            # `j`, matching Julia's column-major order. Every `p` maps to exactly one cell,
            # so cells are disjoint by construction: no @BEGIN/@END reduction needed, unlike
            # the old outer-parallel/serial-inner-loop version
            D = Matrix{Float32}(undef, ncenters, length(block))
            minbatch = getminbatch(npairs)
            @BATCHES minbatch scheduler=scheduler for p in 1:npairs
                j, ci = fldmod1(p, ncenters)
                D[ci, j] = evaluate(dist, X[block[j]], X[centers[ci]])
            end
            costdists += npairs  # every pair evaluated exactly once, unconditionally

            # per-candidate total distance to every center, then the candidate maximizing
            # that -- i.e. farthest from all of them at once, not just from its nearest one
            # -- O(m * ncenters) float additions on an already-filled matrix, negligible
            # next to the npairs distance evaluations above, so this stays a plain serial
            # reduction (not parallelized)
            sumvals = vec(sum(D; dims=1))
            bi = argmax(sumvals)
            best_cand = block[bi]

            # the new center's distances to every previously selected center are already
            # sitting in D's `bi` column -- folding their minimum into a running minimum
            # gives the exact smallest distance between any two final centers, at zero extra
            # distance evaluations
            separation = min(separation, minimum(view(D, :, bi)))

            push!(centers, best_cand)
        end
    end

    assign = zeros(UInt32, N)
    nndists = fill(typemax(Float32), N)
    
    # Create a subdatabase with the selected centers
    C = SubDatabase(X, centers)
    idx = ExhaustiveSearch(dist, C)
    ctx = GenericContext(; scheduler)
    
    # Find the nearest center for each object in X
    knns = [knnqueue(KnnSorted, 1) for _ in 1:N]
    searchbatch!(idx, ctx, X, knns)
    
    # `item.id` is already a position into `centers`, which is what `assign` holds
    for i in 1:N
        if length(knns[i]) > 0
            item = first(knns[i])
            assign[i] = item.id
            nndists[i] = item.dist
        end
    end

    costdists += distance_evaluations(ctx)

    CenterSelection(centers, assign, nndists, maximum(nndists), separation, costdists, 0)
end
