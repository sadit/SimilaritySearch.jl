# This file is a part of SimilaritySearch.jl

import Random: shuffle!
export dnet

"""
    dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer; verbose::Bool=true, reporters=InformativeLog(), scheduler::Symbol=get_batch_scheduler())

Selects one representative per density-based ball, returning the same
[`CenterSelection`](@ref) every other selector here returns, so they are interchangeable.

Each round takes a random surviving object, gives it the `k` nearest objects still in the pool,
and removes all of them. So a center is not chosen to be far from the previous ones -- it is
simply whatever survived outside the balls carved so far, which is enough to spread the centers
out without any farthest-point search (`separation` reports how far, and it lands well above
`randsel`'s in practice).

`numcenters` is a **target, not the count**: the algorithm carves the database into balls of
`k = max(1, length(X) ÷ numcenters)` objects each and keeps going until nothing is left, so it
returns exactly `cld(length(X), k)` centers -- `numcenters` itself when `k` divides evenly, and
one more when it does not (asking for 8 over 300 objects returns 9). Use [`fft`](@ref),
[`randsel`](@ref) or [`multirandsel`](@ref) when the count has to be exact.

!!! warning "`assign` here is not the nearest center"
    Every other selector reports, for each object, the center *closest* to it. `dnet` reports
    the center **whose ball absorbed it**, which is what the carving actually computed: an
    object leaves the pool with the ball that took it, and a center chosen in a later round can
    turn out to be closer. Measured on 120 random points in 4 dimensions with `numcenters=10`,
    that happens for 40% of the objects. Consequently `covering` here is an upper bound on the
    true covering radius rather than the radius itself. A caller who needs a nearest-center
    assignment can compute one from `centers` -- this function deliberately does not pay for it.

    The centers themselves are unaffected: they are a valid ball cover either way.

# Arguments
- `dist`: distance function
- `X`: the objects to be computed
- `numcenters`: number of centers to be computed

# Keyword Arguments
- `verbose`: whether the per-center progress message is produced at all
- `reporters`: where that message goes, see [`AbstractReporter`](@ref). `dnet` takes no context, so a
  caller that has one should pass `reporters=ctx.reporters` for its silencing to reach here; pass
  `reporters=[]` to silence it directly.
- `scheduler`: the [`@BATCHES`](@ref) scheduler stored in the internal `GenericContext`
  used for this call (`:default`, `:static`, `:greedy`, or `:sequential` to disable
  threading entirely). Defaults to [`get_batch_scheduler`](@ref).

# Returns
A [`CenterSelection`](@ref), with the `assign` caveat above. Unlike `fft`/`multirandsel`,
`dnet` is not a greedy farthest-point traversal, so its `separation` is not free: it is measured
afterwards, over the centers actually selected, and the `k(k-1)/2` evaluations that takes are
counted into `costdists` like any other.
"""
function dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer; verbose::Bool=true, reporters=InformativeLog(), scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    N == 0 && return empty_selection()
    numcenters >= 1 || throw(ArgumentError("dnet needs numcenters >= 1, got $numcenters"))

    centers = UInt32[]
    sizehint!(centers, numcenters)
    assign = zeros(UInt32, N)
    assigndist = fill(typemax(Float32), N)

    k = N ÷ numcenters
    k == 0 && (k = 1)

    S = SubDatabase(X, shuffle!(collect(1:N)))
    I = ExhaustiveSearch(dist, S)
    ctx = GenericContext(; scheduler)
    
    res = knnqueue(KnnSorted, k)
    rlist = Int32[]
    
    while length(S.map) > 0
        n = length(S.map)
        search(I, ctx, I[n], reuse!(res, k))
        
        c = S.map[n]
        push!(centers, c)
        pos = UInt32(length(centers))

        # each member of this ball is recorded against the center that took it, which is not
        # necessarily the nearest one -- see this function's docstring, and `CenterSelection`
        for item in res
            orig_id = S.map[item.id]
            assign[orig_id] = pos
            assigndist[orig_id] = item.dist
        end

        verbose && @inform reporters "dnet> center $pos, id: $c, ball-radius: $(maximum(res))"
        
        # drop this ball's members from the candidate pool. `res` names positions in `S`, and
        # removing them from the highest down makes each one a swap with the current last
        # element -- O(1) apiece, and it leaves the survivors in the shuffled order that the
        # choice of center depends on. Sorting the tail instead, as this did before, walked
        # large identifiers toward the end of `S.map`, which is exactly where the next center
        # is taken from: over 600 objects the selected centers averaged id 424 instead of 300
        empty!(rlist)
        append!(rlist, IdView(res))
        sort!(rlist)
        for i in length(rlist):-1:1
            p = rlist[i]
            S.map[p] = S.map[end]
            pop!(S.map)
        end
    end

    costdists = distance_evaluations(ctx)
    separation, seppairs = center_separation(dist, X, centers; scheduler)
    CenterSelection(centers, assign, assigndist, maximum(assigndist), separation,
                    costdists + seppairs, 0)
end
