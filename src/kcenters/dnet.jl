# This file is a part of SimilaritySearch.jl

import Random: shuffle!
export dnet

"""
    dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer; verbose::Bool=true, reporters=InformativeLog(), scheduler::Symbol=get_batch_scheduler())

Selects points far from each other based on density nets, returning the same
[`CenterSelection`](@ref) every other selector here returns, so they are interchangeable.

`numcenters` is a **target, not a guarantee**: the algorithm carves the database into balls of
`length(X) ÷ numcenters` objects and keeps going until nothing is left, so the count it returns
is approximate and usually slightly larger (asking for 8 over 300 objects returns 9). Use
[`fft`](@ref), [`randsel`](@ref) or [`multirandsel`](@ref) when the count has to be exact.

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
A [`CenterSelection`](@ref). Unlike `fft`/`multirandsel`, `dnet` is not a greedy
farthest-point traversal, so its `separation` is not free: it is measured afterwards, over the
centers actually selected, and the `k(k-1)/2` evaluations that takes are counted into
`costdists` like any other.
"""
function dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer; verbose::Bool=true, reporters=InformativeLog(), scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    N == 0 && return empty_selection()
    numcenters >= 1 || throw(ArgumentError("dnet needs numcenters >= 1, got $numcenters"))

    centers = UInt32[]
    sizehint!(centers, numcenters)
    costdists = 0

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
        pos = length(centers)

        verbose && @inform reporters "dnet> center $pos, id: $c, ball-radius: $(maximum(res))"
        
        m = n - length(res)
        empty!(rlist)
        append!(rlist, IdView(res))
        sort!(rlist)
        numzeros = 0
        while length(rlist) > 0
            if rlist[end] > m
                S.map[rlist[end]] = 0
                pop!(rlist)
                numzeros += 1
            else
                break
            end
        end

        E = @view S.map[m+1:end]
        sort!(E)
        E = @view S.map[m+1+numzeros:end]
        if length(E) > 0
            for (i, e) in enumerate(E)
                S.map[rlist[i]] = e
            end
        end

        resize!(S.map, m)
    end
    
    costdists = distance_evaluations(ctx)

    # the balls above absorb objects as they are carved, so an object can end up in a ball
    # whose center is farther away than a center chosen in a later round -- for 40% of the
    # objects, measured. A final exact pass makes `assign` mean here what it means in every
    # other selector, the nearest center, for another `length(X) * k` evaluations: the same
    # order the carving itself already spent
    C = SubDatabase(X, centers)
    nearestidx = ExhaustiveSearch(dist, C)
    nearestctx = GenericContext(; scheduler)
    ids, dists = zeros(UInt32, 1, N), zeros(Float32, 1, N)
    searchbatch!(nearestidx, nearestctx, X, ids, dists)
    assign = vec(ids)
    assigndist = vec(dists)
    costdists += distance_evaluations(nearestctx)

    separation, seppairs = center_separation(dist, X, centers; scheduler)
    CenterSelection(centers, assign, assigndist, maximum(assigndist), separation,
                    costdists + seppairs, 0)
end
