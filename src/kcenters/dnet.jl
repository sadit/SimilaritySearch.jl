# This file is a part of SimilaritySearch.jl

import Random: shuffle!
export dnet

"""
    dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer; verbose::Bool=true, scheduler::Symbol=get_batch_scheduler())

Selects `numcenters` points far from each other based on density nets. It behaves similarly to `fft`,
returning a similar named tuple so they are interchangeable.

# Arguments
- `dist`: distance function
- `X`: the objects to be computed
- `numcenters`: number of centers to be computed

# Keyword Arguments
- `verbose`: controls the verbosity of the function
- `scheduler`: the [`@BATCHES`](@ref) scheduler stored in the internal `GenericContext`
  used for this call (`:default`, `:static`, `:greedy`, or `:sequential` to disable
  threading entirely). Defaults to [`get_batch_scheduler`](@ref).

# Returns
A named tuple with the following fields:
- `centers`: the list of the selected centers (identifiers into ``X``)
- `nn`: the id of the nearest selected center of each object (in ``X`` order, identifiers between 1 and `length(X)`)
- `dists`: the distance from each object in the database to its nearest center (in ``X`` order)
- `costdists`: total number of distance evaluations performed by this call
- `costblocks`: always `0` for `dnet`

Note: unlike `fft`/`multirandsel`, `dnet`'s selection isn't a greedy farthest-point traversal,
so there's no well-defined minimum-separation-among-centers quantity to report here -- no
`dmax` field is returned.
"""
function dnet(dist::SemiMetric, X::AbstractDatabase, numcenters::Integer; verbose::Bool=true, scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    centers = UInt32[]
    sizehint!(centers, numcenters)
    nndists = Vector{Float32}(undef, N)
    fill!(nndists, typemax(Float32))
    nn = zeros(UInt32, N)
    costdists = 0

    N == 0 && return (; centers, nn, dists=nndists, costdists=0, costblocks=0)

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

        for item in res
            orig_id = S.map[item.id]
            nn[orig_id] = c
            nndists[orig_id] = item.dist
        end

        verbose && println(stderr, "dnet -- selected-center: $(length(centers)), id: $c, dmax: $(maximum(res))")
        
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
    (; centers, nn, dists=nndists, costdists, costblocks=0)
end
