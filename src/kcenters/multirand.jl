# This file is a part of SimilaritySearch.jl

export multirandsel

"""
    multirandsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; m::Int=ceil(Int, log2(length(X))), start::Int=0, threads::Bool=true)

Selects `k` centers iteratively. Starts with a random point (or `start` if `start > 0`). In each step, it selects `m` random 
candidates from `X` and adds the one that is farthest from the currently selected centers. Once `k` centers are selected, 
it computes the exact same properties as `fft` or `randsel` for the entire database.

# Arguments
- `dist`: distance function
- `X`: the objects to be computed
- `k`: number of centers to be computed

# Keyword Arguments
- `m`: number of candidates to evaluate per step (default is `ceil(Int, log2(length(X)))`)
- `start`: index of the first center. If 0, a random center is chosen.
- `threads`: whether to use multiple threads for context operations

# Returns
A named tuple with the following fields:
- `centers`: the list of the selected centers (identifiers into ``X``)
- `nn`: the id of the nearest selected center of each object (in ``X`` order, identifiers between 1 and `length(X)`)
- `dists`: the distance from each object in the database to its nearest center (in ``X`` order)
- `dmax`: the max distance evaluated to a center in the net
- `costdists`: total number of distance evaluations performed by this call
- `costblocks`: always `0` for `multirandsel`
"""
function multirandsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; m::Int=ceil(Int, log2(length(X))), start::Int=0, threads::Bool=true)
    N = length(X)
    costdists = 0

    if N == 0
        return (; centers=UInt32[], nn=UInt32[], dists=Float32[], dmax=typemax(Float32), costdists=0, costblocks=0)
    end

    k = min(N, k)
    centers = UInt32[]
    sizehint!(centers, k)

    first_center = start > 0 ? start : rand(1:N)
    push!(centers, first_center)

    candidates = Int[]
    sizehint!(candidates, m)

    # We will keep a local array to find minimum distance to centers for the m candidates
    while length(centers) < k
        empty!(candidates)
        while length(candidates) < m
            c = rand(1:N)
            # just avoiding exact duplicate indices
            if !(c in centers) && !(c in candidates)
                push!(candidates, c)
            end
        end

        best_cand = 0
        max_min_dist = -1f0

        for cand in candidates
            cand_obj = X[cand]
            min_dist = typemax(Float32)
            for center_id in centers
                d = evaluate(dist, cand_obj, X[center_id])
                costdists += 1
                if d < min_dist
                    min_dist = d
                end
            end

            if min_dist > max_min_dist
                max_min_dist = min_dist
                best_cand = cand
            end
        end

        push!(centers, best_cand)
    end

    nn = zeros(UInt32, N)
    nndists = fill(typemax(Float32), N)
    
    # Create a subdatabase with the selected centers
    C = SubDatabase(X, centers)
    idx = ExhaustiveSearch(dist, C)
    ctx = GenericContext(maxbatches = threads ? 8*Threads.nthreads() : 1)
    
    # Find the nearest center for each object in X
    knns = [knnqueue(KnnSorted, 1) for _ in 1:N]
    searchbatch!(idx, ctx, X, knns)
    
    dmax = 0f0
    for i in 1:N
        if length(knns[i]) > 0
            item = first(knns[i])
            nn[i] = centers[item.id]
            nndists[i] = item.dist
            dmax = max(dmax, item.dist)
        end
    end
    
    costdists += distance_evaluations(ctx)
    
    (; centers, nn, dists=nndists, dmax, costdists, costblocks=0)
end
