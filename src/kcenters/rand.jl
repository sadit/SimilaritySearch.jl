# This file is a part of SimilaritySearch.jl

export randsel

"""
    randsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; threads::Bool=true)

Selects `k` centers randomly and computes the exact same properties as `fft` or `dnet` (such as `nn` and `dists`) 
for the rest of the database `X` against these centers.

# Arguments
- `dist`: distance function
- `X`: the objects to be computed
- `k`: number of centers to be computed

# Keyword Arguments
- `threads`: whether to use multiple threads for context operations (if `true`, it uses an internal `GenericContext` allowing up to `Threads.nthreads()` batches)

# Returns
A named tuple with the following fields:
- `centers`: the list of the selected centers (identifiers into ``X``)
- `nn`: the id of the nearest selected center of each object (in ``X`` order, identifiers between 1 and `length(X)`)
- `dists`: the distance from each object in the database to its nearest center (in ``X`` order)
- `dmax`: the max distance evaluated to a center in the net
- `costdists`: total number of distance evaluations performed by this call
- `costblocks`: always `0` for `randsel`
"""
function randsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; threads::Bool=true)
    N = length(X)
    costdists = 0

    if N == 0
        return (; centers=UInt32[], nn=UInt32[], dists=Float32[], dmax=typemax(Float32), costdists=0, costblocks=0)
    end

    k = min(N, k)
    # randomly select k centers
    centers = UInt32.(shuffle(1:N)[1:k])
    
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
    
    costdists = distance_evaluations(ctx)
    
    (; centers, nn, dists=nndists, dmax, costdists, costblocks=0)
end
