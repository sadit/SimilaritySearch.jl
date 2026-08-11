# This file is a part of SimilaritySearch.jl

export randsel

"""
    randsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; scheduler::Symbol=get_batch_scheduler())

Selects `k` centers randomly and computes the exact same properties as `fft` or `dnet` (such as `nn` and `dists`)
for the rest of the database `X` against these centers.

# Arguments
- `dist`: distance function
- `X`: the objects to be computed
- `k`: number of centers to be computed

# Keyword Arguments
- `scheduler`: the [`@BATCHES`](@ref) scheduler stored in the internal `GenericContext`
  used for this call (`:default`, `:static`, `:greedy`, or `:sequential` to disable
  threading entirely). Defaults to [`get_batch_scheduler`](@ref).

# Returns
A named tuple with the following fields:
- `centers`: the list of the selected centers (identifiers into ``X``)
- `nn`: the id of the nearest selected center of each object (in ``X`` order, identifiers between 1 and `length(X)`)
- `dists`: the distance from each object in the database to its nearest center (in ``X`` order)
- `costdists`: total number of distance evaluations performed by this call
- `costblocks`: always `0` for `randsel`
"""
function randsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    costdists = 0

    if N == 0
        return (; centers=UInt32[], nn=UInt32[], dists=Float32[], costdists=0, costblocks=0)
    end

    k = min(N, k)
    # randomly select k centers
    centers = UInt32.(shuffle(1:N)[1:k])

    # Create a subdatabase with the selected centers
    C = SubDatabase(X, centers)
    idx = ExhaustiveSearch(dist, C)
    ctx = GenericContext(; scheduler)
    
    # Find the nearest center for each object in X
    ids, dists = zeros(UInt32, 1, N), zeros(Float32, 1, N)
    searchbatch!(idx, ctx, X, ids, dists) 
    nn = centers[vec(ids)]
    dists = vec(dists)
    costdists = distance_evaluations(ctx)
    
    (; centers, nn, dists, costdists, costblocks=0)
end
