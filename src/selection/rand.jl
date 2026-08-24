# This file is a part of SimilaritySearch.jl

export randsel

"""
    randsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; scheduler::Symbol=get_batch_scheduler())

Selects `k` centers randomly and computes, for every object of `X`, the same properties
[`fft`](@ref) and [`dnet`](@ref) do -- returning the same [`CenterSelection`](@ref), so the four
selectors are interchangeable.

# Arguments
- `dist`: distance function
- `X`: the objects to be computed
- `k`: number of centers to be computed

# Keyword Arguments
- `scheduler`: the [`@BATCHES`](@ref) scheduler stored in the internal `GenericContext`
  used for this call (`:default`, `:static`, `:greedy`, or `:sequential` to disable
  threading entirely). Defaults to [`get_batch_scheduler`](@ref).

# Returns
A [`CenterSelection`](@ref). Its `separation` is measured over the selected centers
afterwards, and those `k(k-1)/2` evaluations are counted into `costdists`; a random selection
has no separation guarantee, which is precisely what the number reports.
"""
function randsel(dist::SemiMetric, X::AbstractDatabase, k::Integer; scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    costdists = 0

    N == 0 && return empty_selection()
    k >= 1 || throw(ArgumentError("randsel needs k >= 1, got $k"))
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
    # `searchbatch!` already answers in positions into `C`, which is exactly what `assign`
    # holds -- no translation back to identifiers into X, and none back again by the caller
    assign = vec(ids)
    assigndist = vec(dists)
    costdists = distance_evaluations(ctx)
    separation, seppairs = center_separation(dist, X, centers; scheduler)

    CenterSelection(centers, assign, assigndist, maximum(assigndist), separation,
                    costdists + seppairs, 0)
end
