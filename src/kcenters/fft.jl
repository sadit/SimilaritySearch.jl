# This file is a part of SimilaritySearch.jl

export fft

"""
    fft(dist::SemiMetric, X::AbstractDatabase, k::Integer; start::Int=0, verbose::Bool=true, reporters=InformativeLog(), scheduler::Symbol=get_batch_scheduler())

Selects `k` items that are far from each other based on the Farthest First Traversal (FFT) algorithm; this is
useful to obtain a diverse, representative subset of `X` (e.g., as candidate centers for clustering).
If `start=0` then a random starting point is selected, otherwise a valid object id of `X` should be given.

# Arguments
- `dist`: the distance function
- `X`: the input database
- `k`: the number of centers (far away items) to select

# Keyword Arguments
- `start`: the identifier of the first center; `0` means a random starting point is selected
- `verbose`: whether the per-center progress message is produced at all
- `reporters`: where that message goes, see [`AbstractReporter`](@ref). `fft` takes no context, so a
  caller that has one should pass `reporters=ctx.reporters` for its silencing to reach here; pass
  `reporters=[]` to silence it directly.
- `scheduler`: the [`@BATCHES`](@ref) scheduler used for the per-pivot distance update
  (`:default`, `:static`, `:greedy`, or `:sequential` to disable threading entirely).
  Defaults to [`get_batch_scheduler`](@ref).

# Returns

A named tuple with the following fields:
- `centers`: the list of the selected centers (identifiers into ``X``)
- `nn`: the id of the nearest selected center of each object (in ``X`` order, identifiers between 1 and `length(X)`)
- `dists`: the distance from each object in the database to its nearest center (in ``X`` order)
- `ε`: the smallest distance among the (`k`) selected centers, i.e., the separation achieved by the traversal
- `costdists`: total number of distance evaluations performed by this call (`k * length(X)`), counted locally (no `ctx` involved)
- `costblocks`: always `0` for `fft` (no block-evaluation concept applies here)

Based on `enet.jl` from `KCenters.jl`

Note: `fft` is well-defined for metric distances

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 10^3))
R = fft(dist, X, 16)
R.centers      # 16 well-separated identifiers into X
R.nn           # nearest selected center for each object of X
R.dists        # distance to the nearest selected center
R.ε            # separation radius achieved by the traversal
R.costdists    # distance evaluations performed by this call
```
"""
function fft(dist::SemiMetric, X::AbstractDatabase, k::Integer; start::Int=0, verbose::Bool=true, reporters=InformativeLog(), scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    centers = UInt32[]
    sizehint!(centers, k)
    εlist = Float32[]
    sizehint!(εlist, k)
    nndists = Vector{Float32}(undef, N)
    fill!(nndists, typemax(Float32))
    nn = zeros(UInt32, N)
    imax::Int = start == 0 ? rand(1:N) : start
    ε::Float32 = typemax(Float32)
    N == 0 && return (; centers, nn, dists=nndists, ε, costdists=0, costblocks=0)
    costdists = 0
    minbatch = getminbatch(N)

    @inbounds for _ in 1:k
        push!(εlist, ε)
        push!(centers, imax)
        verbose && @inform reporters "fft> farthest point $(length(centers)), ε: $ε, imax: $imax, n: $(length(X))"

        pivot = X[imax]
        @BATCHES minbatch scheduler=scheduler for i in 1:N
            d = evaluate(dist, X[i], pivot)
            if d < nndists[i]
                nndists[i] = d
                nn[i] = imax
            end
        end
        costdists += N

        ε, imax = findmax(nndists)
    end

    (; centers, nn, dists=nndists, ε, costdists=costdists, costblocks=0)
end
