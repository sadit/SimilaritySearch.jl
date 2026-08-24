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

A [`CenterSelection`](@ref), the same type every other selector in `KCenters` returns.
`k` is clamped to `length(X)`: asking for more centers than there are objects used to return
the same object several times.

Both radii are exact and free here. The traversal picks each new center at the distance that
separates it from everything selected so far, and that distance decreases monotonically, so
the last one is the `separation`; what remains afterwards, the distance from the farthest
object to its nearest center, is the `covering`.

Based on `enet.jl` from `KCenters.jl`

Note: `fft` is well-defined for metric distances

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 10^3))
R = fft(dist, X, 16)
R.centers               # 16 well-separated identifiers into X
R.assign                # position in R.centers of each object's nearest center
R.centers[R.assign[7]]  # ... as an identifier into X
R.assigndist            # distance to that center
R.covering, R.separation
R.costdists             # distance evaluations performed by this call
```
"""
function fft(dist::SemiMetric, X::AbstractDatabase, k::Integer; start::Int=0, verbose::Bool=true, reporters=InformativeLog(), scheduler::Symbol=get_batch_scheduler())
    N = length(X)
    N == 0 && return empty_selection()
    k >= 1 || throw(ArgumentError("fft needs k >= 1, got $k"))
    k = min(k, N)

    centers = UInt32[]
    sizehint!(centers, k)
    nndists = fill(typemax(Float32), N)
    assign = zeros(UInt32, N)
    imax::Int = start == 0 ? rand(1:N) : start
    ε::Float32 = typemax(Float32)
    separation::Float32 = typemax(Float32)
    costdists = 0
    minbatch = getminbatch(N)

    @inbounds for _ in 1:k
        # `ε` is how far this new center is from everything selected so far, and it only
        # ever shrinks -- so the value carried into the last round is the smallest gap
        # between any two of the centers this call ends up returning
        separation = ε
        push!(centers, imax)
        pos = UInt32(length(centers))
        verbose && @inform reporters "fft> farthest point $pos, ε: $ε, imax: $imax, n: $N"

        pivot = X[imax]
        @BATCHES minbatch scheduler=scheduler for i in 1:N
            d = evaluate(dist, X[i], pivot)
            if d < nndists[i]
                nndists[i] = d
                assign[i] = pos
            end
        end
        costdists += N

        # and whatever is farthest from the selection now is what the selection fails to
        # cover -- the covering radius, which is also the next center were one requested
        ε, imax = findmax(nndists)
    end

    CenterSelection(centers, assign, nndists, ε, separation, costdists, 0)
end
