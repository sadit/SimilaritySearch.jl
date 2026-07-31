# This file is a part of SimilaritySearch.jl

export fft

"""
    fft(dist::SemiMetric, X::AbstractDatabase, k::Integer; start::Int=0, verbose::Bool=true, threads::Bool=true)

Selects `k` items that are far from each other based on the Farthest First Traversal (FFT) algorithm; this is
useful to obtain a diverse, representative subset of `X` (e.g., as candidate centers for clustering).
If `start=0` then a random starting point is selected, otherwise a valid object id of `X` should be given.

# Arguments
- `dist`: the distance function
- `X`: the input database
- `k`: the number of centers (far away items) to select

# Keyword Arguments
- `start`: the identifier of the first center; `0` means a random starting point is selected
- `verbose`: controls the verbosity of the function
- `threads`: whether to parallelize the nearest-center distance updates using multiple threads

# Returns

A named tuple with the following fields:
- `centers`: the list of the selected centers (identifiers into ``X``)
- `nn`: the id of the nearest selected center of each object (in ``X`` order, identifiers between 1 and `length(X)`)
- `dists`: the distance from each object in the database to its nearest center (in ``X`` order)
- `dmax`: the smallest distance among the (`k`) selected centers, i.e., the separation achieved by the traversal

Based on `enet.jl` from `KCenters.jl`

Note: `fft` is well-defined for metric distances

# Examples

```julia
using SimilaritySearch

dist = Dist.L2()
X = MatrixDatabase(rand(Float32, 4, 10^3))
R = fft(dist, X, 16)
R.centers   # 16 well-separated identifiers into X
R.nn        # nearest selected center for each object of X
R.dists     # distance to the nearest selected center
R.dmax      # separation radius achieved by the traversal
```
"""
function fft(dist::SemiMetric, X::AbstractDatabase, k::Integer; start::Int=0, verbose::Bool=true, threads::Bool=true)
    N = length(X)
    centers = UInt32[]
    sizehint!(centers, k)
    dmaxlist = Float32[]
    sizehint!(dmaxlist, k)
    nndists = Vector{Float32}(undef, N)
    fill!(nndists, typemax(Float32))
    nn = zeros(UInt32, N)
    imax::Int = start == 0 ? rand(1:N) : start
    dmax::Float32 = typemax(Float32)
    N == 0 && return (; centers, nn, dists=nndists, dmax)
    minbatch = getminbatch(N)

    @inbounds for _ in 1:k
        push!(dmaxlist, dmax)
        push!(centers, imax)
        verbose && println(stderr, "computing farthest point $(length(centers)), dmax: $dmax, imax: $imax, n: $(length(X))")

        pivot = X[imax]
        if threads
            minbatch=getminbatch(N)
            @BATCH minbatch=minbatch for i in 1:N
                d = evaluate(dist, X[i], pivot)
                if d < nndists[i]
                    nndists[i] = d
                    nn[i] = imax
                end
            end
        else
            for i in 1:N
                d = evaluate(dist, X[i], pivot)
                if d < nndists[i]
                    nndists[i] = d
                    nn[i] = imax
                end
            end
        end

        dmax, imax = findmax(nndists)
    end

    (; centers, nn, dists=nndists, dmax)
end
