# This file is a part of SimilaritySearch.jl

export fft

"""
    fft(dist::SemiMetric, X::AbstractDatabase, k; verbose=true)

Selects `k` items far from each other based on Farthest First Traversal algorithm.

Returns a named tuple with the following fields:
- `centers` contains the list of centers (indexes to ``X``)
- `nn` the id of the nearest center (in ``X`` order, identifiers between 1 to `length(X))
- `nndists` the distance from each object in the database to its nearest centers (in ``X`` order)
- `dmax` smallest distance among centers

Based on `enet.jl` from `KCenters.jl`
"""
function fft(dist::SemiMetric, X::AbstractDatabase, k::Integer; verbose=true)
    N = length(X)
    centers = Int32[]
    dmaxlist = Float32[]
    nndists = Vector{Float32}(undef, N)
    fill!(nndists, typemax(Float32))
    nn = zeros(UInt32, N) 
    imax::Int = rand(1:N)
    dmax::Float32 = typemax(Float32)
    N == 0 && return (; centers, nn, dists=nndists, dmax)
    
    @inbounds for i in 1:N
        push!(dmaxlist, dmax)
        push!(centers, imax)
        verbose && println(stderr, "computing fartest point $(length(centers)), dmax: $dmax, imax: $imax, n: $(length(X))")

        pivot = X[imax]
        @batch minbatch=getminbatch(0, N) for i in 1:N
            d = evaluate(dist, X[i], pivot)
            if d < nndists[i]
                nndists[i] = d
                nn[i] = imax
            end
        end

        dmax, imax = findmax(nndists)
        length(dmaxlist) < k || break
    end

    (; centers, nn, dists=nndists, dmax)
end

