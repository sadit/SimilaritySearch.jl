# This file is a part of SimilaritySearch.jl

export fft

"""
    fft(dist::SemiMetric, X::AbstractDatabase, k; start::Int=0, verbose=true, threads=true)

Selects `k` items far from each other based on Farthest First Traversal algorithm.
If `start=0` then a random starting point is selected a valid object id to `X` should be given otherwise.

Returns a named tuple with the following fields:
- `centers` contains the list of centers (indexes to ``X``)
- `nn` the id of the nearest center (in ``X`` order, identifiers between 1 to `length(X))
- `nndists` the distance from each object in the database to its nearest centers (in ``X`` order)
- `dmax` smallest distance among centers

Based on `enet.jl` from `KCenters.jl`

Note: `fft` is well-defined for metric distances
"""
function fft(dist::SemiMetric, X::AbstractDatabase, k::Integer; start::Int=0, verbose::Bool=true, threads::Bool=true)
    N = length(X)
    centers = Int32[]
    sizehint!(centers, k)
    dmaxlist = Float32[]
    sizehint!(dmaxlist, k)
    nndists = Vector{Float32}(undef, N)
    fill!(nndists, typemax(Float32))
    nn = zeros(UInt32, N)
    imax::Int = start == 0 ? rand(1:N) : start
    dmax::Float32 = typemax(Float32)
    N == 0 && return (; centers, nn, dists=nndists, dmax)
    minbatch = getminbatch(N, Threads.nthreads(), 0)

    @inbounds for _ in 1:k
        push!(dmaxlist, dmax)
        push!(centers, imax)
        verbose && println(stderr, "computing farthest point $(length(centers)), dmax: $dmax, imax: $imax, n: $(length(X))")

        pivot = X[imax]
        if threads
            #Threads.@threads :static for j in 1:minbatch:N
            @batch per=thread minbatch=4 for j in 1:minbatch:N
                for i in j:min(N, j + minbatch - 1)
                    d = evaluate(dist, X[i], pivot)
                    if d < nndists[i]
                        nndists[i] = d
                        nn[i] = imax
                    end
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
