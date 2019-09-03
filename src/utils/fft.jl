export kcenters

"""
    kcenters(dist::Function, X::AbstractVector{T}, k::Int) where T

Computes an approximation to ``k``-centers problem based on farthest first traversal.
Returns the ``(centers, epsilon)`` tuple where centers are the k-centers, 
being at-least ``\epsilon``-distant among each other.
"""
function kcenters(dist::Function, X::AbstractVector{T}, k::Int) where T
    N = length(X)
    centers = zeros(Int, N)
    epsilon::Float64 = typemax(Float64)

    if N == 0
       centers, epsilon
    end
    
    imax::Int = rand(1:N)
    Dmin = [epsilon for i in 1:N]
    Dtmp = Vector{Float64}(undef, N)

    for i in 1:k
        println(stderr, "computing fartest point $i of $k, epsilon: $epsilon, imax: $imax")
        centers[i] = imax

        Threads.@threads for j in 1:N
            @inbounds Dmin[j] = min(dist(X[j], X[imax]), Dmin[j])
        end

        epsilon, imax = findmax(Dmin)
    end

    centers, epsilon
end
