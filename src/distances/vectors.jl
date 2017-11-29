export sim_jaccard, sim_common_prefix, distance
export L1Distance, L2Distance, L2SquaredDistance, LInfDistance, LpDistance

""" L1Distance computes the Manhattan's distance """
mutable struct L1Distance
    calls::Int
    L1Distance() = new(0)
end

function (o::L1Distance)(a::Vector{T}, b::Vector{T})::Float64 where {T <: Real}
    o.calls += 1
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
	m = a[i] - b[i]
        d += ifelse(m > 0, m, -m)
    end

    return d
end

""" L2Distance computes the Euclidean's distance """
mutable struct L2Distance
    calls::Int
    L2Distance() = new(0)
end

function (o::L2Distance)(a::Vector{T}, b::Vector{T})::Float64 where {T <: Real}
    o.calls += 1
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
        m = a[i] - b[i]
        d += m * m
    end

    return sqrt(d)
end

""" L2SquaredDistance computes the Euclidean's distance but squared """
mutable struct L2SquaredDistance
    calls::Int
    L2SquaredDistance() = new(0)
end

function (o::L2SquaredDistance)(a::Vector{T}, b::Vector{T})::Float64 where {T <: Real}
    o.calls += 1
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
        m = a[i] - b[i]
        d += m * m
    end

    return d
end


""" LInfDistance computes the max distance """
mutable struct LInfDistance
    calls::Int
    LInfDistance() = new(0)
end

function (o::LInfDistance)(a::Vector{T}, b::Vector{T})::Float64 where {T <: Real}
    o.calls += 1
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
        m = abs(a[i] - b[i])
        d = max(d, m)
    end

    return d
end

"""
dist_lp computes a generic Minkowski's distance
"""
mutable struct LpDistance
    calls::Int
    p::Float32
    LpDistance(p::F) where {F <: Real} = new(0, convert(Float32, p))
end

function (o::LpDistance)(a::Vector{T}, b::Vector{T})::Float64 where {T <: Real}
    o.calls += 1
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
	m = abs(a[i] - b[i])
	d += m ^ o.p
    end

    return d ^ (1f0 / o.p)
end

