export l1_distance, l2_distance, squared_l2_distance, linf_distance, lp_distance

""" l1_distance computes the Manhattan's distance """
function l1_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
	    m = a[i] - b[i]
        d += ifelse(m > 0, m, -m)
    end

    return d
end

""" l2_distance computes the Euclidean's distance """
function l2_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
        m = a[i] - b[i]
        d += m * m
    end

    return sqrt(d)
end

""" squared_l2_distance computes the Euclidean's distance but squared """
function squared_l2_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
        m = a[i] - b[i]
        d += m * m
    end

    return d
end


""" linf_distance computes the max distance """
function linf_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Real}
   d::T = zero(T)

    @fastmath @inbounds @simd for i = 1:length(a)
        m = abs(a[i] - b[i])
        d = max(d, m)
    end

    return d
end

"""
lp_distance creates a function that computes computes generic Minkowski's distance
"""
function lp_distance(p_::Real)
    p::Float64 = convert(Float64, p_)
    invp = 1.0 / p

    function _lp(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where T
        d::T = zero(T)

        @fastmath @inbounds @simd for i = 1:length(a)
            m = abs(a[i] - b[i])
            d += m ^ p
        end

        d ^ invp
    end
end

