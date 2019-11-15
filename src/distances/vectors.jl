# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export l1_distance, l2_distance, squared_l2_distance, linf_distance, lp_distance

"""
    l1_distance(a, b)::Float64

Computes the Manhattan's distance between `a` and `b`
"""
function l1_distance(a, b)::Float64
    d::Float64 = 0.0 #zero(eltype(a))

    @inbounds @simd for i = 1:length(a)
	    m = a[i] - b[i]
        d += ifelse(m > 0, m, -m)
    end

    d
end

"""
    l2_distance(a, b)::Float64
    
Computes the Euclidean's distance betweem `a` and `b`
"""
function l2_distance(a, b)::Float64
    #d = zero(eltype(a))
    d::Float64 = 0.0

    @inbounds @simd for i = 1:length(a)
        m = a[i] - b[i]
        d += m * m
    end

    sqrt(d)
end

"""
    squared_l2_distance(a, b)::Float64

Computes the squared Euclidean's distance between `a` and `b`
"""
function squared_l2_distance(a, b)::Float64
    # d = zero(eltype(a))
    d::Float64 = 0.0

    @inbounds @simd for i = 1:length(a)
        m = a[i] - b[i]
        d += m * m
    end

    d
end


"""
    linf_distance(a, b)::Float64

Computes the max or Chebyshev'se distance
"""
function linf_distance(a, b)::Float64
   d::Float64 = 0.0 # d = zero(eltype(a))

    @inbounds @simd for i = 1:length(a)
        m = abs(a[i] - b[i])
        d = max(d, m)
    end

    d
end

"""
    lp_distance(p_::Real)

Creates a function that computes computes generic Minkowski's distance with the given `p_`
"""
function lp_distance(p::Real)
    p = convert(Float64, p)
    invp = 1.0 / p

    function _lp(a, b)::Float64
        d::Float64 = 0.0 # d = zero(eltype(a))

        @inbounds @simd for i = 1:length(a)
            m = abs(a[i] - b[i])
            d += m ^ p
        end

        d ^ invp
    end
end
