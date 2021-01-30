# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export L1Distance, L2Distance, SqL2Distance, LInftyDistance, LpDistance

import Distances: evaluate

struct L1Distance <: PreMetric end
struct L2Distance <: PreMetric end
struct SqL2Distance <: PreMetric end
struct LInftyDistance <: PreMetric end

struct LpDistance <: PreMetric
    p::Float32
    pinv::Float32
end

LpDistance(p::Real) = LpDistance(p, 1/p)

StructTypes.StructType(::Type{L1Distance}) = StructTypes.Struct()
StructTypes.StructType(::Type{L2Distance}) = StructTypes.Struct()
StructTypes.StructType(::Type{SqL2Distance}) = StructTypes.Struct()
StructTypes.StructType(::Type{LInftyDistance}) = StructTypes.Struct()
StructTypes.StructType(::Type{LpDistance}) = StructTypes.Struct()


"""
    evaluate(L1Distance, a, b)

Computes the Manhattan's distance between `a` and `b`
"""
function evaluate(::L1Distance, a, b)
    d = zero(eltype(a))

    @inbounds @simd for i = 1:length(a)
	    m = a[i] - b[i]
        d += ifelse(m > 0, m, -m)
    end

    d
end

"""
    evaluate(L2Distance, a, b)
    
Computes the Euclidean's distance betweem `a` and `b`
"""
function evaluate(::L2Distance, a, b)
    d = zero(eltype(a))

    @simd for i = 1:length(a)
        #m = a[i] - b[i]
        @inbounds d += (a[i] - b[i])^2 #m * m
    end

    sqrt(d)
end

"""
    evaluate(::SqL2Distance, a, b)

Computes the squared Euclidean's distance between `a` and `b`
"""
function evaluate(::SqL2Distance, a, b)
    d = zero(eltype(a))

    @simd for i in eachindex(a)
        @inbounds d += (a[i] - b[i])^2
        # d += m * m
    end

    d
end


"""
    evaluate(::LInftyDistance, a, b)

Computes the max or Chebyshev'se distance
"""
@inline function evaluate(::LInftyDistance, a, b)
    d = zero(eltype(a))

    @inbounds @simd for i in eachindex(a)
        m = abs(a[i] - b[i])
        d = max(d, m)
    end

    d
end

"""
    evaluate(lp::LpDistance, a, b)

Computes generic Minkowski's distance
"""
function evaluate(lp::LpDistance, a, b)
    d = zero(eltype(a))

    @inbounds @simd for i in eachindex(a)
        m = abs(a[i] - b[i])
        d += m ^ lp.p
    end

    d ^ lp.pinv
end
