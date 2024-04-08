# This file is a part of SimilaritySearch.jl
using LoopVectorization
export TurboL1Distance, TurboL2Distance, TurboSqL2Distance, TurboNormalizedCosineDistance

"""
    TurboL1Distance()

The `@turbo`ed implementation of [@ref](`L1Distance`) (see [@ref](`LoopVectorization`)'s macro)
"""
struct TurboL1Distance <: SemiMetric end

"""
    TurboL2Distance()

The `@turbo`ed implementation of [@ref](`L2Distance`) (see [@ref](`LoopVectorization`)'s macro)
"""
struct TurboL2Distance <: SemiMetric end

"""
    TurboSqL2Distance()

The `@turbo`ed implementation of [@ref](`SqL2Distance`) (see [@ref](`LoopVectorization`)'s macro)
"""
struct TurboSqL2Distance <: SemiMetric end

"""
    evaluate(::TurboL1Distance, a, b)

Computes the Manhattan's distance between `a` and `b`
"""
function evaluate(::TurboL1Distance, a, b)
    d = zero(eltype(a))

    @turbo thread=1 unroll=4 for i in eachindex(a, b)
	    m = a[i] - b[i]
        d += abs(m)
    end

    d
end

"""
    evaluate(::TurboL2Distance, a, b)

Computes the Euclidean's distance betweem `a` and `b`
"""
function evaluate(::TurboL2Distance, a, b)
    d = zero(eltype(a))

    @turbo thread=1 unroll=4 for i in eachindex(a)
        m = a[i] - b[i]
        d = muladd(m, m, d)
    end

    sqrt(d)
end

"""
    evaluate(::TurboSqL2Distance, a, b)

Computes the squared Euclidean's distance between `a` and `b`
"""
function evaluate(::TurboSqL2Distance, a, b)
    d = zero(eltype(a))

    @turbo thread=1 unroll=4 for i in eachindex(a)
        @inbounds m = a[i] - b[i]
        d = muladd(m, m, d)
    end

    d
end

"""
    TurboNormalizedCosineDistance()

The `@turbo`ed implementation of [@ref](`CosineDistance`) (see [@ref](`LoopVectorization`)'s macro)
"""
struct TurboNormalizedCosineDistance <: SemiMetric end
"""
    evaluate(::TurboNormalizedCosineDistance, a, b)

Computes the cosine distance between two vectors, it expects normalized vectors.
"""
function evaluate(::TurboNormalizedCosineDistance, a, b)
    T = typeof(a[1])
    s = zero(T)

    @turbo thread=1 unroll=4 for i in eachindex(a)
        s = muladd(a[i], b[i], s)
    end

    one(T) - s
end

