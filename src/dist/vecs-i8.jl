# This file is a part of SimilaritySearch.jl
export MAE


"""
    MAE()

"""
struct MAE <: Metric end

@inline function evaluate(::MAE, a, b)::Float32
    d = zero(Int8)

    @fastmath @inbounds @simd for i in eachindex(a, b)
        m = a[i] - b[i]
        m >>= 6
        m = m * m
        m >>= 4
        d += m
    end

    convert(Float32, d)
end
