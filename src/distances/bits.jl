# This file is a part of SimilaritySearch.jl

export BinaryHammingDistance

"""
   BinaryHammingDistance()
   
Binary hamming uses bit wise operations to count the differences between bit strings
"""
struct BinaryHammingDistance <: SemiMetric end

"""
    evaluate(::BinaryHammingDistance, a, b)
    evaluate(::BinaryHammingDistance, a::AbstractVector, b::AbstractVector) where {T<:Unsigned}

Computes the binary hamming distance for bit types and arrays of bit types
"""
function evaluate(::BinaryHammingDistance, a, b)
    d = 0
    @inbounds @simd for i in eachindex(a)
        d += count_ones(a[i] ⊻ b[i])
    end

    d
end

function evaluate(::BinaryHammingDistance, a::T, b::T)::Float64 where {T<:Unsigned}
    count_ones(a ⊻ b)
end


