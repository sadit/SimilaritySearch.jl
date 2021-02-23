# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export BinaryHammingDistance

"""
   BinaryHammingDistance()
   
Binary hamming uses bit wise operations to count the differences between bit strings
"""
struct BinaryHammingDistance <: PreMetric end
StructTypes.StructType(::Type{BinaryHammingDistance}) = StructTypes.Struct()

"""
    evaluate(::BinaryHammingDistance, a, b)::Float64
    evaluate(::BinaryHammingDistance, a::AbstractVector, b::AbstractVector)::Float64 where T<:Unsigned

Computes the binary hamming distance for bit types and arrays of bit types
"""
function evaluate(::BinaryHammingDistance, a::T, b::T)::Float64 where {T<:Unsigned}
    count_ones(a ⊻ b)
end

function evaluate(::BinaryHammingDistance, a, b)
    d = 0
    @inbounds @simd for i in eachindex(a)
        d += count_ones(a[i] ⊻ b[i])
    end

    d
end

