# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export BinaryHammingDistance

# we export a few bit-handling primitives
# BitArray contains too much extra functionality at the cost of O(1) extra words,
# however this could be an issue, since we represent our database as n vectors

struct BinaryHammingDistance <: PreMetric
end

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

