# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export hamming_distance, setbit, resetbit

# we export a few bit-handling primitives
# BitArray contains too much extra functionality at the cost of O(1) extra words,
# however this could be an issue, since we represent our database as n vectors


"""
    hamming_distance(a, b)::Float64
    hamming_distance(a::AbstractVector, b::AbstractVector)::Float64

Computes the binary hamming distance for bit types and arrays of bit types
"""
function hamming_distance(a, b)::Float64
    count_ones(a ⊻ b)
end

function hamming_distance(a::AbstractVector, b::AbstractVector)::Float64
    d = 0
    @inbounds @simd for i in eachindex(a)
        d += count_ones(a[i] ⊻ b[i])
    end

    d
end

"""
   setbit(a::Unsigned, i::Int)

Enables the i-th bit in `a`
"""
function setbit(a::Unsigned, i::Int)
    a | (one(a) << i)
end

"""
    resetbit(a::Unsigned, i::Int)

Disables (0 value) the i-th bit of `a`
"""
function resetbit(a::Unsigned, i::Int)
    a & ~(one(a) << i)
end
