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


export BinaryRogersTanimotoDistance

"""
   BinaryRogersTanimotoDistance()
   
"""
struct BinaryRogersTanimotoDistance <: SemiMetric end

"""
    evaluate(::BinaryRogersTanimotoDistance, a, b)
    evaluate(::BinaryRogersTanimotoDistance, a::AbstractVector, b::AbstractVector) where {T<:Unsigned}

Computes the Rogers Tanimoto dissimilarity for bit types and arrays of bit types
"""
function evaluate(::BinaryRogersTanimotoDistance, a, b)::Float32
    
    tt, tf, ft, ff = 0, 0, 0, 0
    @inbounds @simd for i in eachindex(a)
        tt += count_ones( a[i] &  b[i])
        ft += count_ones(~a[i] &  b[i])
        tf += count_ones( a[i] & ~b[i])
        ff += count_ones(~a[i] & ~b[i])
    end
    
    1f0 - Float32(tt + ff) / Float32(tt + ff + 2 * (tf + ft))
end

export BinaryRusselRaoDissimilarity

"""
   BinaryRusselRaoDissimilarity()
   
"""
struct BinaryRusselRaoDissimilarity <: SemiMetric end

"""
    evaluate(::BinaryRusselRaoDissimilarity, a, b)
    evaluate(::BinaryRusselRaoDissimilarity, a::AbstractVector, b::AbstractVector) where {T<:Unsigned}

Computes the Russel Rao dissimilarity
"""
function evaluate(::BinaryRusselRaoDissimilarity, a, b)::Float32
    
    tt = 0
    n = length(a) * 64
    @inbounds @simd for i in eachindex(a)
        tt += count_ones(a[i] &  b[i])
    end
    
    1f0 - Float32(tt / n)
end
