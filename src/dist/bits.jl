# This file is a part of SimilaritySearch.jl

#export Hamming, RogersTanimoto

"""
   Hamming()
   
Binary hamming uses bit wise operations to count the differences between bit strings
"""
struct Hamming <: Metric end

"""
    evaluate(::Hamming, a, b)
    evaluate(::Hamming, a::AbstractVector, b::AbstractVector) where {T<:Unsigned}

Computes the binary hamming distance for bit types and arrays of bit types
"""
function evaluate(::Hamming, a, b)
    d = 0
    @inbounds @simd for i in eachindex(a)
        d += count_ones(a[i] ⊻ b[i])
    end

    d
end

function evaluate(::Hamming, a::T, b::T)::Float64 where {T<:Unsigned}
    count_ones(a ⊻ b)
end

"""
    RogersTanimoto()

The Rogers-Tanimoto dissimilarity for binary vectors represented as arrays of unsigned
integers (bit strings), where each bit is a binary attribute. Access via
`Dist.Bits.RogersTanimoto`.

Using ``tt`` for the number of bits set to one in both `a` and `b`, ``ff`` for the number
of bits set to zero in both, and ``tf``, ``ft`` for the number of mismatching bits, the
dissimilarity is defined as

```math
RT(u, v) = 1 - \\frac{tt + ff}{tt + ff + 2(tf + ft)}
```

# Examples

```julia
d = Dist.Bits.RogersTanimoto()
evaluate(d, u, v)
```
"""
struct RogersTanimoto <: Metric end

"""
    evaluate(::RogersTanimoto, a, b)
    evaluate(::RogersTanimoto, a::AbstractVector, b::AbstractVector) where {T<:Unsigned}

Computes the Rogers Tanimoto dissimilarity for bit types and arrays of bit types
"""
function evaluate(::RogersTanimoto, a, b)::Float32
    
    tt, tf, ft, ff = 0, 0, 0, 0
    @inbounds @simd for i in eachindex(a)
        tt += count_ones( a[i] &  b[i])
        ft += count_ones(~a[i] &  b[i])
        tf += count_ones( a[i] & ~b[i])
        ff += count_ones(~a[i] & ~b[i])
    end
    
    1f0 - Float32(tt + ff) / Float32(tt + ff + 2 * (tf + ft))
end

export RussellRao

"""
    RussellRao()

The Russell-Rao dissimilarity for binary vectors represented as arrays of unsigned
integers (bit strings), where each bit is a binary attribute. Access via
`Dist.Bits.RussellRao`.

It measures the fraction of bits set to one in both `a` and `b` (``tt``) with respect
to the total number of bits ``n`` (computed as `length(a) * 64`, i.e., it assumes
64-bit words):

```math
RR(u, v) = 1 - \\frac{tt}{n}
```

# Examples

```julia
d = Dist.Bits.RussellRao()
evaluate(d, u, v)
```
"""
struct RussellRao <: PreMetric end

"""
    evaluate(::RussellRao, a, b)
    evaluate(::RussellRao, a::AbstractVector, b::AbstractVector) where {T<:Unsigned}

Computes the Russell Rao dissimilarity
"""
function evaluate(::RussellRao, a, b)::Float32
    
    tt = 0
    n = length(a) * 64
    @inbounds @simd for i in eachindex(a)
        tt += count_ones(a[i] &  b[i])
    end
    
    1f0 - Float32(tt / n)
end
