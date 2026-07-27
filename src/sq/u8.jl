export SQu8, SQu8Vec

### note we need to avoid overflows in high dimensional vectors (i.e., accumulated squared differences like 127^2)

function quant_u8!(vout, v, min::Float32, c::Float32)
    # c = 255f0 / (max - min)
    for j in eachindex(v)
        x = round((v[j] - min) * c; digits=0)
        vout[j] = clamp(x, 0, 255)
    end

    vout
end

function quant_u8!(vout, v::AbstractVector; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    c = (max - min + eps) / 255f0
    quant_u8!(vout, v, min, 1f0/c)
    SQMinC(min, c)
end

"""
    SQu8Vec(v::AbstractVector)

A single vector quantized to 8 bits per coordinate (one `UInt8` code per coordinate,
stored in `V`), along with the linear dequantization parameters (`E::SQMinC`) computed
from the extrema of `v`. Indexing a `SQu8Vec` (`qvec[i]`) dequantizes the `i`-th
coordinate back to a `Float32` approximation of the original value.

This type is the element produced by indexing a [`SQu8`](@ref) database; it is normally
not created directly by users.

# Arguments
- `v`: the input vector to quantize
"""
struct SQu8Vec{VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
end

function SQu8Vec(v::AbstractVector)
    vout = Vector{UInt8}(undef, length(v))
    minc = quant_u8!(vout, v)
    SQu8Vec(minc, vout)
end

Base.@propagate_inbounds function Base.getindex(qvec::SQu8Vec, i::Integer)::Float32
    Float32(qvec.V[i]) * qvec.E.c + qvec.E.min
end

Base.eachindex(a::SQu8Vec) = eachindex(a.V)
Base.eachindex(a::SQu8Vec, b::SQu8Vec) = eachindex(a.V, b.V)
Base.eltype(::SQu8Vec) = Float32
Base.eltype(::Type{T}) where {T<:SQu8Vec} = Float32

"""
    SQu8(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 8 bits per coordinate (one `UInt8` per
coordinate). This reduces the memory footprint of a database of vectors by roughly a
factor of 4 with respect to `Float32` at the cost of precision. Each column is quantized
independently using its own minimum and scale factor, computed from the extrema of the
column so that the whole range `[min, max]` is mapped to the codes `\\{0, 1, \\ldots, 255\\}`.

`SQu8` implements the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu8Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

See also [`sq_global_u8`](@ref) for a variant that shares a single pair of quantization
parameters across all columns instead of computing them per column.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu8(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
struct SQu8 <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}

    function SQu8(X::AbstractMatrix)
        m, n = size(X)
        Q = Matrix{UInt8}(undef, m, n)
        E = Vector{SQMinC}(undef, n)
        minbatch = getminbatch(n)
        @batch per=thread minbatch=minbatch for i in 1:n
            E[i] = quant_u8!(view(Q, :, i), view(X, :, i))
        end

        new(E, Q)
    end
end

Base.eltype(Q::SQu8) = typeof(Q[1])
Base.length(Q::SQu8) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu8, i::Integer) 
   SQu8Vec(Q.E[i], view(Q.Q, :, i))
end


### distances

@inline function dotu8(A::SQu8Vec, B::SQu8Vec)::Float32
    d = zero(Float32)
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        af = Float32(a) * A.E.c + A.E.min
        bf = Float32(b) * B.E.c + B.E.min 
        d += af * bf
    end

    d
end

@inline function dotu8(A::SQu8Vec, B)::Float32
    d = zero(Float32)
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, bf = A.V[i], B[i]
        af = Float32(a) * A.E.c + A.E.min
        d += af * bf
    end

    d
end

dotu8(A, B::SQu8Vec) = dotu8(B, A)

"""
    SQu8NormCosine()

Similar to `Dist.NormCosine` but for 8-bit quantized vectors ([`SQu8Vec`](@ref)); it
assumes that the original (pre-quantization) vectors were already normalized, and
therefore reduces to one minus the dot product:

```math
1 - \\sum_i {u_i v_i}
```

`evaluate` dequantizes coordinate by coordinate (either between two [`SQu8Vec`](@ref),
or between a [`SQu8Vec`](@ref) and a plain vector) and accumulates the products before
computing the final `1 - dot`.
"""
struct SQu8NormCosine <: Metric end

@inline evaluate(::SQu8NormCosine, A, B)::Float32 = 1f0 - dotu8(A, B)

"""
    SQu8L1()

The Manhattan (``L_1``) distance between two 8-bit quantized vectors ([`SQu8Vec`](@ref)).
`evaluate` dequantizes both codes coordinate by coordinate and accumulates the absolute
value of their difference.
"""
struct SQu8L1 <: Metric end

@inline function evaluate(::SQu8L1, A::SQu8Vec, B::SQu8Vec)::Float32
    d = zero(Float32)
    n = length(A.V)

    @fastmath @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        af = Float32(a) * A.E.c + A.E.min
        bf = Float32(b) * B.E.c + B.E.min 
        d += abs(af - bf)
    end

    d
end

function squared_euclidean(A::SQu8Vec, B::SQu8Vec)::Float32
    d = zero(Float32)    
    n = length(A.V)

    @fastmath @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        af = Float32(a) * A.E.c + A.E.min
        bf = Float32(b) * B.E.c + B.E.min 
        d += (af - bf)^2
    end

    d
end

function squared_euclidean(A::SQu8Vec, B)::Float32
    d = zero(Float32)
    n = length(A.V)

    @fastmath @inbounds @simd for i in 1:n
        a, bf = A.V[i], B.V[i]
        af = Float32(a) * A.E.c + A.E.min
        d += (af - bf)^2
    end

    d
end

squared_euclidean(a, b::SQu8Vec) = squared_euclidean(b, a)

"""
    SQu8L2()

The Euclidean (``L_2``) distance between two 8-bit quantized vectors ([`SQu8Vec`](@ref)).
`evaluate` dequantizes coordinate by coordinate, accumulates the squared differences
(see [`SQu8SqL2`](@ref)), and returns its square root.
"""
struct SQu8L2 <: Metric end

@inline evaluate(::SQu8L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SQu8SqL2()

The squared Euclidean distance between two 8-bit quantized vectors ([`SQu8Vec`](@ref)).
`evaluate` dequantizes coordinate by coordinate and accumulates the squared differences
`(af - bf)^2`, avoiding the square root computed by [`SQu8L2`](@ref).
"""
struct SQu8SqL2 <: Metric end

@inline evaluate(::SQu8SqL2, a, b)::Float32 = squared_euclidean(a, b)
