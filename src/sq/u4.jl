"""
    SQu4

Per-vector (per-column) 4-bit scalar quantization: [`quantize`](@ref SQu4.quantize) packs
two 4-bit codes per `UInt8`, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu4.quantize`, etc.
"""
module SQu4

export quantize, SQu4Vec, SQu4Database, L1, L2, SqL2

using ..ScalarQuant: SQMinC, AbstractDatabase, PreMetric, SemiMetric, Metric, getminbatch
using Polyester
import Distances: evaluate

function quant_u4!(vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    m = length(v)  # even, guaranteed by `quantize`/`SQu4Vec`
    k = 1
    j = 1
    @inbounds while j <= m
        a = round((v[j] - min) * c; digits=0)
        a = UInt8(clamp(a, 0, 15))
        b = round((v[j+1] - min) * c; digits=0)
        b = UInt8(clamp(b, 0, 15))

        vout[k] = a | (b << 4)
        j += 2
        k += 1
    end

    vout
end

function quant_u4!(vout::AbstractVector{UInt8}, v::AbstractVector; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    c = (max - min + eps) / 15f0
    quant_u4!(vout, v, min, 1f0/c)
    SQMinC(min, c)
end

"""
    SQu4Vec(v::AbstractVector)

A single vector quantized to 4 bits per coordinate. It stores the packed codes (two
4-bit codes per `UInt8`, `V`) along with the linear dequantization parameters (`E::SQMinC`)
computed from the extrema of `v`. Indexing a `SQu4Vec` (`qvec[i]`) unpacks and dequantizes
the `i`-th coordinate back to a `Float32` approximation of the original value.

This type is the element produced by indexing a [`SQu4`](@ref) database; it is normally
not created directly by users.

# Arguments
- `v`: the input vector to quantize; `length(v)` must be a multiple of `2` (throws
  `ArgumentError` otherwise), since 2 coordinates are packed into each `UInt8`. Pad `v`
  with an extra coordinate if needed.

!!! note
    If `v` needs padding, any plain (non-quantized) vector later compared against the
    resulting `SQu4Vec` via [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref) (e.g. a query vector)
    must be padded to that same length too, since those distances index the plain vector
    positionally and do not know about the padding.
"""
struct SQu4Vec{VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
end

function SQu4Vec(v::AbstractVector)
    length(v) % 2 == 0 || throw(ArgumentError("SQu4Vec: length(v) = $(length(v)) must be a multiple of 2 (2 coordinates are packed per UInt8)"))
    vout = Vector{UInt8}(undef, length(v) ÷ 2)
    minc = quant_u4!(vout, v)
    SQu4Vec(minc, vout)
end

Base.@propagate_inbounds function Base.getindex(qvec::SQu4Vec, i::Integer)::Float32
    if isodd(i)
        i = (i + 1) >> 1
        val = qvec.V[i] & UInt8(0x0f)
    else
        i >>= 1
        val = qvec.V[i] >> 4
    end

    Float32(val) * qvec.E.c + qvec.E.min
end

Base.length(a::SQu4Vec) = 2length(a.V)
Base.eachindex(a::SQu4Vec) = 1:2length(a.V)

function Base.eachindex(a::SQu4Vec, b::SQu4Vec)
    @assert length(a) === length(b)
    eachindex(a.V)
end

Base.eltype(::SQu4Vec) = Float32
Base.eltype(::Type{T}) where {T<:SQu4Vec} = Float32

"""
    quantize(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 4 bits per coordinate, packing two
codes into each `UInt8`. This reduces the memory footprint of a database of vectors by
roughly a factor of 8 with respect to `Float32` at the cost of precision. Each column
is quantized independently using its own minimum and scale factor, computed from the
extrema of the column so that the whole range `[min, max]` is mapped to the codes
`\\{0, 1, \\ldots, 15\\}`.

`quantize` wraps `SQu4Database` that implements the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu4Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized; `size(X, 1)` (the
  dimension) must be a multiple of `2` (throws `ArgumentError` otherwise), since 2
  coordinates are packed into each `UInt8`. Pad `X` with an extra row if needed.

!!! note
    If `X` needs padding, any plain (non-quantized) query vectors later compared against
    the resulting database via [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref) must be padded to
    that same (padded) dimension too, since those distances index the plain vector
    positionally and do not know about the padding.

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu4.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
function quantize(X::AbstractMatrix)
    SQu4Database(X)
end

struct SQu4Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}

    function SQu4Database(X::AbstractMatrix)
        m, n = size(X)
        m % 2 == 0 || throw(ArgumentError("SQu4.quantize: size(X, 1) = $m must be a multiple of 2 (2 coordinates are packed per UInt8)"))
        Q = Matrix{UInt8}(undef, m ÷ 2, n)
        E = Vector{SQMinC}(undef, n)
        minbatch = getminbatch(n)
        @batch per=thread minbatch=minbatch for i in 1:n
            E[i] = quant_u4!(view(Q, :, i), view(X, :, i))
        end

        new(E, Q)
    end
end

Base.eltype(Q::SQu4Database) = typeof(Q[1])
Base.length(Q::SQu4Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu4Database, i::Integer) 
   SQu4Vec(Q.E[i], view(Q.Q, :, i))
end


### distances

"""
    L1()

The Manhattan (``L_1``) distance between two 4-bit quantized vectors ([`SQu4Vec`](@ref)).
`evaluate` dequantizes both codes coordinate by coordinate and accumulates the absolute
value of their difference.
"""
struct L1 <: Metric end

@inline function evaluate(::L1, A::SQu4Vec, B::SQu4Vec)::Float32
    d = zero(Float32)
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        af = Float32(a & 0x0f) * A.E.c + A.E.min
        bf = Float32(b & 0x0f) * B.E.c + B.E.min 
        m = abs(af - bf)
        a >>= 4; b >>= 4
        af = Float32(a) * A.E.c + A.E.min
        bf = Float32(b) * B.E.c + B.E.min
        m += abs(af - bf)
        d += m
    end

    d
end

function squared_euclidean(A::SQu4Vec, B::SQu4Vec)::Float32
    d = zero(Float32)    
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        af = Float32(a & 0x0f) * A.E.c + A.E.min
        bf = Float32(b & 0x0f) * B.E.c + B.E.min 
        m = (af - bf)^2
        a >>= 4; b >>= 4
        af = Float32(a) * A.E.c + A.E.min
        bf = Float32(b) * B.E.c + B.E.min
        m += (af - bf)^2
        d += m
    end

    d
end

function squared_euclidean(A::SQu4Vec, B)::Float32
    d = zero(Float32)
    n = length(A.V)  # == length(B) ÷ 2, exact (see `quantize`/`SQu4Vec`)

    @inbounds @simd for i in 1:n
        a = A.V[i]
        j = 2i - 1
        af = Float32(a & 0x0f) * A.E.c + A.E.min
        bf = B[j]
        m = (af - bf)^2
        a >>= 4
        af = Float32(a) * A.E.c + A.E.min
        bf = B[j+1]
        m += (af - bf)^2
        d += m
    end

    d
end

squared_euclidean(a, b::SQu4Vec) = squared_euclidean(b, a)

"""
    L2()

The Euclidean (``L_2``) distance between two 4-bit quantized vectors ([`SQu4Vec`](@ref)),
or between a [`SQu4Vec`](@ref) and a plain vector. `evaluate` dequantizes coordinate by
coordinate, accumulates the squared differences (see [`SqL2`](@ref)), and returns its
square root.
"""
struct L2 <: Metric end

@inline evaluate(::L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SqL2()

The squared Euclidean distance between two 4-bit quantized vectors ([`SQu4Vec`](@ref)),
or between a [`SQu4Vec`](@ref) and a plain vector. `evaluate` dequantizes coordinate by
coordinate and accumulates the squared differences `(af - bf)^2`, avoiding the
square root computed by [`L2`](@ref).
"""
struct SqL2 <: Metric end

@inline evaluate(::SqL2, a, b)::Float32 = squared_euclidean(a, b)

end