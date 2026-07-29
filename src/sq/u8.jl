"""
    SQu8

Per-vector (per-column) 8-bit scalar quantization: [`quantize`](@ref SQu8.quantize) stores
one `UInt8` code per coordinate, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu8.quantize`, etc. See also [`SQgu8`](@ref ScalarQuant.SQgu8)
for a variant that shares a single pair of quantization parameters across all columns.
"""
module SQu8

export quantize, SQu8Vec, SQu8Database, L1, L2, SqL2, NormCosine

using ..ScalarQuant: SQMinC, AbstractDatabase, PreMetric, SemiMetric, Metric, getminbatch
using Polyester
import Distances: evaluate

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

Base.length(a::SQu8Vec) = length(a.V)
Base.eachindex(a::SQu8Vec) = eachindex(a.V)
Base.eachindex(a::SQu8Vec, b::SQu8Vec) = eachindex(a.V, b.V)
Base.eltype(::SQu8Vec) = Float32
Base.eltype(::Type{T}) where {T<:SQu8Vec} = Float32

"""
    quantize(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 8 bits per coordinate (one `UInt8` per
coordinate). This reduces the memory footprint of a database of vectors by roughly a
factor of 4 with respect to `Float32` at the cost of precision. Each column is quantized
independently using its own minimum and scale factor, computed from the extrema of the
column so that the whole range `[min, max]` is mapped to the codes `\\{0, 1, \\ldots, 255\\}`.

`quantize` creates a `SQu8Database` struct that follows the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu8Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

See also [`SQgu8`](@ref ScalarQuant.SQgu8)'s `quantize` for a variant that shares a single pair of
quantization parameters across all columns instead of computing them per column.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu8.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
function quantize(X::AbstractMatrix)
    SQu8Database(X)
end

struct SQu8Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}

    function SQu8Database(X::AbstractMatrix)
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

Base.eltype(Q::SQu8Database) = typeof(Q[1])
Base.length(Q::SQu8Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu8Database, i::Integer)
   SQu8Vec(Q.E[i], view(Q.Q, :, i))
end

"""
    quantize(db::SQu8Database, v::AbstractVector)

Quantizes a single vector `v` to 8 bits per coordinate, the same way as the vectors
already stored in `db`, returning a [`SQu8Vec`](@ref). Since [`SQu8`](@ref) computes each
vector's own `min`/scale independently from its own extrema (see [`quantize(X::AbstractMatrix)`](@ref)),
this does not read or depend on `db`'s stored data or parameters; `db` is only used to
validate that `v` has the expected dimension. This is convenient, e.g., to quantize a
query vector the same way as the vectors stored in `db`, so that it can be compared
against them with [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref)/[`NormCosine`](@ref).

# Arguments
- `db`: the database `v` should be dimensionally consistent with
- `v`: the vector to quantize; `length(v)` must equal `db`'s vector dimension

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu8.quantize(X);

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQu8.quantize(db, q);  # quantized the same way as db's vectors
```
"""
function quantize(db::SQu8Database, v::AbstractVector)
    expected = size(db.Q, 1)
    length(v) == expected || throw(ArgumentError("SQu8.quantize(db, v): length(v) = $(length(v)) must equal db's vector dimension ($expected)"))
    SQu8Vec(v)
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
    NormCosine()

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
struct NormCosine <: Metric end

@inline evaluate(::NormCosine, A, B)::Float32 = 1f0 - dotu8(A, B)

"""
    L1()

The Manhattan (``L_1``) distance between two 8-bit quantized vectors ([`SQu8Vec`](@ref)).
`evaluate` dequantizes both codes coordinate by coordinate and accumulates the absolute
value of their difference.
"""
struct L1 <: Metric end

@inline function evaluate(::L1, A::SQu8Vec, B::SQu8Vec)::Float32
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
        a, bf = A.V[i], B[i]
        af = Float32(a) * A.E.c + A.E.min
        d += (af - bf)^2
    end

    d
end

squared_euclidean(a, b::SQu8Vec) = squared_euclidean(b, a)

"""
    L2()

The Euclidean (``L_2``) distance between two 8-bit quantized vectors ([`SQu8Vec`](@ref)).
`evaluate` dequantizes coordinate by coordinate, accumulates the squared differences
(see [`SqL2`](@ref)), and returns its square root.
"""
struct L2 <: Metric end

@inline evaluate(::L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SqL2()

The squared Euclidean distance between two 8-bit quantized vectors ([`SQu8Vec`](@ref)).
`evaluate` dequantizes coordinate by coordinate and accumulates the squared differences
`(af - bf)^2`, avoiding the square root computed by [`L2`](@ref).
"""
struct SqL2 <: Metric end

@inline evaluate(::SqL2, a, b)::Float32 = squared_euclidean(a, b)

end