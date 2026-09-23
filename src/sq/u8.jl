"""
    SQu8

Per-vector (per-column) 8-bit scalar quantization: [`quantize`](@ref SQu8.quantize) stores
one `UInt8` code per coordinate, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu8.quantize`, etc. See also [`SQgu8`](@ref ScalarQuant.SQgu8)
for a variant that shares a single pair of quantization parameters across all columns.
"""
module SQu8

export quantize, SQu8Vec, SQu8Database, L1, L2, SqL2, NormCosine

using ..ScalarQuant: SQMinC, AbstractDatabase, PreMetric, SemiMetric, Metric, getminbatch, @BATCHES
using SIMD
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

### Integer kernels.
###
### A dequantized coordinate is `a*c + m`, and `c`/`m` belong to the *vector*, not to the
### database, so two SQu8Vec codes cannot be compared directly the way SQgu8's shared-scale
### ones can. Expanding anyway:
###
###   (a*cA + mA) - (b*cB + mB) = cA*a - cB*b + k,        k = mA - mB
###   Σ(...)² = cA²Σa² + cB²Σb² - 2cAcB Σab + 2k(cA Σa - cB Σb) + n k²
###
### Everything there but `Σab` depends on a single vector, so it is computed once when the
### vector is quantized (`Sa`, `Saa` below) and a distance costs one integer dot product
### instead of dequantizing 2n coordinates into floats. The same expansion serves the dot
### product behind NormCosine.
###
### Overflow: each Int32 lane accumulates at most 255² = 65025 per step, so a single
### unwidened pass is safe up to ~33k steps, i.e. ~528k coordinates -- far beyond any
### vector this is used with. Each kernel runs a 32-lane pass, then at most one 16-lane
### pass, then the scalar remainder: the same shape as SQgu8's, and for the same measured
### reason -- a 16..31 code remainder left to scalar code costs more than vectorizing it.

"Σ of the codes and Σ of their squares, as `Float32`s: the per-vector halves of the expansion above."
@inline function u8sums(v::AbstractVector{UInt8})
    n = length(v); i = 1
    sa = zero(Vec{32,Int32}); saa = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        x = convert(Vec{32,Int32}, vload(Vec{32,UInt8}, v, i))
        sa += x
        saa = muladd(x, x, saa)
        i += 32
    end
    a = Int(sum(sa)); aa = Int(sum(saa))
    @inbounds if i + 15 <= n
        x = convert(Vec{16,Int32}, vload(Vec{16,UInt8}, v, i))
        a += Int(sum(x)); aa += Int(sum(x * x))
        i += 16
    end
    @inbounds while i <= n
        x = Int(v[i]); a += x; aa += x * x; i += 1
    end

    Float32(a), Float32(aa)
end

"Σ aᵢbᵢ over the raw codes -- the only term of the expansion that depends on both vectors."
@inline function u8dotcodes(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        acc = muladd(convert(Vec{32,Int32}, vload(Vec{32,UInt8}, x, i)),
                     convert(Vec{32,Int32}, vload(Vec{32,UInt8}, y, i)), acc)
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        s += Int(sum(convert(Vec{16,Int32}, vload(Vec{16,UInt8}, x, i)) *
                     convert(Vec{16,Int32}, vload(Vec{16,UInt8}, y, i))))
        i += 16
    end
    @inbounds while i <= n; s += Int(x[i]) * Int(y[i]); i += 1; end
    s
end

"Σ (aᵢ-bᵢ)² over the raw codes, for the equal-scale case where it is the whole answer."
@inline function u8sqdiffcodes(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        d = convert(Vec{32,Int32}, vload(Vec{32,UInt8}, x, i)) -
            convert(Vec{32,Int32}, vload(Vec{32,UInt8}, y, i))
        acc = muladd(d, d, acc)
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        d = convert(Vec{16,Int32}, vload(Vec{16,UInt8}, x, i)) -
            convert(Vec{16,Int32}, vload(Vec{16,UInt8}, y, i))
        s += Int(sum(d * d))
        i += 16
    end
    @inbounds while i <= n; d = Int(x[i]) - Int(y[i]); s += d * d; i += 1; end
    s
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
    Sa::Float32      # Σ codes      -- see the expansion above `u8sums`
    Saa::Float32     # Σ codes²
end

"Computes the two code sums for `V`; they are part of the vector, not of any database."
SQu8Vec(E::SQMinC, V::AbstractVector{UInt8}) = SQu8Vec(E, V, u8sums(V)...)

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

"""
    SQu8Database(X::AbstractMatrix)
    SQu8Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})

An [`AbstractDatabase`](@ref) of vectors quantized to 8 bits per coordinate, one `UInt8` per coordinate,
each column carrying its **own** `min`/scale pair (`E[i]`) computed from that vector's own
extrema. Indexing yields a [`SQu8Vec`](@ref), which the distances in this module consume
without dequantizing.

The first constructor quantizes `X` (it is what [`quantize`](@ref) calls). The second takes the
two fields back as they are, quantizing nothing -- that is the one to use after reading `E`/`Q`
from storage, so a stored database goes straight back to work without rebuilding the `Float32`
matrix it came from (which would cost 4x the memory the quantization was chosen to avoid, to
recompute codes that are already in hand). The two must agree: exactly one `SQMinC` per stored
vector, i.e. `length(E) == size(Q, 2)`.

# Fields
- `E::Vector{SQMinC}`: per-column `min`/scale, one entry per stored vector
- `Q::Matrix{UInt8}`: the codes, `size(X, 1)` rows by one column per vector
"""
struct SQu8Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}
    Sa::Vector{Float32}      # per column: Σ codes, Σ codes² -- see `u8sums`. Derived from
    Saa::Vector{Float32}     # `Q` alone, so they are recomputed rather than stored/read.

    function SQu8Database(X::AbstractMatrix)
        m, n = size(X)
        Q = Matrix{UInt8}(undef, m, n)
        E = Vector{SQMinC}(undef, n)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            E[i] = quant_u8!(view(Q, :, i), view(X, :, i))
            Sa[i], Saa[i] = u8sums(view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end

    function SQu8Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})
        length(E) == size(Q, 2) ||
            throw(ArgumentError("SQu8Database: got $(length(E)) quantization parameters for $(size(Q, 2)) columns; there is exactly one `SQMinC` per stored vector"))
        n = size(Q, 2)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            Sa[i], Saa[i] = u8sums(view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end
end

Base.eltype(Q::SQu8Database) = typeof(Q[1])
Base.length(Q::SQu8Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu8Database, i::Integer)
   SQu8Vec(Q.E[i], view(Q.Q, :, i), Q.Sa[i], Q.Saa[i])
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
    # Σ(a*cA + mA)(b*cB + mB) = cAcB Σab + cA mB Σa + cB mA Σb + n mA mB, and only Σab is
    # not already known -- see the note above `u8sums`.
    # The combination is O(1) per distance -- four products -- so it is done in Float64:
    # its terms are large and of opposite signs, and in Float32 their cancellation cost up
    # to 4% of relative accuracy on measured data, while the coordinate loop it replaces
    # had none of that.
    cA, mA = Float64(A.E.c), Float64(A.E.min)
    cB, mB = Float64(B.E.c), Float64(B.E.min)
    Sab = Float64(u8dotcodes(A.V, B.V))
    Float32(cA * cB * Sab + cA * mB * Float64(A.Sa) + cB * mA * Float64(B.Sa) +
            length(A.V) * mA * mB)
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
    cA, mA = A.E.c, A.E.min
    cB, mB = B.E.c, B.E.min

    # Equal scales (which is every comparison of a vector with itself, and every pair of
    # duplicates) collapse to a single integer pass, and that pass is *exact*: identical
    # codes give exactly 0f0, which callers rely on -- `neardup` at radius 0, for one. The
    # general expansion below cannot promise that, since its terms cancel only up to
    # Float32 rounding.
    if cA == cB && mA == mB
        return cA * cA * Float32(u8sqdiffcodes(A.V, B.V))
    end

    # Float64 for the same reason as in `dotu8`: O(1) work per distance, and the expansion
    # subtracts large terms from each other.
    cA64, mA64 = Float64(cA), Float64(mA)
    cB64, mB64 = Float64(cB), Float64(mB)
    k = mA64 - mB64
    Sab = Float64(u8dotcodes(A.V, B.V))
    d = cA64 * cA64 * Float64(A.Saa) + cB64 * cB64 * Float64(B.Saa) - 2 * cA64 * cB64 * Sab +
        2 * k * (cA64 * Float64(A.Sa) - cB64 * Float64(B.Sa)) + length(A.V) * k * k
    Float32(max(0.0, d))   # a difference of positives; rounding can still undershoot zero
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