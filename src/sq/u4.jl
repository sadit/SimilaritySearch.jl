"""
    SQu4

Per-vector (per-column) 4-bit scalar quantization: [`quantize`](@ref SQu4.quantize) packs
two 4-bit codes per `UInt8`, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu4.quantize`, etc.
"""
module SQu4

export quantize, SQu4Vec, SQu4Database, L1, L2, SqL2

using ..ScalarQuant: SQMinC, AbstractDatabase, PreMetric, SemiMetric, Metric, getminbatch, @BATCHES
using SIMD
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
### Integer kernels -- the 4-bit counterpart of SQu8's; see the note there for the algebra.
### Each byte holds two codes in 0:15, so a squared difference or product is at most 225 per
### code and an Int32 lane never overflows for any realistic dimension. Both nibbles of every
### byte are read, padding included, exactly as the coordinate loops these replace did.

"Σ of the codes and Σ of their squares, over both nibbles of every byte."
@inline function u4sums(v::AbstractVector{UInt8})
    n = length(v); i = 1; m = 0x0f
    sa = zero(Vec{32,Int32}); saa = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        b = vload(Vec{32,UInt8}, v, i)
        lo = convert(Vec{32,Int32}, b & m); hi = convert(Vec{32,Int32}, b >>> 4)
        sa += lo + hi
        saa = muladd(lo, lo, muladd(hi, hi, saa))
        i += 32
    end
    a = Int(sum(sa)); aa = Int(sum(saa))
    @inbounds if i + 15 <= n
        b = vload(Vec{16,UInt8}, v, i)
        lo = convert(Vec{16,Int32}, b & m); hi = convert(Vec{16,Int32}, b >>> 4)
        a += Int(sum(lo)) + Int(sum(hi)); aa += Int(sum(lo * lo)) + Int(sum(hi * hi))
        i += 16
    end
    @inbounds while i <= n
        b = v[i]; lo = Int(b & m); hi = Int(b >>> 4)
        a += lo + hi; aa += lo * lo + hi * hi; i += 1
    end

    Float32(a), Float32(aa)
end

"Σ aᵢbᵢ over the unpacked codes."
@inline function u4dotcodes(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x0f; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        acc = muladd(convert(Vec{32,Int32}, bx & m), convert(Vec{32,Int32}, by & m), acc)
        acc = muladd(convert(Vec{32,Int32}, bx >>> 4), convert(Vec{32,Int32}, by >>> 4), acc)
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        s += Int(sum(convert(Vec{16,Int32}, bx & m) * convert(Vec{16,Int32}, by & m))) +
             Int(sum(convert(Vec{16,Int32}, bx >>> 4) * convert(Vec{16,Int32}, by >>> 4)))
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        s += Int(bx & m) * Int(by & m) + Int(bx >>> 4) * Int(by >>> 4)
        i += 1
    end
    s
end

"Σ (aᵢ-bᵢ)² over the unpacked codes, for the equal-scale case."
@inline function u4sqdiffcodes(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x0f; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        dlo = convert(Vec{32,Int32}, bx & m) - convert(Vec{32,Int32}, by & m)
        dhi = convert(Vec{32,Int32}, bx >>> 4) - convert(Vec{32,Int32}, by >>> 4)
        acc = muladd(dlo, dlo, muladd(dhi, dhi, acc))
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        dlo = convert(Vec{16,Int32}, bx & m) - convert(Vec{16,Int32}, by & m)
        dhi = convert(Vec{16,Int32}, bx >>> 4) - convert(Vec{16,Int32}, by >>> 4)
        s += Int(sum(dlo * dlo)) + Int(sum(dhi * dhi))
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        dlo = Int(bx & m) - Int(by & m); dhi = Int(bx >>> 4) - Int(by >>> 4)
        s += dlo * dlo + dhi * dhi
        i += 1
    end
    s
end

struct SQu4Vec{VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
    Sa::Float32      # Σ codes, Σ codes² -- see `u4sums`
    Saa::Float32
end

SQu4Vec(E::SQMinC, V::AbstractVector{UInt8}) = SQu4Vec(E, V, u4sums(V)...)

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

"""
    SQu4Database(X::AbstractMatrix)
    SQu4Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})

An [`AbstractDatabase`](@ref) of vectors quantized to 4 bits per coordinate, two 4-bit codes packed per `UInt8`,
each column carrying its **own** `min`/scale pair (`E[i]`) computed from that vector's own
extrema. Indexing yields a [`SQu4Vec`](@ref), which the distances in this module consume
without dequantizing.

The first constructor quantizes `X` (it is what [`quantize`](@ref) calls). The second takes the
two fields back as they are, quantizing nothing -- that is the one to use after reading `E`/`Q`
from storage, so a stored database goes straight back to work without rebuilding the `Float32`
matrix it came from (which would cost 8x the memory the quantization was chosen to avoid, to
recompute codes that are already in hand). The two must agree: exactly one `SQMinC` per stored
vector, i.e. `length(E) == size(Q, 2)`.

# Fields
- `E::Vector{SQMinC}`: per-column `min`/scale, one entry per stored vector
- `Q::Matrix{UInt8}`: the codes, `size(X, 1) ÷ 2` rows by one column per vector
"""
struct SQu4Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}
    Sa::Vector{Float32}      # per column: Σ codes, Σ codes² -- derived from `Q` alone, so
    Saa::Vector{Float32}     # they are recomputed rather than stored or read back

    function SQu4Database(X::AbstractMatrix)
        m, n = size(X)
        m % 2 == 0 || throw(ArgumentError("SQu4.quantize: size(X, 1) = $m must be a multiple of 2 (2 coordinates are packed per UInt8)"))
        Q = Matrix{UInt8}(undef, m ÷ 2, n)
        E = Vector{SQMinC}(undef, n)
        minbatch = getminbatch(n)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        @BATCHES minbatch for i in 1:n
            E[i] = quant_u4!(view(Q, :, i), view(X, :, i))
            Sa[i], Saa[i] = u4sums(view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end

    function SQu4Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})
        length(E) == size(Q, 2) ||
            throw(ArgumentError("SQu4Database: got $(length(E)) quantization parameters for $(size(Q, 2)) columns; there is exactly one `SQMinC` per stored vector"))
        n = size(Q, 2)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            Sa[i], Saa[i] = u4sums(view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end
end

Base.eltype(Q::SQu4Database) = typeof(Q[1])
Base.length(Q::SQu4Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu4Database, i::Integer)
   SQu4Vec(Q.E[i], view(Q.Q, :, i), Q.Sa[i], Q.Saa[i])
end

"""
    quantize(db::SQu4Database, v::AbstractVector)

Quantizes a single vector `v` to 4 bits per coordinate, the same way as the vectors
already stored in `db`, returning a [`SQu4Vec`](@ref). Since [`SQu4`](@ref) computes each
vector's own `min`/scale independently from its own extrema (see [`quantize(X::AbstractMatrix)`](@ref)),
this does not read or depend on `db`'s stored data or parameters; `db` is only used to
validate that `v` has the expected (padded) dimension. This is convenient, e.g., to
quantize a query vector the same way as the vectors stored in `db`, so that it can be
compared against them with [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref).

# Arguments
- `db`: the database `v` should be dimensionally consistent with
- `v`: the vector to quantize; `length(v)` must equal `db`'s (padded) vector dimension

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu4.quantize(X);

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQu4.quantize(db, q);  # quantized the same way as db's vectors
```
"""
function quantize(db::SQu4Database, v::AbstractVector)
    expected = 2size(db.Q, 1)
    length(v) == expected || throw(ArgumentError("SQu4.quantize(db, v): length(v) = $(length(v)) must equal db's vector dimension ($expected)"))
    SQu4Vec(v)
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
    cA, mA = A.E.c, A.E.min
    cB, mB = B.E.c, B.E.min

    # Equal scales -- every self-comparison, every pair of duplicates -- take an exact
    # integer pass, so identical codes still give exactly 0f0 (see the SQu8 counterpart).
    if cA == cB && mA == mB
        return cA * cA * Float32(u4sqdiffcodes(A.V, B.V))
    end

    cA64, mA64 = Float64(cA), Float64(mA)
    cB64, mB64 = Float64(cB), Float64(mB)
    k = mA64 - mB64
    Sab = Float64(u4dotcodes(A.V, B.V))
    ncoords = 2 * length(A.V)      # both nibbles of every byte, padding included
    d = cA64 * cA64 * Float64(A.Saa) + cB64 * cB64 * Float64(B.Saa) - 2 * cA64 * cB64 * Sab +
        2 * k * (cA64 * Float64(A.Sa) - cB64 * Float64(B.Sa)) + ncoords * k * k
    Float32(max(0.0, d))
end

### Mixed comparisons -- a quantized vector against a plain `Float32` one -- cannot use the
### integer expansion the SQu4Vec/SQu4Vec kernels do: one side is not quantized, so there is
### nothing to keep in integers. What they *can* avoid is the scalar unpacking. Each byte holds
### two coordinates that are adjacent in `B`, and that interleaving is what stops the compiler
### from vectorizing the loop below; doing it explicitly with one shuffle per block recovers it.
"Interleaves the low and high nibble lanes back into coordinate order: [lo1, hi1, lo2, hi2, ...]."
const _U4_ILV = Val(ntuple(t -> (t-1) % 2 == 0 ? (t-1) ÷ 2 : 16 + (t-1) ÷ 2, 32))

function squared_euclidean(A::SQu4Vec, B::SIMD.FastContiguousArray{Float32,1})::Float32
    nb = length(A.V); i = 1
    c = A.E.c; m = A.E.min
    vc = Vec{32,Float32}(c); vm = Vec{32,Float32}(m)
    acc = zero(Vec{32,Float32})

    @inbounds while i + 15 <= nb                 # 16 bytes == 32 coordinates
        b = vload(Vec{16,UInt8}, A.V, i)
        codes = shufflevector(b & 0x0f, b >>> 4, _U4_ILV)
        d = muladd(convert(Vec{32,Float32}, codes), vc, vm) - vload(Vec{32,Float32}, B, 2i - 1)
        acc = muladd(d, d, acc)
        i += 16
    end

    s = sum(acc)
    @inbounds while i <= nb
        a = A.V[i]; j = 2i - 1
        d1 = Float32(a & 0x0f) * c + m - B[j]
        d2 = Float32(a >>> 4) * c + m - B[j+1]
        s += d1 * d1 + d2 * d2
        i += 1
    end

    s
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