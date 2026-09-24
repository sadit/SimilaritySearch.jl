"""
    SQu2

Per-vector (per-column) 2-bit scalar quantization: [`quantize`](@ref SQu2.quantize) packs
four 2-bit codes per `UInt8`, each column keeping its own `min`/scale computed from its
extrema. Accessed as `ScalarQuant.SQu2.quantize`, etc.
"""
module SQu2

export quantize, SQu2Vec, SQu2Database, L1, L2, SqL2

using ..ScalarQuant: SQMinC, AbstractDatabase, PreMetric, SemiMetric, Metric, getminbatch, @BATCHES
using SIMD
import Distances: evaluate

function quant_u2!(vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    n = length(v)
    m = n >> 2  # n ÷ 4; exact since `quantize`/`SQu2Vec` require `length(v) % 4 == 0`

    @inbounds @simd for k in 1:m
        j = ((k-1) << 2) + 1
        x = zero(UInt8)
        for i in 0:3
            a = round((Float32(v[j+i]) - min) * c; digits=0)
            a = UInt8(clamp(a, 0, 3))
            x = x | (a << 2i)
        end

        vout[k] = x
    end

    vout
end

function quant_u2!(vout::AbstractVector{UInt8}, v::AbstractVector; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    c = (max - min + eps) / 3f0
    quant_u2!(vout, v, min, 1f0/c)    
    SQMinC(min, c)
end

"""
    SQu2Vec(v::AbstractVector)

A single vector quantized to 2 bits per coordinate. It stores the packed codes (four
2-bit codes per `UInt8`, `V`) along with the linear dequantization parameters (`E::SQMinC`)
computed from the extrema of `v`. Indexing a `SQu2Vec` (`qvec[i]`) unpacks and dequantizes
the `i`-th coordinate back to a `Float32` approximation of the original value.

This type is the element produced by indexing a [`SQu2`](@ref) database; it is normally
not created directly by users.

# Arguments
- `v`: the input vector to quantize; `length(v)` must be a multiple of `4` (throws
  `ArgumentError` otherwise), since 4 coordinates are packed into each `UInt8`. Pad `v`
  with extra coordinates to the next multiple of 4 if needed.

!!! note
    If `v` needs padding, any plain (non-quantized) vector later compared against the
    resulting `SQu2Vec` via [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref) (e.g. a query vector)
    must be padded to that same length too, since those distances index the plain vector
    positionally and do not know about the padding.
"""
### Integer kernels -- the 2-bit counterpart of SQu8's; see the note there for the algebra.
### Four codes in 0:3 per byte, so a squared difference or product is at most 9 per code.
### All four fields of every byte are read, padding included, as the coordinate loops did.

"Σ of the codes and Σ of their squares, over all four fields of every byte."
@inline function u2sums(v::AbstractVector{UInt8})
    n = length(v); i = 1; m = 0x03
    sa = zero(Vec{32,Int32}); saa = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        b = vload(Vec{32,UInt8}, v, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            x = convert(Vec{32,Int32}, (b >>> sh) & m)
            sa += x
            saa = muladd(x, x, saa)
        end
        i += 32
    end
    a = Int(sum(sa)); aa = Int(sum(saa))
    @inbounds if i + 15 <= n
        b = vload(Vec{16,UInt8}, v, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            x = convert(Vec{16,Int32}, (b >>> sh) & m)
            a += Int(sum(x)); aa += Int(sum(x * x))
        end
        i += 16
    end
    @inbounds while i <= n
        b = v[i]
        for sh in 0:2:6
            x = Int((b >>> sh) & m); a += x; aa += x * x
        end
        i += 1
    end

    Float32(a), Float32(aa)
end

"Σ aᵢbᵢ over the unpacked codes."
@inline function u2dotcodes(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x03; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            acc = muladd(convert(Vec{32,Int32}, (bx >>> sh) & m),
                         convert(Vec{32,Int32}, (by >>> sh) & m), acc)
        end
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            s += Int(sum(convert(Vec{16,Int32}, (bx >>> sh) & m) *
                         convert(Vec{16,Int32}, (by >>> sh) & m)))
        end
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        for sh in 0:2:6
            s += Int((bx >>> sh) & m) * Int((by >>> sh) & m)
        end
        i += 1
    end
    s
end

"Σ (aᵢ-bᵢ)² over the unpacked codes, for the equal-scale case."
@inline function u2sqdiffcodes(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x03; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            d = convert(Vec{32,Int32}, (bx >>> sh) & m) - convert(Vec{32,Int32}, (by >>> sh) & m)
            acc = muladd(d, d, acc)
        end
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            d = convert(Vec{16,Int32}, (bx >>> sh) & m) - convert(Vec{16,Int32}, (by >>> sh) & m)
            s += Int(sum(d * d))
        end
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        for sh in 0:2:6
            d = Int((bx >>> sh) & m) - Int((by >>> sh) & m); s += d * d
        end
        i += 1
    end
    s
end

struct SQu2Vec{VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
    Sa::Float32      # Σ codes, Σ codes² -- see `u2sums`
    Saa::Float32
end

SQu2Vec(E::SQMinC, V::AbstractVector{UInt8}) = SQu2Vec(E, V, u2sums(V)...)

function SQu2Vec(v::AbstractVector)
    length(v) % 4 == 0 || throw(ArgumentError("SQu2Vec: length(v) = $(length(v)) must be a multiple of 4 (4 coordinates are packed per UInt8)"))
    vout = Vector{UInt8}(undef, length(v) ÷ 4)
    minc = quant_u2!(vout, v)
    SQu2Vec(minc, vout)
end

Base.@propagate_inbounds function Base.getindex(qvec::SQu2Vec, i::Integer)::Float32
    i = Int32(i-1)
    b = (i >> 2) + 1
    p = i & 0x3
    val = (qvec.V[b] >> 2p) & 0x3
    Float32(val) * qvec.E.c + qvec.E.min
end

Base.length(a::SQu2Vec) = 4length(a.V)
Base.eachindex(a::SQu2Vec) = 1:4length(a.V)

function Base.eachindex(a::SQu2Vec, b::SQu2Vec)
    @assert length(a) === length(b)
    eachindex(a.V)
end

Base.eltype(::SQu2Vec) = Float32
Base.eltype(::Type{T}) where {T<:SQu2Vec} = Float32

"""
    quantize(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 2 bits per coordinate, packing four
codes into each `UInt8`. This reduces the memory footprint of a database of vectors by
roughly a factor of 16 with respect to `Float32` at the cost of precision. Each column
is quantized independently using its own minimum and scale factor, computed from the
extrema of the column so that the whole range `[min, max]` is mapped to the `\\{0,1,2,3\\}`
codes.

`quantize` wraps `SQu2Database` that implements the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu2Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized; `size(X, 1)` (the
  dimension) must be a multiple of `4` (throws `ArgumentError` otherwise), since 4
  coordinates are packed into each `UInt8`. Pad `X` with extra rows to the next multiple
  of 4 if needed.

!!! note
    If `X` needs padding, any plain (non-quantized) query vectors later compared against
    the resulting database via [`L1`](@ref)/[`L2`](@ref)/[`SqL2`](@ref) must be padded to
    that same (padded) dimension too, since those distances index the plain vector
    positionally and do not know about the padding.

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu2.quantize(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
function quantize(X::AbstractMatrix)
    m, n = size(X)
    m % 4 == 0 || throw(ArgumentError("SQu2.quantize: size(X, 1) = $m must be a multiple of 4 (4 coordinates are packed per UInt8)"))
    Q = Matrix{UInt8}(undef, m ÷ 4, n)
    E = Vector{SQMinC}(undef, n)
    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        E[i] = quant_u2!(view(Q, :, i), view(X, :, i))
    end

    SQu2Database(E, Q)
end

struct SQu2Database <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}
    Sa::Vector{Float32}      # per column: Σ codes, Σ codes² -- derived from `Q` alone, so
    Saa::Vector{Float32}     # they are recomputed rather than stored or read back

    """
        SQu2Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})

    Rebuilds a database from its stored fields, quantizing nothing; the code sums are
    recomputed from `Q` in one pass, so nothing but `E` and `Q` has to be persisted.
    """
    function SQu2Database(E::AbstractVector{SQMinC}, Q::AbstractMatrix{UInt8})
        length(E) == size(Q, 2) ||
            throw(ArgumentError("SQu2Database: got $(length(E)) quantization parameters for $(size(Q, 2)) columns; there is exactly one `SQMinC` per stored vector"))
        n = size(Q, 2)
        Sa = Vector{Float32}(undef, n)
        Saa = Vector{Float32}(undef, n)
        minbatch = getminbatch(n)
        @BATCHES minbatch for i in 1:n
            Sa[i], Saa[i] = u2sums(view(Q, :, i))
        end

        new(E, Q, Sa, Saa)
    end
end

Base.eltype(Q::SQu2Database) = typeof(Q[1])
Base.length(Q::SQu2Database) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu2Database, i::Integer)
   SQu2Vec(Q.E[i], view(Q.Q, :, i), Q.Sa[i], Q.Saa[i])
end

"""
    quantize(db::SQu2Database, v::AbstractVector)

Quantizes a single vector `v` to 2 bits per coordinate, the same way as the vectors
already stored in `db`, returning a [`SQu2Vec`](@ref). Since [`SQu2`](@ref) computes each
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

julia> db = ScalarQuant.SQu2.quantize(X);

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQu2.quantize(db, q);  # quantized the same way as db's vectors
```
"""
function quantize(db::SQu2Database, v::AbstractVector)
    expected = 4size(db.Q, 1)
    length(v) == expected || throw(ArgumentError("SQu2.quantize(db, v): length(v) = $(length(v)) must equal db's vector dimension ($expected)"))
    SQu2Vec(v)
end


### distances

"""
    L1()

A Manhattan-like (``L_1``) distance for [`SQu2Vec`](@ref) (2-bit quantized) vectors.
`evaluate` dequantizes both codes coordinate by coordinate and accumulates their
difference `af - bf`.

Note: unlike the general [`L1`](@ref) distance, this implementation does not take the
absolute value of the per-coordinate difference before accumulating, so the result is
not guaranteed to be non-negative; it should be understood as an approximation intended
for relative ranking of 2-bit quantized vectors rather than a true metric.
"""
struct L1 <: Metric end

@inline function evaluate(::L1, A::SQu2Vec, B::SQu2Vec)::Float32
    d = zero(Float32)    
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        m = zero(Float32)
        for p in 0:2:6
            af = Float32((a >> p) & 0x03) * A.E.c + A.E.min
            bf = Float32((b >> p) & 0x03) * B.E.c + B.E.min
            m += (af - bf)
        end

        d += m
    end

    d
end

function squared_euclidean(A::SQu2Vec, B::SQu2Vec)::Float32
    cA, mA = A.E.c, A.E.min
    cB, mB = B.E.c, B.E.min

    # Equal scales -- every self-comparison, every pair of duplicates -- take an exact
    # integer pass, so identical codes still give exactly 0f0 (see the SQu8 counterpart).
    if cA == cB && mA == mB
        return cA * cA * Float32(u2sqdiffcodes(A.V, B.V))
    end

    cA64, mA64 = Float64(cA), Float64(mA)
    cB64, mB64 = Float64(cB), Float64(mB)
    k = mA64 - mB64
    Sab = Float64(u2dotcodes(A.V, B.V))
    ncoords = 4 * length(A.V)      # all four fields of every byte, padding included
    dd = cA64 * cA64 * Float64(A.Saa) + cB64 * cB64 * Float64(B.Saa) - 2 * cA64 * cB64 * Sab +
         2 * k * (cA64 * Float64(A.Sa) - cB64 * Float64(B.Sa)) + ncoords * k * k
    d = Float32(max(0.0, dd))

    d
end

### Mixed comparisons -- see the note in u4.jl. Here a byte holds four coordinates that are
### adjacent in `B`, so the interleave is four-way and costs two rounds of shuffles; it pays for
### itself several times over, since the scalar loop below is the slowest kernel in this family.
const _U2_ILV8  = Val(ntuple(t -> (t-1) % 2 == 0 ? (t-1) ÷ 2 : 8 + (t-1) ÷ 2, 16))
const _U2_ILV16 = Val(ntuple(t -> begin
                                     g = (t-1) ÷ 4; o = (t-1) % 4
                                     o < 2 ? 2g + o : 16 + 2g + (o - 2)
                                 end, 32))

function squared_euclidean(A::SQu2Vec, B::SIMD.FastContiguousArray{Float32,1})::Float32
    nb = length(A.V); i = 1
    c = A.E.c; m = A.E.min
    vc = Vec{32,Float32}(c); vm = Vec{32,Float32}(m)
    acc = zero(Vec{32,Float32})

    @inbounds while i + 7 <= nb                  # 8 bytes == 32 coordinates
        b = vload(Vec{8,UInt8}, A.V, i)
        v0 = b & 0x03; v1 = (b >>> 2) & 0x03; v2 = (b >>> 4) & 0x03; v3 = b >>> 6
        codes = shufflevector(shufflevector(v0, v1, _U2_ILV8),
                              shufflevector(v2, v3, _U2_ILV8), _U2_ILV16)
        d = muladd(convert(Vec{32,Float32}, codes), vc, vm) - vload(Vec{32,Float32}, B, 4i - 3)
        acc = muladd(d, d, acc)
        i += 8
    end

    s = sum(acc)
    @inbounds while i <= nb
        a = A.V[i]; j = 4i - 3
        for p in 0:3
            d = Float32((a >> 2p) & 0x03) * c + m - B[j+p]
            s += d * d
        end
        i += 1
    end

    s
end

function squared_euclidean(A::SQu2Vec, B)::Float32
    d = zero(Float32)
    n = length(A.V)  # == length(B) ÷ 4, exact (see `quantize`/`SQu2Vec`)

    @inbounds @simd for k in 1:n
        j = ((k - 1) << 2) + 1    # B index (each 4)
        a = A.V[k]
        m = zero(Float32)
        for p in 0:3
            af = Float32((a >> 2p) & 0x03) * A.E.c + A.E.min
            bf = B[j+p]
            m += (af - bf)^2
        end

        d += m
    end

    d
end

squared_euclidean(a, b::SQu2Vec) = squared_euclidean(b, a)

"""
    L2()

The Euclidean (``L_2``) distance between two 2-bit quantized vectors ([`SQu2Vec`](@ref)),
or between a [`SQu2Vec`](@ref) and a plain vector. `evaluate` dequantizes coordinate by
coordinate, accumulates the squared differences (see [`SqL2`](@ref)), and returns its
square root.
"""
struct L2 <: Metric end

@inline evaluate(::L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SqL2()

The squared Euclidean distance between two 2-bit quantized vectors ([`SQu2Vec`](@ref)),
or between a [`SQu2Vec`](@ref) and a plain vector. `evaluate` dequantizes coordinate by
coordinate and accumulates the squared differences `(af - bf)^2`, avoiding the
square root computed by [`L2`](@ref).
"""
struct SqL2 <: Metric end

@inline evaluate(::SqL2, a, b)::Float32 = squared_euclidean(a, b)

end