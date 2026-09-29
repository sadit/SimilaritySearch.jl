# This file is a part of SimilaritySearch.jl

export Cosine

### Distances over SQVec.
###
### Every distance between two quantized vectors is one call into the integer kernels of
### `codes.jl` plus O(1) arithmetic on the per-vector sums, so it is defined once for every
### width and both families. The O(1) combination runs in Float64: its terms are large and of
### opposite signs, and in Float32 their cancellation cost up to 4% of relative accuracy on
### measured data, while the coordinate loop it replaces had none of that.
###
### Whatever has to look at dequantized coordinates -- a plain `Float32` query, or `L1` --
### goes through the per-width scans at the end of this file.

"Dot product of the two dequantized vectors, from their codes -- see [`Cosine`](@ref)."
@inline function quantdot(a::SQVec{B}, b::SQVec{B})::Float64 where {B}
    # Σ(a*cA + mA)(b*cB + mB) = cAcB Σab + cA mB Σa + cB mA Σb + n mA mB, and only Σab is
    # not already known -- see the note above `codesums`.
    cA, mA = Float64(a.E.c), Float64(a.E.min)
    cB, mB = Float64(b.E.c), Float64(b.E.min)
    cA * cB * Float64(dotcodes(Val(B), a.V, b.V)) + cA * mB * Float64(a.Sa) + cB * mA * Float64(b.Sa) +
        length(a) * mA * mB
end

"Dot product of a dequantized vector with a plain one, coordinate by coordinate."
dotmixed(A::SQVec{W}, v::AbstractVector) where {W} = _scan(_dotop, Val(W), A, v)

"Euclidean norm of the dequantized vector, from its stored code sums."
@inline function quantnorm(a::SQVec)::Float64
    c, m = Float64(a.E.c), Float64(a.E.min)
    sqrt(max(0.0, c * c * Float64(a.Saa) + 2 * c * m * Float64(a.Sa) + length(a) * m * m))
end

"""
    Cosine()

Cosine dissimilarity (`1 - cos`) between two quantized vectors, computed from their codes:
the dot product of the dequantized vectors divided by their dequantized norms.

Both halves of that need per-vector quantities the codes alone do not carry, and both are
already stored by every [`SQVec`](@ref) (`Sa = Σ codes`, `Saa = Σ codes²`):

- the **dot product** expands to `cA·cB·Σab + cA·mB·Σa + cB·mA·Σb + n·mA·mB`, so dropping
  everything but `Σab` -- as a raw-code dot product does -- only preserves order when the
  offsets are zero. On centered data (any ordinary embedding) it does not: measured over
  20k unit-norm vectors in dim 128, ranking by the raw code dot product gave recall@10 of
  **0.005** against exact cosine, while the full expansion gave **0.97** (issue #77);
- the **norms**, `‖â‖² = c²·Saa + 2·c·m·Sa + n·m²`, correct the drift quantization leaves in
  a vector that was normalized before being quantized. Against a plain vector the query's
  norm is computed on the spot. That matters more the fewer bits
  there are: on non-negative data, recall@10 went 0.9685 -> 0.979 at 8 bits, 0.590 -> 0.666
  at 4 bits, and 0.111 -> 0.126 at 2 bits.

Both must come from the codes, not from the original vector: on the SISAP 2025 benchmarks,
norms measured from the `Float32` vector instead lost 0.01-0.05 of recall@10 and sums of
the original vector in the dot product lost 0.01-0.41, because quantization rescales each
vector slightly and the code sums carry exactly that rescaling (issue #87).
"""
struct Cosine <: SemiMetric end

function evaluate(::Cosine, a::SQVec, b::SQVec)::Float32
    na, nb = quantnorm(a), quantnorm(b)
    (na == 0 || nb == 0) && return 1f0
    Float32(1.0 - clamp(quantdot(a, b) / (na * nb), -1.0, 1.0))
end

# against a plain vector: the query's norm is not stored anywhere, so it costs one pass over
# the query per pair on top of the mixed dot product; a query known to be normalized can use
# `NormCosine` instead, which is the same ranking without that pass
function evaluate(::Cosine, a::SQVec, q::AbstractVector)::Float32
    na = quantnorm(a)
    nq2 = 0.0
    @inbounds @simd for i in eachindex(q)
        nq2 += Float64(q[i]) * Float64(q[i])
    end
    (na == 0 || nq2 == 0) && return 1f0
    Float32(1.0 - clamp(Float64(dotmixed(a, q)) / (na * sqrt(nq2)), -1.0, 1.0))
end

evaluate(c::Cosine, q::AbstractVector, a::SQVec)::Float32 = evaluate(c, a, q)

"""
    NormCosine()

Similar to `Dist.NormCosine` but for quantized vectors ([`SQVec`](@ref)): it assumes that
the original (pre-quantization) vectors were already normalized, and therefore reduces to
one minus the dot product,

```math
1 - \\sum_i {u_i v_i}
```

Between two quantized vectors the dot product comes from the integer expansion behind
[`Cosine`](@ref); against a plain vector it is accumulated coordinate by coordinate. When
the data may have drifted off the unit sphere through quantization, [`Cosine`](@ref)
corrects for it and ranks better at every width.
"""
struct NormCosine <: Metric end

@inline evaluate(::NormCosine, A::SQVec, B::SQVec)::Float32 = 1f0 - Float32(quantdot(A, B))
@inline evaluate(::NormCosine, A::SQVec, B::AbstractVector)::Float32 = 1f0 - dotmixed(A, B)
@inline evaluate(::NormCosine, A::AbstractVector, B::SQVec)::Float32 = 1f0 - dotmixed(B, A)

"""
    L1()

The Manhattan (``L_1``) distance between two quantized vectors ([`SQVec`](@ref)), or between
a quantized vector and a plain one. `evaluate` dequantizes coordinate by coordinate and
accumulates the absolute value of the differences.
"""
struct L1 <: Metric end

function _l1(A::SQVec{W}, B::SQVec{W})::Float32 where {W}
    # Equal scales collapse to `c Σ|a - b|` over the codes, an integer sum that Float32 holds
    # exactly, so identical codes give exactly 0f0 the way `SqL2`'s equal-scale branch does.
    # The dequantized scan cannot promise that: `@fastmath` may contract one side's `a*c + m`
    # into an FMA and not the other's, and a vector against itself came out at 1e-7.
    if A.E == B.E
        return A.E.c * _scanpair(_absdiffop, Val(W), A, B, 1f0, 0f0, 1f0, 0f0)
    end

    _scanpair(_absdiffop, Val(W), A, B, A.E.c, A.E.min, B.E.c, B.E.min)
end
_l1(A::SQVec{W}, B::AbstractVector) where {W} = _scan(_absdiffop, Val(W), A, B)

@inline evaluate(::L1, A::SQVec, B::SQVec)::Float32 = _l1(A, B)
@inline evaluate(::L1, A::SQVec, B::AbstractVector)::Float32 = _l1(A, B)
@inline evaluate(::L1, A::AbstractVector, B::SQVec)::Float32 = _l1(B, A)

function squared_euclidean(A::SQVec{W}, B::SQVec{W})::Float32 where {W}
    cA, mA = A.E.c, A.E.min
    cB, mB = B.E.c, B.E.min

    # Equal scales (every comparison in the global family, and every self-comparison or pair
    # of duplicates in the per-vector one) collapse to a single integer pass, and that pass
    # is *exact*: identical codes give exactly 0f0, which callers rely on -- `neardup` at
    # radius 0, for one. The general expansion below cannot promise that, since its terms
    # cancel only up to Float32 rounding.
    if cA == cB && mA == mB
        return cA * cA * Float32(sqdiffcodes(Val(W), A.V, B.V))
    end

    cA64, mA64 = Float64(cA), Float64(mA)
    cB64, mB64 = Float64(cB), Float64(mB)
    k = mA64 - mB64
    Sab = Float64(dotcodes(Val(W), A.V, B.V))
    n = length(A)      # every field of every byte, padding included
    d = cA64 * cA64 * Float64(A.Saa) + cB64 * cB64 * Float64(B.Saa) - 2 * cA64 * cB64 * Sab +
        2 * k * (cA64 * Float64(A.Sa) - cB64 * Float64(B.Sa)) + n * k * k
    Float32(max(0.0, d))   # a difference of positives; rounding can still undershoot zero
end

# against a plain vector: any indexable one through the scalar scan, a contiguous Float32 one
# through the width's own kernel
squared_euclidean(A::SQVec{W}, B::AbstractVector) where {W} = _scan(_sqdiffop, Val(W), A, B)
squared_euclidean(A::SQVec{W}, B::SIMD.FastContiguousArray{Float32,1}) where {W} = _sqeuclid_mixed(Val(W), A, B)
squared_euclidean(A::AbstractVector, B::SQVec) = squared_euclidean(B, A)

"""
    L2()

The Euclidean (``L_2``) distance between two quantized vectors ([`SQVec`](@ref)), or between
a quantized vector and a plain vector: the square root of [`SqL2`](@ref).
"""
struct L2 <: Metric end

@inline evaluate(::L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SqL2()

The squared Euclidean distance between two quantized vectors ([`SQVec`](@ref)), or between a
quantized vector and a plain vector, avoiding the square root computed by [`L2`](@ref).

Between two quantized vectors it never dequantizes: with equal scales it is one exact
integer pass over the codes, and otherwise the expansion of `Σ (a·cA + mA - b·cB - mB)²`
into one integer dot product plus the per-vector sums. Against a plain `Float32` vector it
unpacks the codes to floats with SIMD and accumulates `(â - b)²`.
"""
struct SqL2 <: Metric end

@inline evaluate(::SqL2, a, b)::Float32 = squared_euclidean(a, b)

### Float kernels: everything that has to look at dequantized coordinates.
###
### `L1` cannot use the integer expansion (`|a - b|` does not expand), and neither can a
### quantized vector against a plain `Float32` one (an unquantized query, above all), since
### one side has no codes. Both run through the scalar scans below: one loop per width,
### unpacking every field of a byte in place, parametrized by the per-coordinate operation.
### LLVM vectorizes those loops on its own, and better than an explicit `Vec{32,Float32}`
### formulation of the same thing: at 8 bits the plain byte loop takes 44 ns per
### 384-coordinate `SqL2` pair against 77 for the explicit one, and 16-lane blocks with one,
### two or four accumulators landed between 61 and 97 across the widths and operations. The
### exception is `SqL2` at 4 and 2 bits, where the interleaving of a byte's coordinates stops
### the auto-vectorizer and one shuffle per block recovers it (69 and 80 ns against 108 and
### 101 for the scalar scan); those two kernels are hand-written at the end.

# the per-coordinate operations
# `@fastmath` here and in the scans is what the hand-written loops these replace had: it lets
# the compiler contract and reassociate, which is worth 15% at 8 bits (55 -> 46 ns per pair)
@inline _sqdiffop(a, b, acc) = @fastmath (d = a - b; acc + d * d)
@inline _dotop(a, b, acc) = @fastmath acc + a * b
@inline _absdiffop(a, b, acc) = @fastmath acc + abs(a - b)

"Reduces `op` over the dequantized coordinates of `A` and the plain vector `B`."
@inline function _scan(op::F, ::Val{8}, A::SQVec, B)::Float32 where {F}
    d = zero(Float32); n = length(A.V); c = A.E.c; m = A.E.min
    @fastmath @inbounds @simd for i in 1:n
        d = op(Float32(A.V[i]) * c + m, Float32(B[i]), d)
    end

    d
end

@inline function _scan(op::F, ::Val{4}, A::SQVec, B)::Float32 where {F}
    d = zero(Float32); n = length(A.V); c = A.E.c; m = A.E.min
    @fastmath @inbounds @simd for i in 1:n
        a = A.V[i]; j = 2i - 1
        d = op(Float32(a & 0x0f) * c + m, Float32(B[j]), d)
        d = op(Float32(a >>> 4) * c + m, Float32(B[j+1]), d)
    end

    d
end

@inline function _scan(op::F, ::Val{2}, A::SQVec, B)::Float32 where {F}
    d = zero(Float32); n = length(A.V); c = A.E.c; m = A.E.min
    @fastmath @inbounds @simd for i in 1:n
        a = A.V[i]; j = 4i - 3
        d = op(Float32(a & 0x03) * c + m, Float32(B[j]), d)
        d = op(Float32((a >>> 2) & 0x03) * c + m, Float32(B[j+1]), d)
        d = op(Float32((a >>> 4) & 0x03) * c + m, Float32(B[j+2]), d)
        d = op(Float32(a >>> 6) * c + m, Float32(B[j+3]), d)
    end

    d
end

"Reduces `op` over the coordinates of two quantized vectors of the same width, each dequantized with the affine map given (`code * c + m`)."
@inline function _scanpair(op::F, ::Val{8}, A::SQVec, B::SQVec, cA::Float32, mA::Float32, cB::Float32, mB::Float32)::Float32 where {F}
    d = zero(Float32); n = length(A.V)
    @fastmath @inbounds @simd for i in 1:n
        d = op(Float32(A.V[i]) * cA + mA, Float32(B.V[i]) * cB + mB, d)
    end

    d
end

@inline function _scanpair(op::F, ::Val{4}, A::SQVec, B::SQVec, cA::Float32, mA::Float32, cB::Float32, mB::Float32)::Float32 where {F}
    d = zero(Float32); n = length(A.V)
    @fastmath @inbounds @simd for i in 1:n
        a = A.V[i]; b = B.V[i]
        d = op(Float32(a & 0x0f) * cA + mA, Float32(b & 0x0f) * cB + mB, d)
        d = op(Float32(a >>> 4) * cA + mA, Float32(b >>> 4) * cB + mB, d)
    end

    d
end

@inline function _scanpair(op::F, ::Val{2}, A::SQVec, B::SQVec, cA::Float32, mA::Float32, cB::Float32, mB::Float32)::Float32 where {F}
    d = zero(Float32); n = length(A.V)
    @fastmath @inbounds @simd for i in 1:n
        a = A.V[i]; b = B.V[i]
        d = op(Float32(a & 0x03) * cA + mA, Float32(b & 0x03) * cB + mB, d)
        d = op(Float32((a >>> 2) & 0x03) * cA + mA, Float32((b >>> 2) & 0x03) * cB + mB, d)
        d = op(Float32((a >>> 4) & 0x03) * cA + mA, Float32((b >>> 4) & 0x03) * cB + mB, d)
        d = op(Float32(a >>> 6) * cA + mA, Float32(b >>> 6) * cB + mB, d)
    end

    d
end

### `SqL2` against a contiguous `Float32` vector, at 4 and 2 bits: each byte holds coordinates
### that are adjacent in `B`, and that interleaving is what stops the compiler from vectorizing
### the scan above; doing it explicitly with one shuffle per block recovers it.

_sqeuclid_mixed(::Val{8}, A::SQVec, B) = _scan(_sqdiffop, Val(8), A, B)

"Interleaves the low and high nibble lanes back into coordinate order: [lo1, hi1, lo2, hi2, ...]."
const _U4_ILV = Val(ntuple(t -> (t-1) % 2 == 0 ? (t-1) ÷ 2 : 16 + (t-1) ÷ 2, 32))

function _sqeuclid_mixed(::Val{4}, A::SQVec, B)::Float32
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

# Here a byte holds four coordinates that are adjacent in `B`, so the interleave is four-way
# and costs two rounds of shuffles; it pays for itself several times over, since the scalar
# scan it replaces is the slowest kernel in this family.
const _U2_ILV8  = Val(ntuple(t -> (t-1) % 2 == 0 ? (t-1) ÷ 2 : 8 + (t-1) ÷ 2, 16))
const _U2_ILV16 = Val(ntuple(t -> begin
                                     g = (t-1) ÷ 4; o = (t-1) % 4
                                     o < 2 ? 2g + o : 16 + 2g + (o - 2)
                                 end, 32))

function _sqeuclid_mixed(::Val{2}, A::SQVec, B)::Float32
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
