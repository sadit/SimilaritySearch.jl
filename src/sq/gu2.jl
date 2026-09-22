"""
    SQgu2

Global (database-wide) 2-bit scalar quantization: [`quantize`](@ref SQgu2.quantize) maps
every coordinate of every vector using a single shared `min`/scale pair, packing four
2-bit codes per `UInt8`, and [`NormCosine`](@ref SQgu2.NormCosine)/[`SqL2`](@ref
SQgu2.SqL2) compare the resulting codes directly with SIMD. Accessed as
`ScalarQuant.SQgu2.quantize`, etc.

It is the coarsest member of the global family (`SQgu2`/`SQgu4`/`SQgu8`): four codes fit
in one byte, so a `Float32` database shrinks by 16x, at the cost of only four levels per
coordinate. Codes are laid out low bits first (coordinate `4k+1` in bits `0:1`, `4k+2` in
bits `2:3`, and so on), exactly like [`SQu2`](@ref ScalarQuant.SQu2)'s per-column variant.
"""
module SQgu2

export quantize, quantize!, NormCosine, SqL2

using ..ScalarQuant: getminbatch, sqglobalscale, Dist, @BATCHES
using Statistics: quantile
using SIMD

"Quantizes `v` into `vout` (four 2-bit codes packed per `UInt8`) using the global `min`/scale `c`; returns `vout`."
function quant_global_u2!(vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    m = length(v)
    k = 1
    j = 1
    @inbounds while j <= m
        x = zero(UInt8)
        for p in 0:3
            i = j + p
            i > m && break
            a = round((Float32(v[i]) - min) * c; digits=0)
            x |= UInt8(clamp(a, 0, 3)) << 2p
        end

        vout[k] = x
        j += 4
        k += 1
    end

    vout
end

"""
    quantize(X::AbstractMatrix; minmax=nothing, quant=[0.025, 0.975], samplesize=0)

Scalar-quantizes every entry of `X` to 2 bits using a single, global pair of
dequantization parameters shared by all columns, unlike [`SQu2`](@ref ScalarQuant.SQu2)'s
`quantize` which computes an independent `min`/scale per column. As with
[`SQgu4`](@ref ScalarQuant.SQgu4)/[`SQgu8`](@ref ScalarQuant.SQgu8), this is useful when
the columns of `X` share a comparable value range, since a single global range provides
enough precision while being cheaper to compute and store.

Codes are packed four per `UInt8` (low bits first), so the returned matrix has
`cld(size(X, 1), 4)` rows. Packing four dimensions into a single byte, combined with a
*global* (rather than per-column) `min`/scale, lets [`SqL2`](@ref) and
[`NormCosine`](@ref) operate directly on the packed codes with SIMD, without any
per-element dequantization: since every column shares the same affine mapping,
comparisons and (squared) differences computed in code space are already proportional to
the ones in the original space.

The global `[min, max]` range is estimated by sampling entries of `X` and taking
quantiles of the sample (to be robust to outliers), unless it is provided explicitly via
`minmax`. Every entry `x` is then mapped as `round(clamp((x - min) * c, 0, 3))` with
`c = 3 / (max - min + 1e-6)`.

# Arguments
- `X`: the matrix to quantize; each entry is quantized independently but using shared
  `min`/`max` values
- `minmax`: an optional `(min, max)` tuple giving the value range to use; when `nothing`
  (the default) the range is estimated from a random sample of the entries of `X` using
  `quant`
- `quant`: the lower and upper quantiles (of the sampled entries of `X`) used to estimate
  `min` and `max` when `minmax` is not given
- `samplesize`: the number of entries sampled (with replacement) from `X` to estimate the
  quantiles; when `0` (the default) it is set to `ceil(Int, length(X)^0.5)`

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> Q = ScalarQuant.SQgu2.quantize(X; minmax=(0f0, 1f0));  # explicit range

julia> size(Q), eltype(Q)  # (2, 1000), UInt8
```
"""
function quantize(X::AbstractMatrix;
        minmax=nothing,
        quant=[0.025, 0.975],
        samplesize=0
    )
    m, n = size(X)
    Q = Matrix{UInt8}(undef, cld(m, 4), n)
    min, max = _minmax(vec(X), minmax, quant, samplesize)
    c = sqglobalscale(3, min, max)
    min = Float32(min)

    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        quant_global_u2!(view(Q, :, i), view(X, :, i), min, c)
    end

    Q
end

"""
    quantize(v::AbstractVector; minmax=nothing, quant=[0.025, 0.975], samplesize=0)

Scalar-quantizes a single vector `v` to 2 bits, using the same global scheme as
[`quantize(X::AbstractMatrix)`](@ref), producing a `Vector{UInt8}` (four codes per byte)
of length `cld(length(v), 4)`, instead of a `Matrix{UInt8}`.

!!! warning
    To produce codes that are meaningfully comparable (e.g. for distance computations
    with [`NormCosine`](@ref)/[`SqL2`](@ref)) to those of an already-quantized dataset,
    `minmax` **must** be the exact same `(min, max)` pair used to quantize that dataset
    (e.g., a query vector must be quantized with the dataset's `minmax`, not its own).
    Leaving `minmax=nothing` here estimates a *new*, independent range from `v` alone,
    which will in general **not** match the range used for a previously-quantized
    dataset, silently producing incompatible, meaningless codes.

# Arguments
- `v`: the vector to quantize
- `minmax`: an optional `(min, max)` tuple giving the value range to use; when `nothing`
  (the default) the range is estimated from a random sample of `v`'s entries using
  `quant`. **Must match the dataset's `minmax`** if `v` is to be compared against an
  existing quantized dataset.
- `quant`: the lower and upper quantiles used to estimate `min` and `max` when `minmax`
  is not given
- `samplesize`: the number of entries sampled (with replacement) from `v` to estimate the
  quantiles; when `0` (the default) it is set to `ceil(Int, length(v)^0.5)`
"""
function quantize(v::AbstractVector;
        minmax=nothing,
        quant=[0.025, 0.975],
        samplesize=0
    )
    vout = Vector{UInt8}(undef, cld(length(v), 4))
    min, max = _minmax(v, minmax, quant, samplesize)
    quant_global_u2!(vout, v, Float32(min), sqglobalscale(3, min, max))
    vout
end

"""
    quantize!(vout::AbstractVector{UInt8}, v::AbstractVector, minmax) -> vout

In-place, allocation-free [`quantize`](@ref) of a single vector into a caller-provided
`vout` of length `cld(length(v), 4)`, using the explicit `(min, max)` range `minmax`.
Intended for encoding loops that reuse their output buffer (see
`Projections.quantsketch`); the range is never estimated here, precisely so every vector
encoded through it stays comparable.
"""
function quantize!(vout::AbstractVector{UInt8}, v::AbstractVector, minmax)
    min, max = minmax
    quant_global_u2!(vout, v, Float32(min), sqglobalscale(3, min, max))
end

function _minmax(v, minmax, quant, samplesize)
    minmax === nothing || return minmax
    n = length(v)
    samplesize = samplesize === 0 ? ceil(Int, n^0.5) : samplesize
    quantile(rand(v, samplesize), quant)
end



### SIMD kernels.
###
### gu4.jl widens each byte into `Int32`/`UInt32` lanes, which is the natural thing to do
### when a byte holds two codes. Here a byte holds *four*, so that layout would pay four
### widenings-to-32-bits per byte and spend the whole register file on accumulators --
### measurably worse per byte than the 4-bit kernel despite moving half the memory.
###
### These kernels accumulate in `Int16` lanes instead, which fits the arithmetic exactly:
### a 2-bit code is in `0:3`, so a per-field product or squared difference is at most `9`
### and a whole byte contributes at most `4 * 9 == 36` to its lane. That halves the lane
### width, so one SIMD operation covers `N = 32` bytes instead of 16, and it collapses the
### four per-field accumulators into a single `muladd` chain per chunk, leaving registers
### free for `UNROLL = 4` independent chains.
###
### The catch of a narrow accumulator is overflow, so the loop runs in blocks of at most
### `BLOCK` bytes and widens into a scalar `Int` at the end of each: `36 * (BLOCK / CHUNK)`
### stays far below `typemax(Int16)`, and the widening happens once per 64KB rather than
### once per byte. Every sketch this module is likely to see fits in a single block.

const _U2_N = 32                    # bytes (== 128 codes) per SIMD operation
const _U2_UNROLL = 4
const _U2_CHUNK = _U2_N * _U2_UNROLL
const _U2_BLOCK = 512 * _U2_CHUNK   # 512 iterations * 36 per lane == 18432 < typemax(Int16)

"Accumulates, in `Int16` lanes, the per-field squared differences of the `N` bytes of `x`/`y` at `i`."
@inline function _u2_sqdiff(x, y, i, acc::Vec{N,Int16}) where {N}
    vx = vload(Vec{N,UInt8}, x, i)
    vy = vload(Vec{N,UInt8}, y, i)
    m = 0x03
    d0 = convert(Vec{N,Int16}, vx & m)         - convert(Vec{N,Int16}, vy & m)
    d1 = convert(Vec{N,Int16}, (vx >>> 2) & m) - convert(Vec{N,Int16}, (vy >>> 2) & m)
    d2 = convert(Vec{N,Int16}, (vx >>> 4) & m) - convert(Vec{N,Int16}, (vy >>> 4) & m)
    d3 = convert(Vec{N,Int16}, vx >>> 6)       - convert(Vec{N,Int16}, vy >>> 6)
    muladd(d0, d0, muladd(d1, d1, muladd(d2, d2, muladd(d3, d3, acc))))
end

"Accumulates, in `Int16` lanes, the per-field products of the `N` bytes of `x`/`y` at `i`."
@inline function _u2_dot(x, y, i, acc::Vec{N,Int16}) where {N}
    vx = vload(Vec{N,UInt8}, x, i)
    vy = vload(Vec{N,UInt8}, y, i)
    m = 0x03
    a0 = convert(Vec{N,Int16}, vx & m);         b0 = convert(Vec{N,Int16}, vy & m)
    a1 = convert(Vec{N,Int16}, (vx >>> 2) & m); b1 = convert(Vec{N,Int16}, (vy >>> 2) & m)
    a2 = convert(Vec{N,Int16}, (vx >>> 4) & m); b2 = convert(Vec{N,Int16}, (vy >>> 4) & m)
    a3 = convert(Vec{N,Int16}, vx >>> 6);       b3 = convert(Vec{N,Int16}, vy >>> 6)
    muladd(a0, b0, muladd(a1, b1, muladd(a2, b2, muladd(a3, b3, acc))))
end

"""
    _u2_reduce(kernel, x, y) -> (Int, Int)

Runs `kernel` (`_u2_sqdiff` or `_u2_dot`) over as much of `x`/`y` as SIMD can cover, and
returns the accumulated total together with the index of the first byte it did *not*
process -- the caller finishes those (fewer than 16) scalar-wise. It handles the blocking
that keeps the `Int16` lanes from overflowing, the partially-unrolled remainder, and the
half-width cleanup pass below.
"""
@inline function _u2_reduce(kernel::F, x, y) where {F}
    N, CHUNK, BLOCK = _U2_N, _U2_CHUNK, _U2_BLOCK
    n = length(x)
    res = 0
    i = 1

    @inbounds while i + N - 1 <= n
        stop = Base.min(n, i + BLOCK - 1)
        acc1 = zero(Vec{N,Int16}); acc2 = zero(Vec{N,Int16})
        acc3 = zero(Vec{N,Int16}); acc4 = zero(Vec{N,Int16})

        while i + CHUNK - 1 <= stop
            acc1 = kernel(x, y, i,      acc1)
            acc2 = kernel(x, y, i + N,  acc2)
            acc3 = kernel(x, y, i + 2N, acc3)
            acc4 = kernel(x, y, i + 3N, acc4)
            i += CHUNK
        end

        while i + N - 1 <= stop
            acc1 = kernel(x, y, i, acc1)
            i += N
        end

        # widened one accumulator at a time: their *sum* can exceed Int16 even when each
        # one cannot (this loop runs at least once, so `i` always advances -- no hang)
        res += Int(sum(convert(Vec{N,Int32}, acc1))) + Int(sum(convert(Vec{N,Int32}, acc2))) +
               Int(sum(convert(Vec{N,Int32}, acc3))) + Int(sum(convert(Vec{N,Int32}, acc4)))
    end

    # Half-width cleanup: `N = 32` bytes is a lot to require before any vector work
    # happens, and the loop above takes nothing at all when fewer than that remain. A
    # 2-bit sketch of 64 hyperplanes is exactly 16 bytes, so without this pass it was
    # decoded entirely by the caller's scalar tail -- 64 shifts and masks, measured at
    # ~67ns against ~14ns here. Any length leaves at most 31 bytes for the loop above to
    # refuse, so one 16-lane pass is all that is ever needed; a single pass accumulates at
    # most 36 per lane, far from Int16 overflow, so it needs no blocking of its own.
    @inbounds while i + 15 <= n
        acc = kernel(x, y, i, zero(Vec{16,Int16}))
        res += Int(sum(convert(Vec{16,Int32}, acc)))
        i += 16
    end

    res, i
end

"""
    NormCosine()

Dissimilarity between two vectors quantized with [`quantize`](@ref) (four globally-scaled
2-bit codes per byte), computed as the negative dot product of the raw packed codes.
Since both vectors share the same global `min`/scale, the dot product of codes is an
affine, order-preserving proxy of the dot product of the original (typically
pre-normalized) vectors, so no per-element dequantization is needed. `evaluate` unpacks
each byte into its four 2-bit fields and accumulates their products with SIMD.
"""
struct NormCosine <: Dist.SemiMetric
end

function Dist.evaluate(::NormCosine, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Byte arrays must be the same length"))

    res, i = _u2_reduce(_u2_dot, x, y)
    n = length(x)

    @inbounds while i <= n
        xv, yv = x[i], y[i]
        for p in 0:2:6
            res += Int((xv >>> p) & 0x03) * Int((yv >>> p) & 0x03)
        end
        i += 1
    end

    -Float32(res)
end

"""
    SqL2()

Squared Euclidean distance between two vectors quantized with [`quantize`](@ref) (four
globally-scaled 2-bit codes per byte). Since both vectors share the same global
`min`/scale, the squared difference of the raw codes is proportional to the squared
difference of the original values, so `evaluate` accumulates squared code differences
directly, unpacking each byte's four 2-bit fields with SIMD, without any per-element
dequantization.
"""
struct SqL2 <: Dist.SemiMetric
end

function Dist.evaluate(::SqL2, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Byte arrays must be the same length"))

    res, i = _u2_reduce(_u2_sqdiff, x, y)
    n = length(x)

    @inbounds while i <= n
        x_val, y_val = x[i], y[i]
        for p in 0:2:6
            d = Int((x_val >>> p) & 0x03) - Int((y_val >>> p) & 0x03)
            res += d * d
        end
        i += 1
    end

    Float32(res)
end

end
