"""
    SQgu4

Global (database-wide) 4-bit scalar quantization: [`quantize`](@ref SQgu4.quantize) maps
every coordinate of every vector using a single shared `min`/scale pair, packing two
4-bit codes per `UInt8`, and [`NormCosine`](@ref SQgu4.NormCosine)/[`SqL2`](@ref
SQgu4.SqL2) compare the resulting codes directly with SIMD. Accessed as
`ScalarQuant.SQgu4.quantize`, etc.
"""
module SQgu4

export quantize, NormCosine, SqL2

using ..ScalarQuant: getminbatch, Dist
using Statistics: quantile
using Polyester
using SIMD

"Quantizes `v` into `vout` (two 4-bit codes packed per `UInt8`) using the global `min`/scale `c`; returns `vout`."
function quant_global_u4!(vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    m = length(v)
    k = 1
    j = 1
    @inbounds while j <= m
        a = round((v[j] - min) * c; digits=0)
        a = UInt8(clamp(a, 0, 15))
        b = zero(UInt8)
        if j+1 <= m
            b = let b = round((v[j+1] - min) * c; digits=0)
                UInt8(clamp(b, 0, 15))
            end
        end

        vout[k] = a | (b << 4)
        j += 2
        k += 1
    end

    vout
end

"""
    quantize(X::AbstractMatrix; minmax=nothing, quant=[0.025, 0.975], samplesize=0)

Scalar-quantizes every entry of `X` to 4 bits using a single, global pair of
dequantization parameters shared by all columns, unlike [`SQu4`](@ref ScalarQuant.SQu4)'s `quantize`
which computes an independent `min`/scale per column. As with [`SQgu8`](@ref ScalarQuant.SQgu8)'s
`quantize`, this is useful when the columns of `X` share a comparable value range, since
a single global range provides enough precision while being cheaper to compute and store.

Codes are packed two per `UInt8` (low nibble, high nibble), exactly like [`SQu4`](@ref ScalarQuant.SQu4)'s,
so the returned matrix has `ceil(Int, size(X, 1) / 2)` rows. Packing pairs of dimensions
into a single byte, combined with a *global* (rather than per-column) `min`/scale, lets
[`SqL2`](@ref) and [`NormCosine`](@ref) operate directly on the packed codes
with SIMD, without any per-element dequantization: since every column shares the same
affine mapping, comparisons and (squared) differences computed in code space are already
proportional to the ones in the original space.

The global `[min, max]` range is estimated by sampling entries of `X` and taking
quantiles of the sample (to be robust to outliers), unless it is provided explicitly via
`minmax`. Every entry `x` is then mapped as `round(clamp((x - min) * c, 0, 15))` with
`c = 15 / (max - min + 1e-6)`.

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

julia> Q = ScalarQuant.SQgu4.quantize(X; minmax=(0f0, 1f0));  # explicit range

julia> size(Q), eltype(Q)  # (4, 1000), UInt8
```
"""
function quantize(X::AbstractMatrix;
        minmax=nothing,
        quant=[0.025, 0.975],
        samplesize=0
    )
    m, n = size(X)
    Q = Matrix{UInt8}(undef, ceil(Int, m / 2), n)

    min, max = if minmax === nothing
        let  V = vec(X),
             n = length(V),
             samplesize = samplesize === 0 ? ceil(Int, n^0.5) : samplesize
             S = rand(V, samplesize)
            quantile(S, quant)
        end
    else
        minmax
    end

    c = Float32(15 / (max - min + 1e-6))
    min = Float32(min)

    minbatch = getminbatch(n)
    @batch per=thread minbatch=minbatch for i in 1:n
        quant_global_u4!(view(Q, :, i), view(X, :, i), min, c)
    end

    Q
end

"""
    quantize(v::AbstractVector; minmax=nothing, quant=[0.025, 0.975], samplesize=0)

Scalar-quantizes a single vector `v` to 4 bits, using the same global scheme as
[`quantize(X::AbstractMatrix)`](@ref), producing a `Vector{UInt8}` (nibble-packed, two
codes per byte) of length `ceil(Int, length(v) / 2)`, instead of a `Matrix{UInt8}`.

!!! warning
    To produce codes that are meaningfully comparable (e.g. for distance computations
    with [`NormCosine`](@ref)/[`SqL2`](@ref)) to those of an already-quantized dataset,
    `minmax` **must** be the exact same `(min, max)` pair used to quantize that dataset
    (e.g., a query vector must be quantized with the dataset's `minmax`, not its own).
    Leaving `minmax=nothing` here estimates a *new*, independent range from `v` alone,
    which will in general **not** match the range used for a previously-quantized
    dataset, silently producing incompatible, meaningless codes. Since
    [`quantize(X::AbstractMatrix)`](@ref) does not return the `(min, max)` it used
    internally unless it was given explicitly, callers that need to quantize additional
    vectors later (e.g. queries) should always pass `minmax` explicitly when building the
    dataset too, so that the same pair can be reused here.

# Arguments
- `v`: the vector to quantize
- `minmax`: an optional `(min, max)` tuple giving the value range to use; when `nothing`
  (the default) the range is estimated from a random sample of `v`'s entries using
  `quant`. **Must match the dataset's `minmax`** if `v` is to be compared against an
  existing quantized dataset.
- `quant`: the lower and upper quantiles (of the sampled entries of `v`) used to estimate
  `min` and `max` when `minmax` is not given
- `samplesize`: the number of entries sampled (with replacement) from `v` to estimate the
  quantiles; when `0` (the default) it is set to `ceil(Int, length(v)^0.5)`

# Examples

```julia
julia> using SimilaritySearch

julia> minmax = (0f0, 1f0);

julia> X = rand(Float32, 8, 1000);

julia> Q = ScalarQuant.SQgu4.quantize(X; minmax);  # dataset, using an explicit range

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQgu4.quantize(q; minmax);  # query, using the *same* range

julia> length(qv), eltype(qv)  # (4, UInt8)
```
"""
function quantize(v::AbstractVector;
        minmax=nothing,
        quant=[0.025, 0.975],
        samplesize=0
    )
    m = length(v)
    vout = Vector{UInt8}(undef, ceil(Int, m / 2))

    min, max = if minmax === nothing
        let samplesize = samplesize === 0 ? ceil(Int, m^0.5) : samplesize
            S = rand(v, samplesize)
            quantile(S, quant)
        end
    else
        minmax
    end

    c = Float32(15 / (max - min + 1e-6))
    min = Float32(min)
    quant_global_u4!(vout, v, min, c)

    vout
end


### the following SIMD kernels follow the same unroll/accumulate scheme as gu8.jl,
### but each `UInt8` holds two packed 4-bit codes (low nibble / high nibble) that must be
### unpacked before being combined

"""
    NormCosine()

Dissimilarity between two vectors quantized with [`quantize`](@ref) (nibble-packed,
globally-scaled 4-bit codes), computed as the negative dot product of the raw packed
codes. Since both vectors share the same global `min`/scale, the dot product of codes is
an affine, order-preserving proxy of the dot product of the original (typically
pre-normalized) vectors, so no per-element dequantization is needed. `evaluate` unpacks
each byte into its low and high nibble and accumulates their products with SIMD.
"""
struct NormCosine <: Dist.SemiMetric
end

function Dist.evaluate(::NormCosine, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Byte arrays must be the same length"))

    # N=16: each byte expands into two 32-bit lanes (low + high nibble), so N=16 keeps
    # the 4 unrolled chunks (8 accumulators) from spilling out of the SIMD register file.
    N = 16
    UNROLL = 4
    CHUNK = N * UNROLL # 64 bytes (128 dimensions) per iteration

    acc1_low  = zero(Vec{N, UInt32})
    acc1_high = zero(Vec{N, UInt32})
    acc2_low  = zero(Vec{N, UInt32})
    acc2_high = zero(Vec{N, UInt32})
    acc3_low  = zero(Vec{N, UInt32})
    acc3_high = zero(Vec{N, UInt32})
    acc4_low  = zero(Vec{N, UInt32})
    acc4_high = zero(Vec{N, UInt32})

    n = length(x)
    limit_unrolled = n - CHUNK + 1
    i = 1
    mask = 0x0f

    # --- PHASE 1: The Unrolled Loop ---
    @inbounds while i <= limit_unrolled
        vx1 = vload(Vec{N, UInt8}, x, i)
        vy1 = vload(Vec{N, UInt8}, y, i)
        acc1_low  = muladd(convert(Vec{N, UInt32}, vx1 & mask),  convert(Vec{N, UInt32}, vy1 & mask),  acc1_low)
        acc1_high = muladd(convert(Vec{N, UInt32}, vx1 >>> 4),   convert(Vec{N, UInt32}, vy1 >>> 4),   acc1_high)

        vx2 = vload(Vec{N, UInt8}, x, i + N)
        vy2 = vload(Vec{N, UInt8}, y, i + N)
        acc2_low  = muladd(convert(Vec{N, UInt32}, vx2 & mask),  convert(Vec{N, UInt32}, vy2 & mask),  acc2_low)
        acc2_high = muladd(convert(Vec{N, UInt32}, vx2 >>> 4),   convert(Vec{N, UInt32}, vy2 >>> 4),   acc2_high)

        vx3 = vload(Vec{N, UInt8}, x, i + 2N)
        vy3 = vload(Vec{N, UInt8}, y, i + 2N)
        acc3_low  = muladd(convert(Vec{N, UInt32}, vx3 & mask),  convert(Vec{N, UInt32}, vy3 & mask),  acc3_low)
        acc3_high = muladd(convert(Vec{N, UInt32}, vx3 >>> 4),   convert(Vec{N, UInt32}, vy3 >>> 4),   acc3_high)

        vx4 = vload(Vec{N, UInt8}, x, i + 3N)
        vy4 = vload(Vec{N, UInt8}, y, i + 3N)
        acc4_low  = muladd(convert(Vec{N, UInt32}, vx4 & mask),  convert(Vec{N, UInt32}, vy4 & mask),  acc4_low)
        acc4_high = muladd(convert(Vec{N, UInt32}, vx4 >>> 4),   convert(Vec{N, UInt32}, vy4 >>> 4),   acc4_high)

        i += CHUNK
    end

    acc_total = acc1_low + acc1_high + acc2_low + acc2_high +
                acc3_low + acc3_high + acc4_low + acc4_high

    # --- PHASE 2: Single SIMD Loop Cleanup ---
    limit_single = n - N + 1
    @inbounds while i <= limit_single
        vx = vload(Vec{N, UInt8}, x, i)
        vy = vload(Vec{N, UInt8}, y, i)

        acc_total = muladd(convert(Vec{N, UInt32}, vx & mask), convert(Vec{N, UInt32}, vy & mask), acc_total)
        acc_total = muladd(convert(Vec{N, UInt32}, vx >>> 4),  convert(Vec{N, UInt32}, vy >>> 4),  acc_total)
        i += N
    end

    res = sum(acc_total)

    # --- PHASE 3: Scalar Tail Cleanup ---
    @inbounds while i <= n
        xv, yv = x[i], y[i]
        res += UInt32(xv & mask) * UInt32(yv & mask)
        res += UInt32(xv >>> 4) * UInt32(yv >>> 4)
        i += 1
    end

    -Float32(res)
end

"""
    SqL2()

Squared Euclidean distance between two vectors quantized with [`quantize`](@ref)
(nibble-packed, globally-scaled 4-bit codes). Since both vectors share the same global
`min`/scale, the squared difference of the raw codes is proportional to the squared
difference of the original values, so `evaluate` accumulates squared code differences
directly, unpacking each byte's low and high nibble with SIMD, without any
per-element dequantization.
"""
struct SqL2 <: Dist.SemiMetric
end

function Dist.evaluate(::SqL2, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Byte arrays must be the same length"))

    # We use N=16 here instead of 32.
    # Why? Because every 1 byte splits into TWO 32-bit accumulators.
    # N=16 prevents "register spilling" on AVX2 architectures, keeping everything in the CPU's fast registers.
    N = 16
    UNROLL = 4
    CHUNK = N * UNROLL # 64 bytes (128 dimensions) per iteration

    # We need 8 accumulators total: 4 for the lower nibbles, 4 for the upper nibbles
    acc1_low  = zero(Vec{N, Int32})
    acc1_high = zero(Vec{N, Int32})
    acc2_low  = zero(Vec{N, Int32})
    acc2_high = zero(Vec{N, Int32})
    acc3_low  = zero(Vec{N, Int32})
    acc3_high = zero(Vec{N, Int32})
    acc4_low  = zero(Vec{N, Int32})
    acc4_high = zero(Vec{N, Int32})

    n = length(x)
    limit_unrolled = n - CHUNK + 1
    i = 1

    # Mask to isolate the bottom 4 bits (00001111 in binary)
    mask = 0x0f

    # --- PHASE 1: The Unrolled Loop ---
    @inbounds while i <= limit_unrolled
        # Chunk 1
        vx1 = vload(Vec{N, UInt8}, x, i)
        vy1 = vload(Vec{N, UInt8}, y, i)

        # Unpack lower nibbles (bits 0-3) and widen
        diff1_low = convert(Vec{N, Int32}, vx1 & mask) - convert(Vec{N, Int32}, vy1 & mask)
        acc1_low  = muladd(diff1_low, diff1_low, acc1_low)

        # Unpack upper nibbles (bits 4-7) by logical right-shift and widen
        diff1_high = convert(Vec{N, Int32}, vx1 >>> 4) - convert(Vec{N, Int32}, vy1 >>> 4)
        acc1_high  = muladd(diff1_high, diff1_high, acc1_high)

        # Chunk 2
        vx2 = vload(Vec{N, UInt8}, x, i + N)
        vy2 = vload(Vec{N, UInt8}, y, i + N)
        diff2_low  = convert(Vec{N, Int32}, vx2 & mask) - convert(Vec{N, Int32}, vy2 & mask)
        acc2_low   = muladd(diff2_low, diff2_low, acc2_low)
        diff2_high = convert(Vec{N, Int32}, vx2 >>> 4) - convert(Vec{N, Int32}, vy2 >>> 4)
        acc2_high  = muladd(diff2_high, diff2_high, acc2_high)

        # Chunk 3
        vx3 = vload(Vec{N, UInt8}, x, i + 2N)
        vy3 = vload(Vec{N, UInt8}, y, i + 2N)
        diff3_low  = convert(Vec{N, Int32}, vx3 & mask) - convert(Vec{N, Int32}, vy3 & mask)
        acc3_low   = muladd(diff3_low, diff3_low, acc3_low)
        diff3_high = convert(Vec{N, Int32}, vx3 >>> 4) - convert(Vec{N, Int32}, vy3 >>> 4)
        acc3_high  = muladd(diff3_high, diff3_high, acc3_high)

        # Chunk 4
        vx4 = vload(Vec{N, UInt8}, x, i + 3N)
        vy4 = vload(Vec{N, UInt8}, y, i + 3N)
        diff4_low  = convert(Vec{N, Int32}, vx4 & mask) - convert(Vec{N, Int32}, vy4 & mask)
        acc4_low   = muladd(diff4_low, diff4_low, acc4_low)
        diff4_high = convert(Vec{N, Int32}, vx4 >>> 4) - convert(Vec{N, Int32}, vy4 >>> 4)
        acc4_high  = muladd(diff4_high, diff4_high, acc4_high)

        i += CHUNK
    end

    # Combine all 8 accumulators into a single running total
    acc_total = acc1_low + acc1_high +
                acc2_low + acc2_high +
                acc3_low + acc3_high +
                acc4_low + acc4_high

    # --- PHASE 2: Single SIMD Loop Cleanup ---
    limit_single = n - N + 1
    @inbounds while i <= limit_single
        vx = vload(Vec{N, UInt8}, x, i)
        vy = vload(Vec{N, UInt8}, y, i)

        diff_low  = convert(Vec{N, Int32}, vx & mask) - convert(Vec{N, Int32}, vy & mask)
        acc_total = muladd(diff_low, diff_low, acc_total)

        diff_high = convert(Vec{N, Int32}, vx >>> 4) - convert(Vec{N, Int32}, vy >>> 4)
        acc_total = muladd(diff_high, diff_high, acc_total)

        i += N
    end

    res = sum(acc_total)

    # --- PHASE 3: Scalar Tail Cleanup ---
    @inbounds while i <= n
        # Unpack the tail byte manually
        x_val, y_val = x[i], y[i]

        x_low, y_low   = Int32(x_val & mask), Int32(y_val & mask)
        x_high, y_high = Int32(x_val >>> 4), Int32(y_val >>> 4)

        diff_low  = x_low - y_low
        diff_high = x_high - y_high

        res += (diff_low * diff_low) + (diff_high * diff_high)
        i += 1
    end

    convert(Float32, res)
end

end