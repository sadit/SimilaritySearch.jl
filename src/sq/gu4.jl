"""
    SQgu4

Global (database-wide) 4-bit scalar quantization: [`quantize`](@ref SQgu4.quantize) maps
every coordinate of every vector using a single shared `min`/scale pair, packing two
4-bit codes per `UInt8`, and [`NormCosine`](@ref SQgu4.NormCosine)/[`SqL2`](@ref
SQgu4.SqL2) compare the resulting codes directly with SIMD. Accessed as
`ScalarQuant.SQgu4.quantize`, etc.
"""
module SQgu4

export quantize, quantize!, NormCosine, SqL2

using ..ScalarQuant: getminbatch, sqglobalscale, Dist, @BATCHES
using Statistics: quantile
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

    c = sqglobalscale(15, min, max)
    min = Float32(min)

    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
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

    c = sqglobalscale(15, min, max)
    min = Float32(min)
    quant_global_u4!(vout, v, min, c)

    vout
end



"""
    quantize!(vout::AbstractVector{UInt8}, v::AbstractVector, minmax) -> vout

In-place, allocation-free [`quantize`](@ref) of a single vector into a caller-provided
`vout` of length `cld(length(v), 2)`, using the explicit `(min, max)` range `minmax`.
Intended for encoding loops that reuse their output buffer (see
`Projections.quantsketch`); the range is never estimated here, precisely so every vector
encoded through it stays comparable.
"""
function quantize!(vout::AbstractVector{UInt8}, v::AbstractVector, minmax)
    min, max = minmax
    quant_global_u4!(vout, v, Float32(min), sqglobalscale(15, min, max))
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
    # One 32-lane accumulator, not eight 16-lane ones: the unrolled shape it replaces was
    # measurably slower (40.6ns against 16.4ns at 256 codes) -- with two nibble chains per
    # unrolled block the loop keeps far more live vector state than the FMA latency it hides.
    # A lane accumulates at most 2*225 per step, so Int32 cannot overflow at any usable size.
    mask = 0x0f
    n = length(x)
    i = 1
    acc = zero(Vec{32, UInt32})

    @inbounds while i + 31 <= n
        vx = vload(Vec{32, UInt8}, x, i)
        vy = vload(Vec{32, UInt8}, y, i)
        acc = muladd(convert(Vec{32, UInt32}, vx & mask), convert(Vec{32, UInt32}, vy & mask), acc)
        acc = muladd(convert(Vec{32, UInt32}, vx >>> 4),  convert(Vec{32, UInt32}, vy >>> 4),  acc)
        i += 32
    end

    res = Int(sum(acc))

    # --- PHASE 2: Single SIMD Loop Cleanup ---
    @inbounds while i + 15 <= n
        vx = vload(Vec{16, UInt8}, x, i)
        vy = vload(Vec{16, UInt8}, y, i)
        res += Int(sum(convert(Vec{16, UInt32}, vx & mask) * convert(Vec{16, UInt32}, vy & mask)))
        res += Int(sum(convert(Vec{16, UInt32}, vx >>> 4) * convert(Vec{16, UInt32}, vy >>> 4)))
        i += 16
    end

    # --- PHASE 2b: Half-Width SIMD Cleanup (chunks of 8) ---
    # Phase 2 needs a full `N = 16` bytes, so an 8..15 byte remainder went to the scalar
    # loop below, which unpacks two nibbles per byte by hand. It costs more than simply
    # having more data to vectorize: scanning 65536 vectors, 24 bytes took 22.3ns against
    # 12.8ns for 32. One pass suffices, since phase 2 leaves at most 15 bytes.
    @inbounds if i + 7 <= n
        vx = vload(Vec{8, UInt8}, x, i)
        vy = vload(Vec{8, UInt8}, y, i)
        res += Int(sum(convert(Vec{8, UInt32}, vx & mask) * convert(Vec{8, UInt32}, vy & mask)))
        res += Int(sum(convert(Vec{8, UInt32}, vx >>> 4) * convert(Vec{8, UInt32}, vy >>> 4)))
        i += 8
    end

    # --- PHASE 3: Scalar Tail Cleanup ---
    @inbounds while i <= n
        xv, yv = x[i], y[i]
        res += Int(xv & mask) * Int(yv & mask)
        res += Int(xv >>> 4) * Int(yv >>> 4)
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
    # See the note in NormCosine above: one 32-lane accumulator instead of eight 16-lane ones.
    mask = 0x0f
    n = length(x)
    i = 1
    acc = zero(Vec{32, Int32})

    @inbounds while i + 31 <= n
        vx = vload(Vec{32, UInt8}, x, i)
        vy = vload(Vec{32, UInt8}, y, i)
        dlo = convert(Vec{32, Int32}, vx & mask) - convert(Vec{32, Int32}, vy & mask)
        dhi = convert(Vec{32, Int32}, vx >>> 4) - convert(Vec{32, Int32}, vy >>> 4)
        acc = muladd(dlo, dlo, muladd(dhi, dhi, acc))
        i += 32
    end

    res = Int(sum(acc))

    # --- PHASE 2: Single SIMD Loop Cleanup ---
    @inbounds while i + 15 <= n
        vx = vload(Vec{16, UInt8}, x, i)
        vy = vload(Vec{16, UInt8}, y, i)

        diff_low  = convert(Vec{16, Int32}, vx & mask) - convert(Vec{16, Int32}, vy & mask)
        diff_high = convert(Vec{16, Int32}, vx >>> 4) - convert(Vec{16, Int32}, vy >>> 4)
        res += Int(sum(diff_low * diff_low)) + Int(sum(diff_high * diff_high))
        i += 16
    end

    # --- PHASE 2b: Half-Width SIMD Cleanup (chunks of 8) ---
    # See the note in NormCosine above: phase 2 needs a full 16 bytes and leaves at most
    # 15, so one 8-lane pass covers the only remainder worth vectorizing.
    @inbounds if i + 7 <= n
        vx = vload(Vec{8, UInt8}, x, i)
        vy = vload(Vec{8, UInt8}, y, i)

        d_low  = convert(Vec{8, Int32}, vx & mask) - convert(Vec{8, Int32}, vy & mask)
        d_high = convert(Vec{8, Int32}, vx >>> 4) - convert(Vec{8, Int32}, vy >>> 4)
        res += Int(sum(d_low * d_low)) + Int(sum(d_high * d_high))
        i += 8
    end

    # --- PHASE 3: Scalar Tail Cleanup ---
    @inbounds while i <= n
        # Unpack the tail byte manually
        x_val, y_val = x[i], y[i]

        x_low, y_low   = Int(x_val & mask), Int(y_val & mask)
        x_high, y_high = Int(x_val >>> 4), Int(y_val >>> 4)

        diff_low  = x_low - y_low
        diff_high = x_high - y_high

        res += (diff_low * diff_low) + (diff_high * diff_high)
        i += 1
    end

    convert(Float32, res)
end

end