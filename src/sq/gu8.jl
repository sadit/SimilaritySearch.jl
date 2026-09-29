"""
    SQgu8

Global (database-wide) 8-bit scalar quantization: [`quantize`](@ref SQgu8.quantize)
maps every coordinate of every vector using a single shared `min`/scale pair, and
[`SqL2`](@ref SQgu8.SqL2) compares the resulting
codes directly with SIMD. Accessed as `ScalarQuant.SQgu8.quantize`, etc.
"""
module SQgu8

export quantize, quantize!, SqL2

using ..ScalarQuant: getminbatch, sqglobalscale, sqrange, Dist, @BATCHES, packcodes!, sqdiffcodes
using Statistics: quantile
using SIMD

"""
    quantize(X::AbstractMatrix; minmax=nothing, quant=nothing, samplesize=0)

Scalar-quantizes every entry of `X` to 8 bits (`UInt8`) using a single, global pair of
dequantization parameters shared by all columns, unlike [`SQu8`](@ref ScalarQuant.SQu8)'s `quantize`
which computes an independent `min`/scale per column. This is useful, e.g., when the columns of `X` are
known to share a comparable value range and a single global range provides enough
precision while being cheaper to compute and store.

The global `[min, max]` range is estimated by sampling entries of `X` and taking
quantiles of the sample (to be robust to outliers), unless it is provided explicitly via
`minmax`. Every entry `x` is then mapped as `round(clamp((x - min) * c, 0, 255))` with
`c = 255 / (max - min + 1e-6)`.

# Arguments
- `X`: the matrix to quantize; each entry is quantized independently but using shared
  `min`/`max` values
- `minmax`: an optional `(min, max)` tuple giving the value range to use; when `nothing`
  (the default) the range is chosen from a random sample of the entries of `X` by
  [`sqautorange`](@ref ScalarQuant.sqautorange), or by `quant` when that is given
- `quant`: a fixed lower/upper quantile pair (of the sampled entries of `X`) to use as the
  range instead of searching for it. `nothing` (the default) runs
  [`sqautorange`](@ref ScalarQuant.sqautorange), which places each end of the range where it
  minimizes the error the codes would incur -- the optimum moves with the code width, so a
  fixed pair cannot be right at 2, 4 and 8 bits at once. Pass `[0.025, 0.975]` for the
  pre-search behaviour
- `samplesize`: the number of entries sampled (with replacement) from `X` to estimate the
  quantiles; when `0` (the default) it is set to `ceil(Int, length(X)^0.5)`

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> Q = ScalarQuant.SQgu8.quantize(X; minmax=(0f0, 1f0));  # explicit range

julia> size(Q), eltype(Q)  # (8, 1000), UInt8
```
"""
function quantize(X::AbstractMatrix;
        minmax=nothing,
        quant=nothing,
        samplesize=0
    )
    m, n = size(X)
    Q = Matrix{UInt8}(undef, m, n)
    
    min, max = sqrange(vec(X), 255; minmax, quant, samplesize)

    c = sqglobalscale(255, min, max)
    min = Float32(min)

    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        packcodes!(Val(8), view(Q, :, i), view(X, :, i), min, c)
    end

    Q
end

"""
    quantize(v::AbstractVector; minmax=nothing, quant=nothing, samplesize=0)

Scalar-quantizes a single vector `v` to 8 bits (`UInt8`), using the same global scheme as
[`quantize(X::AbstractMatrix)`](@ref), producing a `Vector{UInt8}` (one code per
coordinate) instead of a `Matrix{UInt8}`.

!!! warning
    To produce codes that are meaningfully comparable (e.g. for distance computations
    with [`SqL2`](@ref)) to those of an already-quantized dataset,
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
- `quant`: a fixed lower/upper quantile pair (of the sampled entries of `v`) to use as the
  range instead of searching for it. `nothing` (the default) runs
  [`sqautorange`](@ref ScalarQuant.sqautorange), which places each end of the range where it
  minimizes the error the codes would incur -- the optimum moves with the code width, so a
  fixed pair cannot be right at 2, 4 and 8 bits at once. Pass `[0.025, 0.975]` for the
  pre-search behaviour
- `samplesize`: the number of entries sampled (with replacement) from `v` to estimate the
  quantiles; when `0` (the default) it is set to `ceil(Int, length(v)^0.5)`

# Examples

```julia
julia> using SimilaritySearch

julia> minmax = (0f0, 1f0);

julia> X = rand(Float32, 8, 1000);

julia> Q = ScalarQuant.SQgu8.quantize(X; minmax);  # dataset, using an explicit range

julia> q = rand(Float32, 8);

julia> qv = ScalarQuant.SQgu8.quantize(q; minmax);  # query, using the *same* range

julia> length(qv), eltype(qv)  # (8, UInt8)
```
"""
function quantize(v::AbstractVector;
        minmax=nothing,
        quant=nothing,
        samplesize=0
    )
    m = length(v)
    vout = Vector{UInt8}(undef, m)

    min, max = sqrange(v, 255; minmax, quant, samplesize)

    c = sqglobalscale(255, min, max)
    min = Float32(min)
    packcodes!(Val(8), vout, v, min, c)

    vout
end


"""
    quantize!(vout::AbstractVector{UInt8}, v::AbstractVector, minmax) -> vout

In-place, allocation-free [`quantize`](@ref) of a single vector into a caller-provided
`vout` of length `length(v)`, using the explicit `(min, max)` range `minmax`.
Intended for encoding loops that reuse their output buffer (see
`Projections.quantsketch`); the range is never estimated here, precisely so every vector
encoded through it stays comparable.
"""
function quantize!(vout::AbstractVector{UInt8}, v::AbstractVector, minmax)
    min, max = minmax
    packcodes!(Val(8), vout, v, Float32(min), sqglobalscale(255, min, max))
end


"""
    SqL2()

Squared Euclidean distance between two vectors quantized with [`quantize`](@ref)
(globally-scaled 8-bit codes). Since both vectors share the same global `min`/scale, the
squared difference of the raw codes is proportional to the squared difference of the
original values, so `evaluate` accumulates squared code differences directly with SIMD,
widening each `UInt8` code to `Int32` to safely represent negative differences, without
any per-element dequantization.
"""
struct SqL2 <: Dist.SemiMetric
end

function Dist.evaluate(::SqL2, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Vectors must be the same length"))
    Float32(sqdiffcodes(Val(8), x, y))
end

end