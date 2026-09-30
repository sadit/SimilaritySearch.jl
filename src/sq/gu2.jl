"""
    SQgu2

Global (database-wide) 2-bit scalar quantization: [`quantize`](@ref SQgu2.quantize) maps
every coordinate of every vector using a single shared `min`/scale pair, packing four
2-bit codes per `UInt8`, and [`SqL2`](@ref SQgu2.SqL2) compares the resulting codes directly with SIMD. Accessed as
`ScalarQuant.SQgu2.quantize`, etc.

It is the coarsest member of the global family (`SQgu2`/`SQgu4`/`SQgu8`): four codes fit
in one byte, so a `Float32` database shrinks by 16x, at the cost of only four levels per
coordinate. Codes are laid out low bits first (coordinate `4k+1` in bits `0:1`, `4k+2` in
bits `2:3`, and so on), exactly like [`SQu2`](@ref ScalarQuant.SQu2)'s per-column variant.
"""
module SQgu2

export quantize, quantize!, SqL2

using ..ScalarQuant: getminbatch, sqglobalscale, sqrange, Dist, @BATCHES, packcodes!, sqdiffcodes
using Statistics: quantile
using SIMD

"""
    quantize(X::AbstractMatrix; minmax=nothing, quant=nothing, samplesize=0)

Scalar-quantizes every entry of `X` to 2 bits using a single, global pair of
dequantization parameters shared by all columns, unlike [`SQu2`](@ref ScalarQuant.SQu2)'s
`quantize` which computes an independent `min`/scale per column. As with
[`SQgu4`](@ref ScalarQuant.SQgu4)/[`SQgu8`](@ref ScalarQuant.SQgu8), this is useful when
the columns of `X` share a comparable value range, since a single global range provides
enough precision while being cheaper to compute and store.

Codes are packed four per `UInt8` (low bits first), so the returned matrix has
`cld(size(X, 1), 4)` rows. Packing four dimensions into a single byte, combined with a
*global* (rather than per-column) `min`/scale, lets [`SqL2`](@ref) and
[`SqL2`](@ref) operates directly on the packed codes with SIMD, without any
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

julia> Q = ScalarQuant.SQgu2.quantize(X; minmax=(0f0, 1f0));  # explicit range

julia> size(Q), eltype(Q)  # (2, 1000), UInt8
```
"""
function quantize(X::AbstractMatrix;
        minmax=nothing,
        quant=nothing,
        samplesize=0
    )
    m, n = size(X)
    Q = Matrix{UInt8}(undef, cld(m, 4), n)
    min, max = sqrange(vec(X), 3; minmax, quant, samplesize)
    c = sqglobalscale(3, min, max)
    min = Float32(min)

    minbatch = getminbatch(n)
    @BATCHES minbatch for i in 1:n
        packcodes!(Val(2), view(Q, :, i), view(X, :, i), min, c)
    end

    Q
end

"""
    quantize(v::AbstractVector; minmax=nothing, quant=nothing, samplesize=0)

Scalar-quantizes a single vector `v` to 2 bits, using the same global scheme as
[`quantize(X::AbstractMatrix)`](@ref), producing a `Vector{UInt8}` (four codes per byte)
of length `cld(length(v), 4)`, instead of a `Matrix{UInt8}`.

!!! warning
    To produce codes that are meaningfully comparable (e.g. for distance computations
    with [`SqL2`](@ref)) to those of an already-quantized dataset,
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
- `quant`: a fixed lower/upper quantile pair to use as the range instead of searching for
  it; `nothing` (the default) runs [`sqautorange`](@ref ScalarQuant.sqautorange). Used when `minmax`
  is not given
- `samplesize`: the number of entries sampled (with replacement) from `v` to estimate the
  quantiles; when `0` (the default) it is set to `ceil(Int, length(v)^0.5)`
"""
function quantize(v::AbstractVector;
        minmax=nothing,
        quant=nothing,
        samplesize=0
    )
    vout = Vector{UInt8}(undef, cld(length(v), 4))
    min, max = sqrange(v, 3; minmax, quant, samplesize)
    packcodes!(Val(2), vout, v, Float32(min), sqglobalscale(3, min, max))
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
    packcodes!(Val(2), vout, v, Float32(min), sqglobalscale(3, min, max))
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
    Float32(sqdiffcodes(Val(2), x, y))
end

end
