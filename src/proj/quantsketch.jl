# This file is a part of SimilaritySearch.jl

export QuantSketch, quantsketch, sketchvalues, sketchvalues!, sketchbits, sketchsize

"""
    sketchvalues!(out::AbstractVector{Float32}, model, obj) -> out
    sketchvalues(model, obj) -> Vector{Float32}

The real-valued vector a sketch model produces for `obj` *before* it is discretized --
the quantity [`bitsketch`](@ref) reduces to one sign bit per component and
[`QuantSketch`](@ref) keeps at 2, 4 or 8 bits instead. `out` must have length
`outdim(model)`.

Two families of models are supported, with a deliberately shared sign convention (a
component is `>= 0` exactly when [`bitsketch`](@ref) would set its bit to `1`):

- **rotations/projections** ([`RandomProjections`](@ref), [`HadamardProjection`](@ref),
  [`PCAProjection`](@ref)): the projected coordinates themselves, i.e. `transform!`. The
  magnitude says how far the object sits from the hyperplane normal to that direction.
- **metric hyperplanes** ([`DistantHyperplanes`](@ref),
  [`AnchoredDistantHyperplanes`](@ref), [`RandomHyperplanes`](@ref)): the *signed margin*
  of `obj` against each hyperplane `(a, b)`, `d(obj, b) - d(obj, a)`, which is positive
  when `obj` is closer to `a` (the side those models encode as bit `1`) and whose
  magnitude says how decisively so.

# Arguments
- `out`: the output vector, of length `outdim(model)`
- `model`: the projection or hyperplane model
- `obj`: the object to characterize
"""
function sketchvalues! end

sketchvalues!(out::AbstractVector, rp::RandomProjections, obj) = transform!(rp, out, obj)
sketchvalues!(out::AbstractVector, hp::HadamardProjection, obj) = transform!(hp, out, obj)
sketchvalues!(out::AbstractVector, p::PCAProjection, obj) = transform!(p, out, obj)

function sketchvalues!(out::AbstractVector, m::Union{DistantHyperplanes,AnchoredDistantHyperplanes}, obj)
    @inbounds for i in eachindex(m.H)
        a, b = m.H[i]
        out[i] = evaluate(m.dist, obj, m.C[b]) - evaluate(m.dist, obj, m.C[a])
    end

    out
end

function sketchvalues!(out::AbstractVector, m::RandomHyperplanes, obj)
    @inbounds for i in 1:outdim(m)
        out[i] = evaluate(m.dist, m.refs[2i], obj) - evaluate(m.dist, m.refs[2i-1], obj)
    end

    out
end

sketchvalues(model, obj) = sketchvalues!(Vector{Float32}(undef, outdim(model)), model, obj)

"""
    hyperplanewidths(model) -> Vector{Float32}

The separation `d(a, b)` between the two anchors of every hyperplane of a metric
hyperplane model ([`DistantHyperplanes`](@ref), [`AnchoredDistantHyperplanes`](@ref),
[`RandomHyperplanes`](@ref)). By the triangle inequality the margin of any object against
hyperplane `i` (see [`sketchvalues!`](@ref)) lies in `[-w[i], w[i]]`, so dividing by `w`
puts every component of the sketch on the same, dimensionless `[-1, 1]` scale -- which is
what makes a *single global* quantization range (the [`SQgu2`](@ref
ScalarQuant.SQgu2)/[`SQgu4`](@ref ScalarQuant.SQgu4)/[`SQgu8`](@ref ScalarQuant.SQgu8)
family, and hence their SIMD code-space distances) meaningful across hyperplanes built
from anchor pairs that are, individually, arbitrarily far apart or close together. See
[`QuantSketch`](@ref)'s `normalize` keyword.

Returns `nothing` for models that have no such natural per-component width (the
rotation/projection family, whose coordinates already share one scale).
"""
hyperplanewidths(model) = nothing

hyperplanewidths(m::Union{DistantHyperplanes,AnchoredDistantHyperplanes}) =
    Float32[evaluate(m.dist, m.C[p[1]], m.C[p[2]]) for p in m.H]

hyperplanewidths(m::RandomHyperplanes) =
    Float32[evaluate(m.dist, m.refs[2i-1], m.refs[2i]) for i in 1:outdim(m)]

"""
    QuantSketch{NBITS,MODEL}

An `NBITS`-per-component sketch encoder: it wraps any sketch `model` that
[`sketchvalues!`](@ref) understands and, instead of keeping one *sign* bit per component
the way [`bitsketch`](@ref) does, keeps an `NBITS`-wide unsigned code per component,
quantized with the global (database-wide) scalar quantizers
[`SQgu2`](@ref ScalarQuant.SQgu2)/[`SQgu4`](@ref ScalarQuant.SQgu4)/[`SQgu8`](@ref
ScalarQuant.SQgu8).

The point is that a sign bit throws away *how far* an object sits from each hyperplane:
an object hugging a hyperplane and one far across it get the same bit, so Hamming
distance cannot tell a marginal disagreement from a decisive one. An `NBITS` code keeps
that magnitude, and the sketches are then compared with the squared-Euclidean distance
over the packed codes ([`distance`](@ref)`(qs)`), which the `SQgu*` kernels evaluate with
SIMD directly on the codes -- no dequantization -- because a single global affine map is
shared by every component of every sketch.

`NBITS == 1` is supported and is exactly [`bitsketch`](@ref)'s encoding (`UInt64`-packed
sign bits, compared with [`Hamming`](@ref)), so a sweep over `1, 2, 4, 8` bits runs
through one uniform API instead of two.

Encode with [`quantsketch`](@ref)`(qs, obj_or_db)`, compare with
[`distance`](@ref)`(qs)`, and see [`SketchedSearch`](@ref) for the ready-made
index-and-query pipeline built on top.

# Memory

At `m = outdim(model)` components a sketch occupies `m * NBITS / 8` bytes, i.e. `NBITS`
times what the binary sketch of the same model costs. A fair comparison across widths
therefore holds *bits* fixed, not components: 1024 binary hyperplanes, 512 at 2 bits, 256
at 4 bits and 128 at 8 bits all occupy 128 bytes.

    QuantSketch(model, nbits::Int, X;
                minmax=nothing, normalize::Bool=true,
                quant=[0.025, 0.975], samplesize::Int=0)

Fits an encoder for `model` at `nbits` bits per component, estimating the global
quantization range from `X`.

# Arguments
- `model`: the sketch model; anything [`sketchvalues!`](@ref) accepts
- `nbits`: `1`, `2`, `4` or `8`; anything else raises `ArgumentError`
- `X`: the dataset the range is estimated from -- an `AbstractDatabase` or a matrix whose
  columns are the objects. Ignored when `nbits == 1` (a sign needs no range) or when
  `minmax` is given explicitly.

# Keyword Arguments
- `minmax`: an explicit `(min, max)` global range, skipping the estimation from `X`
- `normalize`: whether to divide each component by its hyperplane's width (see
  [`hyperplanewidths`](@ref)) before quantizing, putting every component on a common
  `[-1, 1]` scale. Defaults to `true` and is a no-op for models with no such width (the
  rotation/projection family).
- `quant`: the lower/upper quantiles of the sampled values used to estimate `minmax`;
  quantiles rather than extrema so a few outliers cannot flatten the useful range
- `samplesize`: how many objects of `X` are characterized to estimate `minmax`; `0` (the
  default) picks `clamp(ceil(Int, sqrt(length(X))), 1, length(X))`

# Examples

```julia
julia> using SimilaritySearch, SimilaritySearch.Projections

julia> X = MatrixDatabase(rand(Float32, 8, 10_000));

julia> m = DistantHyperplanes(SimilaritySearch.Dist.L2(), X, 128; verbose=false);

julia> qs = QuantSketch(m, 4, X);          # 4 bits per hyperplane instead of 1

julia> B = quantsketch(qs, X);             # MatrixDatabase of packed codes

julia> distance(qs)                        # SQgu4.SqL2(), evaluated on the packed codes
```
"""
struct QuantSketch{NBITS,MODEL}
    model::MODEL
    scale::Vector{Float32}          # per-component multiplier; empty means identity
    minmax::Tuple{Float32,Float32}
end

function QuantSketch(model, nbits::Int, X;
        minmax=nothing,
        normalize::Bool=true,
        quant=[0.025, 0.975],
        samplesize::Int=0
    )
    nbits in (1, 2, 4, 8) || throw(ArgumentError("QuantSketch: nbits=$nbits must be one of 1, 2, 4 or 8"))
    w = normalize ? hyperplanewidths(model) : nothing
    scale = w === nothing ? Float32[] : Float32[1f0 / Base.max(x, eps(Float32)) for x in w]
    qs = QuantSketch{nbits,typeof(model)}(model, scale, (-1f0, 1f0))
    nbits == 1 && return qs                     # a sign bit needs no quantization range
    minmax === nothing || return QuantSketch{nbits,typeof(model)}(model, scale, (Float32(minmax[1]), Float32(minmax[2])))

    db = _asdatabase(X)
    n = length(db)
    n > 0 || throw(ArgumentError("QuantSketch: cannot estimate the quantization range from an empty dataset"))
    ss = samplesize == 0 ? clamp(ceil(Int, sqrt(n)), 1, n) : Base.min(samplesize, n)
    m = outdim(model)
    V = Vector{Float32}(undef, ss * m)

    minbatch = getminbatch(ss)
    @BATCHES minbatch begin
        @BEGINBATCH
            buf = Vector{Float32}(undef, m)
        @LOOP for i in 1:ss
            _qs_values!(qs, buf, db[rand(1:n)])
            copyto!(V, (i - 1) * m + 1, buf, 1, m)
        end
    end

    lo, hi = quantile(V, quant)
    QuantSketch{nbits,typeof(model)}(model, scale, (Float32(lo), Float32(hi)))
end

_asdatabase(X::AbstractDatabase) = X
_asdatabase(X::AbstractMatrix) = MatrixDatabase(X)

"""
    sketchbits(qs::QuantSketch)

The number of bits `qs` spends on each component (`1`, `2`, `4` or `8`).
"""
sketchbits(::QuantSketch{NBITS}) where NBITS = NBITS

"""
    outdim(qs::QuantSketch)

The number of components (hyperplanes/directions) of the underlying model, i.e. how many
codes each sketch holds -- not how many machine words it occupies (see [`sketchsize`](@ref)).
"""
outdim(qs::QuantSketch) = outdim(qs.model)

"""
    sketchsize(qs::QuantSketch)

The length, in storage words, of one sketch produced by `qs`: `cld(outdim(qs), 64)`
`UInt64` words at 1 bit, and `cld(outdim(qs), 8 ÷ nbits)` `UInt8` bytes otherwise.
"""
sketchsize(qs::QuantSketch{1}) = cld(outdim(qs), 64)
sketchsize(qs::QuantSketch{2}) = cld(outdim(qs), 4)
sketchsize(qs::QuantSketch{4}) = cld(outdim(qs), 2)
sketchsize(qs::QuantSketch{8}) = outdim(qs)

_codetype(::QuantSketch{1}) = UInt64
_codetype(::QuantSketch) = UInt8

"""
    distance(qs::QuantSketch)

The distance function the sketches produced by `qs` must be compared with: [`Hamming`](@ref)
at 1 bit, and the matching `SQgu*.SqL2` (squared Euclidean over the packed codes,
evaluated with SIMD without dequantizing) at 2, 4 and 8 bits.
"""
distance(::QuantSketch{1}) = Hamming()
distance(::QuantSketch{2}) = SQgu2.SqL2()
distance(::QuantSketch{4}) = SQgu4.SqL2()
distance(::QuantSketch{8}) = SQgu8.SqL2()

_qs_pack!(qs::QuantSketch{1}, out, vals) = packsigns!(out, vals)
_qs_pack!(qs::QuantSketch{2}, out, vals) = SQgu2.quantize!(out, vals, qs.minmax)
_qs_pack!(qs::QuantSketch{4}, out, vals) = SQgu4.quantize!(out, vals, qs.minmax)
_qs_pack!(qs::QuantSketch{8}, out, vals) = SQgu8.quantize!(out, vals, qs.minmax)

function _qs_values!(qs::QuantSketch, out::AbstractVector{Float32}, obj)
    sketchvalues!(out, qs.model, obj)
    s = qs.scale
    if length(s) == length(out)
        @inbounds @simd for i in eachindex(out)
            out[i] *= s[i]
        end
    end

    out
end

"""
    quantsketch(qs::QuantSketch, obj) -> Vector
    quantsketch(qs::QuantSketch, X::AbstractDatabase; minbatch::Int=4) -> MatrixDatabase
    quantsketch(qs::QuantSketch, X::AbstractMatrix; minbatch::Int=4) -> Matrix

Encodes `obj` (or every object of `X`) with `qs`, returning `sketchsize(qs)`-long codes --
`UInt64` sign words at 1 bit, packed `UInt8` codes otherwise. Sketches are comparable only
against others produced by the *same* `qs` (the quantization range and, when `normalize`
is on, the per-hyperplane scales are baked into it), and must be compared with
[`distance`](@ref)`(qs)`.

The collection methods mirror [`bitsketch`](@ref)'s return conventions -- a database in,
a `MatrixDatabase` out; a matrix in, a `Matrix` out -- and never materialize the dense
`(outdim, n)` matrix of real values: each batch keeps a single `outdim`-long scratch
vector reused across the objects it encodes.

# Arguments
- `qs`: the fitted encoder
- `obj`/`X`: the object, or collection of objects, to sketch
- `minbatch`: (collection methods only) minimum number of objects processed per parallel task
"""
function quantsketch(qs::QuantSketch, obj)
    out = Vector{_codetype(qs)}(undef, sketchsize(qs))
    buf = Vector{Float32}(undef, outdim(qs))
    _qs_pack!(qs, out, _qs_values!(qs, buf, obj))
    out
end

function quantsketch(qs::QuantSketch, X::AbstractDatabase; minbatch::Int=4)
    MatrixDatabase(_quantsketch_matrix(qs, X; minbatch))
end

quantsketch(qs::QuantSketch, X::AbstractMatrix; minbatch::Int=4) =
    _quantsketch_matrix(qs, MatrixDatabase(X); minbatch)

function _quantsketch_matrix(qs::QuantSketch, X::AbstractDatabase; minbatch::Int=4)
    n = length(X)
    m = outdim(qs)
    D = Matrix{_codetype(qs)}(undef, sketchsize(qs), n)

    @BATCHES minbatch begin
        @BEGINBATCH
            buf = Vector{Float32}(undef, m)
        @LOOP for i in 1:n
            _qs_pack!(qs, view(D, :, i), _qs_values!(qs, buf, X[i]))
        end
    end

    D
end

"""
    quantsketch(model, nbits::Int, X; minbatch::Int=4, kwargs...) -> (codes, qs)

Convenience one-step form: fits a [`QuantSketch`](@ref) for `model` at `nbits` bits over
`X` and immediately encodes `X` with it, returning **both** the codes and the encoder, so
the very same encoder can sketch further objects (queries, above all) comparably. Extra
keyword arguments are forwarded to [`QuantSketch`](@ref).

# Examples

```julia
julia> using SimilaritySearch, SimilaritySearch.Projections

julia> X = MatrixDatabase(rand(Float32, 8, 10_000));

julia> m = RandomHyperplanes(SimilaritySearch.Dist.L2(), SubDatabase(X, rand(1:10_000, 256)), 128);

julia> B, qs = quantsketch(m, 4, X);

julia> bq = quantsketch(qs, X[1]);   # a query, encoded the same way
```
"""
function quantsketch(model, nbits::Int, X; minbatch::Int=4, kwargs...)
    qs = QuantSketch(model, nbits, X; kwargs...)
    quantsketch(qs, X; minbatch), qs
end
