# This file is a part of SimilaritySearch.jl

export SQVec, RangePolicy, SymmetricRange, ExtremaRange

"""
    RangePolicy

How the per-vector family places a vector's own quantization range `[min, max]` before mapping
it onto the codes `0:levels(B)`. Two policies:

- [`SymmetricRange`](@ref) **(the default)**: `mean ± k·σ` of the vector's coordinates, with `k`
  searched per vector (coarse then fine over `0.5:0.05:4.5`) to minimize that vector's own
  [`sqdistortion`](@ref), or fixed with `SymmetricRange(k=2.5)`. The coordinates beyond the range
  saturate; the search decides how many, and the bulk gets the levels.
- [`ExtremaRange`](@ref): the vector's own extrema, the rule up to 1.6.3. Nothing is clipped, and
  one outlying coordinate stretches the range and coarsens every other code of that vector.

Measured on yahooaq (100K vectors, 384 dimensions, exhaustive recall@10 against the exact
answer, symmetric / asymmetric kernels): extrema 0.544 / 0.616 at 2 bits, 0.913 / 0.931 at 4,
0.9935 / 0.9947 at 8; the searched symmetric range 0.760 / 0.822, 0.924 / 0.940 and 0.9931 /
0.9953, ahead of the global family's `sqautorange` at 2 and 4 bits. With 3 or 15 levels,
stretching the range to the farthest coordinate leaves the bulk of the vector on one or two
codes; at 8 bits the extrema of 384 near-Gaussian coordinates sit near ±2.9σ, inside the
width that search chooses, and the two agree.
"""
abstract type RangePolicy end

"The vector's own extrema as its range: nothing saturates (the rule up to 1.6.3)."
struct ExtremaRange <: RangePolicy end

"""
    SymmetricRange(; k=0, search=:fast)

`mean ± k·σ` of the vector's coordinates as its range; `k == 0` searches the factor per vector.
`search=:fast` (the default) reads the factor off a 64-bin histogram of the coordinates'
deviations in one pass and scores every candidate in constant time, with the rounding error
modelled as `step²/12` and the saturation error summed exactly from the histogram -- about
2.5 µs per vector at 384 dimensions, the cost of taking the extrema. `search=:refined` then
evaluates the real grid's [`sqdistortion`](@ref) at that factor and its two neighbouring
candidates (three passes over the vector, ~6 µs). `search=:exact` evaluates it for every
candidate (28 passes, ~32 µs), the reference the other two are validated against: on yahooaq
(100K vectors, exhaustive recall@10) the fast search gives up 0.02 of the exact one's recall at
2 bits and 0.005 at 4. See [`RangePolicy`](@ref).
"""
struct SymmetricRange <: RangePolicy
    k::Float32
    search::Symbol
end
SymmetricRange(; k=0, search::Symbol=:fast) = (search in (:fast, :refined, :exact) || throw(ArgumentError("SymmetricRange: search must be :fast, :refined or :exact")); SymmetricRange(Float32(k), search))

"The per-vector family's default range policy."
const DEFAULT_RANGE = SymmetricRange()

"`(min, step)` of `v` under `policy` at `L` levels."
@inline function vectorrange(::ExtremaRange, v::AbstractVector, L::Integer; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    (min, (max - min + eps) / Float32(L))
end

function vectorrange(p::SymmetricRange, v::AbstractVector, L::Integer; eps::Float32=1f-6)
    s1 = 0.0; s2 = 0.0
    @inbounds for x in v
        s1 += Float64(x); s2 += Float64(x) * Float64(x)
    end
    n = length(v)
    μ = s1 / n
    σ = sqrt(max(0.0, s2 / n - μ * μ))
    σ > 0 || return vectorrange(ExtremaRange(), v, L; eps)
    μ32, σ32 = Float32(μ), Float32(σ)
    k = p.k > 0 ? p.k :
        p.search === :exact ? _searchk(v, L, μ32, σ32) :
        p.search === :refined ? _refinek(v, L, μ32, σ32, _searchk_fast(v, L, μ32, σ32)) :
        _searchk_fast(v, L, μ32, σ32)
    (μ32 - k * σ32, (2k * σ32 + eps) / Float32(L))
end

# the real grid's distortion at the fast factor and its two neighbouring candidates
function _refinek(v::AbstractVector, L::Integer, μ::Float32, σ::Float32, k0::Float32)
    w = _KMAX / _KBINS
    argmin(k -> sqdistortion(v, L, μ - k * σ, μ + k * σ), (max(0.5f0, k0 - w), k0, min(_KMAX, k0 + w)))
end

# The fast search. One pass bins the deviations |x - μ| in units of σ, 64 bins of width 4.5/64
# up to 4.5σ (anything beyond lands in the last bin, which every candidate saturates). The
# candidate factors are the bin edges from 0.5σ on, so for each of them the saturated set is
# exactly the bins above the edge: with suffix sums of the counts, deviations and squared
# deviations the saturation error Σ(d - kσ)² is exact and the rounding error is the uniform
# `(n - m)·step²/12`, `step = 2kσ/L`. The model is coarse at 2 bits (three levels are not a
# uniform rounder) and tight from 4 bits on; its recall matched the exact search's on
# yahooaq and ccnews within measurement.
const _KBINS = 64
const _KMAX = 4.5f0
function _searchk_fast(v::AbstractVector, L::Integer, μ::Float32, σ::Float32)
    cnt = zeros(Int32, _KBINS); s1 = zeros(Float32, _KBINS); s2 = zeros(Float32, _KBINS)
    w = _KMAX / _KBINS
    invσw = 1f0 / (σ * w)
    @inbounds for x in v
        d = abs(Float32(x) - μ)
        j = min(_KBINS, 1 + unsafe_trunc(Int, d * invσw))
        cnt[j] += 1; s1[j] += d; s2[j] += d * d
    end
    # suffix sums: what bins j..end hold
    n = length(v)
    best = _KMAX; bestD = Inf32
    m = 0; S1 = 0f0; S2 = 0f0
    step2 = (2f0 * σ / Float32(L))^2 / 12f0   # times k² gives the rounding error per coordinate
    @inbounds for j in _KBINS:-1:1
        m += cnt[j]; S1 += s1[j]; S2 += s2[j]
        k = (j - 1) * w            # the lower edge of bin j: a candidate factor saturates bins j..end
        k < 0.5f0 && break
        R = k * σ
        D = (S2 - 2f0 * R * S1 + m * R * R) + (n - m) * step2 * k * k
        D < bestD && (bestD = D; best = k)
    end
    best
end

# the symmetric factor minimizing this vector's own distortion: a coarse sweep every 0.25 over
# 0.5:4.5, then a fine sweep every 0.05 around its best (28 distortion passes over the vector
# instead of 81). The Gaussian optima are near 1.5, 2.5 and 3.9 at 2, 4 and 8 bits.
function _searchk(v::AbstractVector, L::Integer, μ::Float32, σ::Float32)
    best = argmin(k -> sqdistortion(v, L, μ - k * σ, μ + k * σ), 0.5f0:0.25f0:4.5f0)
    argmin(k -> sqdistortion(v, L, μ - k * σ, μ + k * σ), max(0.5f0, best - 0.25f0):0.05f0:min(4.5f0, best + 0.25f0))
end

"""
    quantvector!(::Val{B}, vout::AbstractVector{UInt8}, v::AbstractVector; range=DEFAULT_RANGE, eps=1f-6) -> SQMinC

Quantizes `v` into `vout` on its **own** range, placed by `range` (a [`RangePolicy`](@ref);
the default searches a symmetric `mean ± k·σ`), so a vector on a different scale from the rest
is encoded as faithfully as any other. Returns the [`SQMinC`](@ref) that dequantizes it; `eps`
keeps a constant vector's range from collapsing.
"""
function quantvector!(B::Val, vout::AbstractVector{UInt8}, v::AbstractVector; range::RangePolicy=DEFAULT_RANGE, eps::Float32=1f-6)
    min, c = vectorrange(range, v, levels(B); eps)
    packcodes!(B, vout, v, min, 1f0/c)
    SQMinC(min, c)
end

"""
    SQVec{B,VEC<:AbstractVector{UInt8}}

    SQVec{B}(v::AbstractVector)
    SQVec{B}(E::SQMinC, V::AbstractVector{UInt8})
    SQVec{B}(E::SQMinC, V::AbstractVector{UInt8}, Sa, Saa)

A single vector quantized to `B` bits per coordinate (2, 4 or 8): the packed codes `V`
(`codesperbyte(B)` coordinates to a byte, low bits first), the affine dequantization
parameters `E` (coordinate `i` is `code * E.c + E.min`), and the two code sums `Sa = Σ codes`
and `Saa = Σ codes²` that let every distance in this module be computed from one integer
pass over the codes (see the note above `codesums` in `codes.jl`). Indexing (`qvec[i]`)
dequantizes coordinate `i` to a `Float32`.

It is the element type of every quantized database here, in both families: a globally
quantized vector *is* a per-vector one whose `E` happens to be shared with the rest of its
database. `SQu8Vec`, `SQu4Vec` and `SQu2Vec` are aliases for `SQVec{8}`, `SQVec{4}` and
`SQVec{2}`.

The first constructor quantizes `v` on its own range ([`quantvector!`](@ref), placed by `range`). `length(v)`
must be a multiple of `codesperbyte(B)` (2 at 4 bits, 4 at 2 bits), or an `ArgumentError`
is thrown: pad `v` if needed, and then pad any plain vector later compared against the
result the same way, since the mixed distances index it positionally. The other two take
stored codes back as they are, recomputing the sums from the codes when they are not given.
"""
struct SQVec{B,VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
    Sa::Float32      # Σ codes      -- see the expansion above `codesums`
    Saa::Float32     # Σ codes²
end

SQVec{B}(E::SQMinC, V::VEC, Sa::Real, Saa::Real) where {B,VEC<:AbstractVector{UInt8}} =
    SQVec{B,VEC}(E, V, Float32(Sa), Float32(Saa))

SQVec{B}(E::SQMinC, V::AbstractVector{UInt8}) where {B} = SQVec{B}(E, V, codesums(Val(B), V)...)

function SQVec{B}(v::AbstractVector; range::RangePolicy=DEFAULT_RANGE) where {B}
    cpb = codesperbyte(Val(B))
    length(v) % cpb == 0 ||
        throw(ArgumentError("SQVec{$B}: length(v) = $(length(v)) must be a multiple of $cpb ($cpb coordinates are packed per UInt8)"))
    vout = Vector{UInt8}(undef, length(v) ÷ cpb)
    E = quantvector!(Val(B), vout, v; range)
    SQVec{B}(E, vout)
end

"The code width of `q`, in bits."
codewidth(::SQVec{B}) where {B} = B

Base.@propagate_inbounds function Base.getindex(q::SQVec{B}, i::Integer)::Float32 where {B}
    Float32(getcode(Val(B), q.V, i)) * q.E.c + q.E.min
end

Base.length(q::SQVec{B}) where {B} = codesperbyte(Val(B)) * length(q.V)
Base.eachindex(q::SQVec) = 1:length(q)
Base.eltype(::SQVec) = Float32
Base.eltype(::Type{<:SQVec}) = Float32
