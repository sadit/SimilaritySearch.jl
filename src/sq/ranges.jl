# This file is a part of SimilaritySearch.jl

### Range policies of the per-vector family.
###
### A per-vector code is `x̂ = min + c·a`, `a ∈ 0:L`, with `(min, c)` taken from the vector's own
### coordinates. A policy decides where `[min, max]` goes: on the extrema (nothing clipped), or
### symmetric about the vector's mean, `mean ± k·σ`, letting the coordinates beyond saturate so
### the bulk gets the levels. The symmetric policies differ only in how they pick `k`. Measured
### on yahooaq and ccnews (100K vectors, 384 dimensions, exhaustive recall@10, symmetric /
### asymmetric kernels; `scripts/benchmarks/bench-sq-ranges.jl` in BitSketches2026):
###
### | policy                 | 2 bits        | 4 bits        | 8 bits        | µs/vector |
### |------------------------|---------------|---------------|---------------|-----------|
### | extrema                | 0.544 / 0.616 | 0.913 / 0.931 | 0.994 / 0.995 | 2.3       |
### | calibrated k           | 0.741 / 0.803 | 0.912 / 0.927 | 0.992 / 0.994 | 1.1       |
### | calibrated k, refined  | 0.752 / 0.815 | 0.916 / 0.932 | 0.994 / 0.995 | 4.4       |
### | histogram, 64 bins     | 0.740 / 0.805 | 0.919 / 0.933 | 0.992 / 0.994 | 3.1       |
### | histogram, refined     | 0.758 / 0.822 | 0.921 / 0.939 | 0.992 / 0.994 | 6.3       |
### | exact search           | 0.760 / 0.822 | 0.924 / 0.940 | 0.993 / 0.995 | 32        |
###                                                   (yahooaq; ccnews ranks them the same)
###
### On ccnews the extrema were the only policy that did not lose at 8 bits (0.981 / 0.991 against
### 0.980-0.982 / 0.981-0.985). The histogram with 32 or 16 bins was below 64 at every width for
### 0.1-0.2 µs less. [`AutoRange`](@ref), the default, takes the calibrated `k` at 2 bits, the
### 64-bin histogram at 4 and the extrema at 8.

export RangePolicy, AutoRange, ExtremaRange, FixedRange, CalibratedRange, HistogramRange, RefinedRange, ExactRange

"""
    RangePolicy

Where the per-vector family places a vector's own range `[min, max]` before mapping it onto the
codes `0:levels(B)`. [`AutoRange`](@ref) (the default) picks by width among
[`ExtremaRange`](@ref), [`FixedRange`](@ref), [`CalibratedRange`](@ref), [`HistogramRange`](@ref),
[`RefinedRange`](@ref) and [`ExactRange`](@ref). A policy is applied with `vectorrange(policy, v,
L) -> (min, step)`; one that learns from data (the calibrated `k`) is fitted with
`calibrate(policy, X, L)` when an `SQEncoder` is built from a matrix, and the encoder keeps the
fitted policy. A per-vector database built from a matrix only resolves the policy: it must hold the
same codes that growing it with `push_item!` would.
"""
abstract type RangePolicy end

"A policy of the form `mean ± k·σ`; subtypes say how `k` is chosen (`_factor`)."
abstract type SymmetricPolicy <: RangePolicy end

"""
    ExtremaRange()

The vector's own extrema: nothing saturates, and one outlying coordinate stretches the range and
coarsens every other code of the vector. The rule at every width up to 1.6.3, and [`AutoRange`](@ref)'s
at 8 bits, where a clipped coordinate costs more than the finer step buys back.
"""
struct ExtremaRange <: RangePolicy end

"""
    FixedRange(k)

`mean ± k·σ` with a fixed `k`. The best fixed factors on near-Gaussian coordinates are about 1.5,
2.55 and 4.0 at 2, 4 and 8 bits ([`defaultk`](@ref)); at 4 bits a fixed factor gives up what a
per-vector search gains.
"""
struct FixedRange <: SymmetricPolicy
    k::Float32
end
FixedRange(; k::Real) = FixedRange(Float32(k))

"""
    CalibratedRange(; k=0, samplesize=4096, grid=0.5:0.05:4.5)

`mean ± k·σ` with one `k` for every vector, calibrated on `samplesize` vectors of the data by the
mean of their own distortions over `grid` ([`calibrate`](@ref), ~0.4 s at 384 dimensions); per
vector only the mean and σ are computed, one pass. Uncalibrated (`k == 0`, e.g. a lone
`SQVec{B}(v)`), it uses [`defaultk`](@ref). [`AutoRange`](@ref)'s policy at 2 bits: the same recall as
a per-vector histogram search at a third of its cost.
"""
struct CalibratedRange <: SymmetricPolicy
    k::Float32
    samplesize::Int
    grid::StepRangeLen{Float32,Float64,Float64,Int}
end
CalibratedRange(; k::Real=0, samplesize::Integer=4096, grid=0.5f0:0.05f0:4.5f0) =
    CalibratedRange(Float32(k), Int(samplesize), StepRangeLen{Float32,Float64,Float64,Int}(grid))

"""
    HistogramRange(; bins=64, kmax=4.5)

`mean ± k·σ` with `k` searched per vector on a histogram of the deviations `|x - mean|/σ`, `bins`
bins up to `kmax·σ`, filled in one pass; each candidate (a bin edge from 0.5) is scored in
constant time with the saturation error summed exactly from the histogram and the rounding error
modelled as `step²/12`. About 3 µs per vector at 384 dimensions. [`AutoRange`](@ref)'s policy at 4 bits.
32 and 16 bins were measured below 64 at every width for 0.1-0.2 µs less.
"""
struct HistogramRange <: SymmetricPolicy
    bins::Int
    kmax::Float32
end
function HistogramRange(; bins::Integer=64, kmax::Real=4.5)
    bins >= 8 || throw(ArgumentError("HistogramRange: bins must be at least 8"))
    HistogramRange(Int(bins), Float32(kmax))
end

"""
    RefinedRange(inner=CalibratedRange(); step=0.1)

The symmetric policy `inner`'s `k`, then the real grid's [`sqdistortion`](@ref) of the vector at
`k - step`, `k` and `k + step`, keeping the best: three passes more. Refining the calibrated `k`
(4.4 µs) or the histogram's (6.3 µs) recovers most of the exact search at 2 bits.
"""
struct RefinedRange{P<:SymmetricPolicy} <: SymmetricPolicy
    inner::P
    step::Float32
end
RefinedRange(inner::SymmetricPolicy=CalibratedRange(); step::Real=0.1) = RefinedRange(inner, Float32(step))

"""
    ExactRange(; kmin=0.5, kmax=4.5)

`mean ± k·σ` with `k` minimizing the vector's own [`sqdistortion`](@ref): a sweep every 0.25, then
every 0.05 around its best (28 passes over the vector, ~32 µs). The reference the others are
measured against.
"""
struct ExactRange <: SymmetricPolicy
    kmin::Float32
    kmax::Float32
end
ExactRange(; kmin::Real=0.5, kmax::Real=4.5) = ExactRange(Float32(kmin), Float32(kmax))

"""
    AutoRange()

The per-vector family's default, resolved by width ([`resolverange`](@ref)): [`CalibratedRange`](@ref)
at 2 bits, [`HistogramRange`](@ref) at 4, [`ExtremaRange`](@ref) at 8.
"""
struct AutoRange <: RangePolicy end

"The per-vector family's default range policy."
const DEFAULT_RANGE = AutoRange()

"""
    resolverange(policy, B) -> RangePolicy

The concrete policy `policy` stands for at `B` bits: `AutoRange()` becomes the policy chosen for
that width, any other policy itself.
"""
resolverange(p::RangePolicy, ::Integer) = p
resolverange(::AutoRange, B::Integer) = B <= 2 ? CalibratedRange() : B <= 4 ? HistogramRange() : ExtremaRange()

"""
    defaultk(L) -> Float32

The factor a calibrated policy uses before calibration: the calibrated `k` measured on yahooaq and
ccnews (1.5, 2.55, 4.0 at 3, 15, 255 levels), which are also the Gaussian optima of this grid.
"""
defaultk(L::Integer) = L <= 3 ? 1.5f0 : L <= 15 ? 2.55f0 : 4.0f0

"""
    calibrate(policy, X::AbstractMatrix, L) -> RangePolicy

The policy fitted to the columns of `X` at `L` levels: a [`CalibratedRange`](@ref) without `k` gets
the one minimizing the mean distortion of `samplesize` columns; a [`RefinedRange`](@ref) calibrates
its inner policy; every other policy is returned as it is.
"""
calibrate(p::RangePolicy, ::AbstractMatrix, ::Integer) = p
calibrate(p::RefinedRange, X::AbstractMatrix, L::Integer) = RefinedRange(calibrate(p.inner, X, L), p.step)
function calibrate(p::CalibratedRange, X::AbstractMatrix, L::Integer; rng::AbstractRNG=Xoshiro(0x5a))
    p.k > 0 && return p
    n = size(X, 2)
    n == 0 && return p
    ids = n <= p.samplesize ? (1:n) : rand(rng, 1:n, p.samplesize)
    ms = [_meansigma(view(X, :, i)) for i in ids]
    k = argmin(p.grid) do k
        acc = 0.0
        for (i, (μ, σ)) in zip(ids, ms)
            σ > 0 && (acc += sqdistortion(view(X, :, i), L, μ - k * σ, μ + k * σ))
        end
        acc
    end
    CalibratedRange(Float32(k), p.samplesize, p.grid)
end

"Mean and population σ of `v` in one pass."
@inline function _meansigma(v::AbstractVector)
    s1 = 0.0; s2 = 0.0
    @inbounds for x in v
        s1 += Float64(x); s2 += Float64(x) * Float64(x)
    end
    n = length(v)
    μ = s1 / n
    Float32(μ), Float32(sqrt(max(0.0, s2 / n - μ * μ)))
end

"""
    vectorrange(policy, v, L; eps=1f-6) -> (min, step)

Where `policy` places `v`'s range at `L` levels.
"""
@inline function vectorrange(::ExtremaRange, v::AbstractVector, L::Integer; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    (min, (max - min + eps) / Float32(L))
end

vectorrange(::AutoRange, v::AbstractVector, L::Integer; eps::Float32=1f-6) =
    vectorrange(resolverange(AutoRange(), L <= 3 ? 2 : L <= 15 ? 4 : 8), v, L; eps)

function vectorrange(p::SymmetricPolicy, v::AbstractVector, L::Integer; eps::Float32=1f-6)
    μ, σ = _meansigma(v)
    σ > 0 || return vectorrange(ExtremaRange(), v, L; eps)
    k = _factor(p, v, L, μ, σ)
    (μ - k * σ, (2k * σ + eps) / Float32(L))
end

_factor(p::FixedRange, v, L, μ, σ) = p.k
_factor(p::CalibratedRange, v, L, μ, σ) = p.k > 0 ? p.k : defaultk(L)

function _factor(p::RefinedRange, v, L, μ, σ)
    k0 = _factor(p.inner, v, L, μ, σ)
    argmin(k -> sqdistortion(v, L, μ - k * σ, μ + k * σ), (max(0.5f0, k0 - p.step), k0, k0 + p.step))
end

function _factor(p::ExactRange, v, L, μ, σ)
    best = argmin(k -> sqdistortion(v, L, μ - k * σ, μ + k * σ), p.kmin:0.25f0:p.kmax)
    argmin(k -> sqdistortion(v, L, μ - k * σ, μ + k * σ), max(p.kmin, best - 0.25f0):0.05f0:min(p.kmax, best + 0.25f0))
end

# One pass bins the deviations |x - μ| in units of σ; the candidates are the bin edges from 0.5σ,
# and for each the saturated set is exactly the bins above it: suffix sums of counts, deviations
# and squared deviations give the saturation error Σ(d - kσ)², and the rounding error is the
# uniform `(n - m)·step²/12`, `step = 2kσ/L`.
function _factor(p::HistogramRange, v, L, μ, σ)
    bins = p.bins
    cnt = zeros(Int32, bins); s1 = zeros(Float32, bins); s2 = zeros(Float32, bins)
    w = p.kmax / bins
    invσw = 1f0 / (σ * w)
    @inbounds for x in v
        d = abs(Float32(x) - μ)
        j = min(bins, 1 + unsafe_trunc(Int, d * invσw))
        cnt[j] += 1; s1[j] += d; s2[j] += d * d
    end
    n = length(v)
    best = p.kmax; bestD = Inf32
    m = 0; S1 = 0f0; S2 = 0f0
    step2 = (2f0 * σ / Float32(L))^2 / 12f0
    @inbounds for j in bins:-1:1
        m += cnt[j]; S1 += s1[j]; S2 += s2[j]
        k = (j - 1) * w
        k < 0.5f0 && break
        R = k * σ
        D = (S2 - 2f0 * R * S1 + m * R * R) + (n - m) * step2 * k * k
        D < bestD && (bestD = D; best = k)
    end
    best
end
