# This file is a part of SimilaritySearch.jl

"""
    ScalarQuant

Uniform scalar quantization of vector databases at 2, 4 and 8 bits per coordinate, in two
families that differ in *where the quantization range comes from*:

- **Global** (`SQgu2`, `SQgu4`, `SQgu8`, and [`GlobalQuantDatabase`](@ref)): a single
  `min`/scale pair shared by the whole database, estimated once from a sample of all
  coordinate values. Codes of different vectors are directly comparable, so the distances
  consume them with SIMD and nothing is stored per vector.
- **Per-vector** (`SQu2`, `SQu4`, `SQu8`): every stored vector carries its own `min`/scale
  ([`SQMinC`](@ref)), taken from that vector's own extrema. Nothing is ever clipped, and a
  vector living on a different scale than the rest is quantized as faithfully as any other;
  the price is 8 bytes per vector and a distance that must fold both vectors' parameters in
  (see `SQu8`'s `dotu8`).

# Choosing a family

Use a **global** quantizer when what you will quantize -- the database, the queries, and
whatever arrives after the index is built -- comes from the same distribution as the sample
the range was estimated from. That is the common case for normalized embeddings out of a
single model, and it is the cheaper family at query time.

Use a **per-vector** quantizer when coordinates may move outside what that sample showed:
several embedding models or preprocessing pipelines mixed into one database, unnormalized
data, scales that drift as the collection grows. A global range is an assumption about the
whole database, and a vector that breaks it saturates against the ends of the range, losing
precisely the large coordinates distances depend on. A per-vector range cannot be broken,
because it is derived from the vector itself.

The trade has a second edge. A per-vector range is set by that vector's own extrema, so one
outlying coordinate stretches the range and costs resolution for every other coordinate of
that vector. Per-vector adapts to scale differences *between* vectors; it does not help
against a heavy-tailed coordinate distribution, which costs resolution in both families.

# Choosing the range of a global quantizer

The range is a real parameter, and its best value depends on the code width, not only on the
data. Clipping at `±L` (in units of the coordinate standard deviation) trades two errors
against each other: the step is `2L / levels`, so a wider range coarsens every code, while a
narrower one saturates more mass. More levels make the step cheap, so the optimal `L` grows
with the width. For a Gaussian marginal, minimizing quantization error over
[`sqglobalscale`](@ref)'s own reconstruction grid gives

| bits | optimal `L` | mass it clips |
|:-----|:------------|:--------------|
| 2    | `1.50σ`     | 13.4%         |
| 4    | `2.52σ`     | 1.2%          |
| 8    | `3.93σ`     | 0.009%        |

Those factors are specific to this grid, and at 2 bits that matters. `sqglobalscale` places
`levels + 1` reconstruction points spanning `[min, max]` inclusive, `2L / levels` apart --
not the `2^bits` cells of width `2L / 2^bits` that textbook tables assume. At 2 bits the two
differ by a third in step size (`2L/3` against `2L/4`), which moves the optimum from the
textbook `2.0σ` to `1.50σ`; at 8 bits they agree to within 0.4% (255 points against 256
cells), which is why the difference is easy to miss.

The default policy here (`quant=[0.025, 0.975]`, i.e. `±1.96σ` under a Gaussian) is therefore
too wide for `SQgu2` and too tight for `SQgu8`. Measured against the benchmarks' own gold
standards on three ANN datasets (600K-3M vectors, 384 dimensions, coordinate kurtosis
3.04-3.05, so Gaussian to measurement precision), moving from the default to the optimal
range gained 0.033 to 0.057 of recall@10 at 2 bits and 0.005 to 0.010 at 4 bits; at 8 bits,
on five datasets, the same fix was worth 0.027 to 0.079.

Three practical notes on estimating it:

- Prefer passing `minmax` as *an interior quantile times a width-dependent factor* over
  asking for an extreme quantile directly. The default sample holds `sqrt(length(X))` values
  (~15K for a 600K x 384 database), which puts the 8-bit optimum's 99.99th percentile beyond
  anything the sample can resolve, while the 98th is stable.
- Anchor the range on the median, not on the origin or on the quantile interval's midpoint,
  and widen each side by its own distance to the median. All three agree on a symmetric
  centered marginal, and each of the other two breaks on a different shape: scaling both
  quantiles by one factor displaces a shifted marginal as it widens it (measured at 1.3-1.4x
  the error of the best range), while scaling about the midpoint drags `min` below zero on
  one-signed data such as ReLU outputs, spending levels where there is no data (2.6x the
  error, -0.046 of recall at 4 bits).
- Never take the range from the extrema. One outlier fixes it for the whole database and the
  codes collapse onto the middle: over the same datasets, `min/max` cost 0.15 at 2 bits where
  the extremes sat at `5.9σ`, and 0.67 where a single vector pushed them to `11.7σ`. It is
  also not reproducible, since the sample maximum grows like `σ sqrt(2 ln n)`.
- The factors above assume a near-Gaussian marginal, which a random rotation upstream
  guarantees and ordinary text embeddings satisfy. Under heavier tails the optimum moves out
  fast -- near `7σ` at 8 bits for a Laplace marginal -- so when the data has not been
  Gaussianized, pick `L` by minimizing quantization error over the sample rather than
  trusting the table. That search costs milliseconds and tracked the recall optimum closely
  here, though at 2 bits it lands slightly wide: recall peaked a little tighter than the
  error did.
"""
module ScalarQuant

using Distances: PreMetric, SemiMetric, Metric
using Statistics: quantile
using StatsBase
import Distances: evaluate
using ..SimilaritySearch: AbstractDatabase, getminbatch, Dist, @BATCHES
#using ..Dist: fastacos

"""
    SQMinC(min::Float32, c::Float32)

Internal helper struct that stores the per-vector dequantization parameters used by
the scalar quantization schemes in `ScalarQuant` (i.e., `SQu2`, `SQu4`, `SQu8`). Given a
quantized (integer) coordinate `q`, the corresponding approximate original value is
recovered as `q * c + min`.
"""
struct SQMinC
    min::Float32
    c::Float32
end

"""
    sqglobalscale(levels::Integer, min, max)

The scale factor shared by every *global* quantizer (`SQgu2`/`SQgu4`/`SQgu8`): maps the
range `[min, max]` onto the `levels + 1` integer codes `0:levels` as
`code = round(clamp((x - min) * c, 0, levels))`. The `1e-6` in the denominator keeps a
degenerate (`min == max`) range from producing `Inf`.

# Arguments
- `levels`: the largest code the target width can hold (`3`, `15` or `255`)
- `min`, `max`: the global value range being mapped
"""
sqglobalscale(levels::Integer, min, max) = Float32(levels / (max - min + 1e-6))

"""
    sqdistortion(S, levels, min, max)

Mean squared error the codes of a global quantizer over `[min, max]` would inflict on the
values `S`, counting both halves of what the range trades: the rounding of what falls inside
and the saturation of what falls outside. This is the quantity [`sqautorange`](@ref)
minimizes, and it is computed through the very same arithmetic
[`sqglobalscale`](@ref) hands the quantizers, so the range it picks is optimal for the grid
that will actually be used rather than for an idealized one.
"""
function sqdistortion(S::AbstractVector, levels::Integer, min::Real, max::Real)
    max > min || return Inf64
    c = sqglobalscale(levels, min, max)
    lo = Float32(min)
    top = Float32(levels)
    acc = 0.0
    @inbounds for x in S
        q = round(clamp((Float32(x) - lo) * c, 0f0, top))
        d = Float32(x) - (q / c + lo)
        acc += Float64(d) * Float64(d)
    end

    acc / length(S)
end

"""
    sqautorange(V, levels; samplesize=0, factors=0.30f0:0.05f0:3.0f0, passes=3)

Picks the `(min, max)` a global quantizer of `levels + 1` codes should use for `V`, by
minimizing [`sqdistortion`](@ref) over a sample of `V` itself. Returns a `(Float32, Float32)`
pair, ready for `minmax`.

The search runs over **each side separately**: the range is anchored at the median and each
end is placed at its own multiple of that side's distance to the `[0.02, 0.98]` quantiles,
`(min, max) = (med - a(med - q02), med + b(q98 - med))`, with `a` and `b` found by coordinate
descent over `factors`. Three properties come out of that parametrization, and all three were
measured to matter:

- **the right width follows the code width.** Widening coarsens every code by `2L/levels`
  while narrowing saturates more mass, and more levels make the first cheap, so the optimum
  moves out as `levels` grows: for a Gaussian marginal it sits near `1.50σ` at 2 bits and
  `3.93σ` at 8. One fixed quantile pair cannot serve both; searching finds each.
- **the median anchors it.** Scaling both quantiles by a single factor displaces a marginal
  that is not centered as it widens it, and anchoring at the quantile interval's midpoint
  instead drags `min` below zero on one-signed data (ReLU outputs, term weights), spending
  codes where there is no data. Anchoring at the median does neither, and `med == q02` pins
  a one-signed range at its own floor for free.
- **no distributional assumption.** Tabulated loading factors are Gaussian values, and the
  optimum moves out fast under heavier tails (near `7σ` at 8 bits for a Laplace marginal).
  The search reads the shape off the sample instead.

The sample is `1024 (levels + 1)` values, or `sqrt(length(V))` when that is larger, because
the tail the optimum clips shrinks with the code width and has to be visible in the sample to
be placed. Cost is a few hundred passes over it -- milliseconds even at 8 bits -- against a
quantization that touches every entry.
"""
function sqautorange(V::AbstractVector, levels::Integer;
        samplesize::Int=0, factors=0.30f0:0.05f0:3.0f0, passes::Int=3
    )
    n = length(V)
    # The sample has to resolve the tail the optimum will clip, and that tail shrinks as the
    # codes get finer: at 8 bits the best range saturates ~0.009% of the mass, which `sqrt(n)`
    # values (2.8K for a 7.7M-entry matrix) cannot see at all -- the search then settles on
    # whatever the noise above ~3σ suggests, and returns a range that is both too narrow and
    # visibly asymmetric on symmetric data. Scaling the floor with `levels` keeps a few dozen
    # sampled values beyond the optimum's own clipping point in every width.
    ss = samplesize == 0 ? clamp(Base.max(ceil(Int, sqrt(n)), 1024 * (levels + 1)), 1, n) :
                           Base.min(samplesize, n)
    S = ss < n ? rand(V, ss) : collect(V)
    med, qlo, qhi = quantile(S, (0.5, 0.02, 0.98))
    slo, shi = med - qlo, qhi - med

    # A degenerate spread (a constant column, or a sample that is more than 98% one value)
    # leaves nothing to place: fall back to the sample's own extrema, which `sqglobalscale`
    # already guards against collapsing.
    if slo <= 0 && shi <= 0
        lo, hi = extrema(S)
        return (Float32(lo), Float32(hi))
    end

    # Coordinate descent, coarse then fine: the distortion is smooth in both factors, so a
    # sweep at 1/10th of the range followed by a local sweep at full resolution finds the same
    # pair as the full grid for about a seventh of the evaluations -- which matters at 8 bits,
    # where the sample is large enough that the search would otherwise outcost the encoding.
    lo, hi = Float32(first(factors)), Float32(last(factors))
    step = Float32(Base.step(factors))
    coarse = lo:(10step):hi
    a = b = 1.0f0
    for pass in 1:passes
        grid(x) = pass == 1 ? coarse : Base.max(lo, x - 10step):step:Base.min(hi, x + 10step)
        slo > 0 && (a = argmin(f -> sqdistortion(S, levels, med - f * slo, med + b * shi), grid(a)))
        shi > 0 && (b = argmin(f -> sqdistortion(S, levels, med - a * slo, med + f * shi), grid(b)))
    end

    (Float32(med - a * slo), Float32(med + b * shi))
end

"""
    sqrange(V, levels; minmax=nothing, quant=nothing, samplesize=0)

The `(min, max)` every global quantizer in this module resolves before encoding: `minmax`
verbatim when given, else the fixed quantile pair `quant` when given, else
[`sqautorange`](@ref)'s search. Pass `quant=[0.025, 0.975]` to get the fixed-quantile
behaviour that used to be the default.
"""
function sqrange(V::AbstractVector, levels::Integer;
        minmax=nothing, quant=nothing, samplesize::Int=0
    )
    minmax === nothing || return (Float32(minmax[1]), Float32(minmax[2]))
    quant === nothing && return sqautorange(V, levels; samplesize)
    n = length(V)
    ss = samplesize === 0 ? ceil(Int, n^0.5) : samplesize
    lo, hi = quantile(rand(V, ss), quant)
    (Float32(lo), Float32(hi))
end

include("gu8.jl")
include("gu4.jl")
include("gu2.jl")
include("u8.jl")
include("u4.jl")
include("u2.jl")
include("gdb.jl")

end