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

include("gu8.jl")
include("gu4.jl")
include("gu2.jl")
include("u8.jl")
include("u4.jl")
include("u2.jl")
include("gdb.jl")

end