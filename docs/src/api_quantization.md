```@meta

CurrentModule = SimilaritySearch
DocTestSetup = quote
    using SimilaritySearch
end
```


# Quantization, sketches and estimators

## Scalar quantization (`ScalarQuant` submodule)
Reduces the memory footprint of a database by quantizing each coordinate to 2, 4 or 8
bits, in two families that differ in where the quantization range comes from (see the
module docstring for how to choose). Both produce the same vector type, `SQVec`, and share
one set of distances that read the packed codes directly; the per-width submodules keep
their un-prefixed API (`ScalarQuant.SQu8.quantize`, `ScalarQuant.SQu8.SqL2`, ...) as aliases.
```@docs
ScalarQuant
ScalarQuant.SQMinC
ScalarQuant.SQVec
ScalarQuant.quantvector!
ScalarQuant.sqdistortion
ScalarQuant.levels
```

### The quantized database (`QuantDatabase`)
One database type for both families, over any storage of `UInt8` vectors: static
(`MatrixDatabase`), growing in blocks (`BlockMatrixDatabase`), on disk (`MMapMatrixDatabase`)
or anything indexable. `push_item!`/`append_items!` quantize on the way in with the
database's own parameters.
```@docs
ScalarQuant.QuantDatabase
ScalarQuant.quantize
ScalarQuant.isglobal
ScalarQuant.codewidth
```

### Distances over quantized vectors
Defined once for every width and both families. Between two quantized vectors each is one
integer pass over the codes plus the per-vector sums; against a plain `Float32` vector the
codes are unpacked to floats with SIMD.
```@docs
ScalarQuant.SqL2
ScalarQuant.L2
ScalarQuant.L1
ScalarQuant.NormCosine
ScalarQuant.Cosine
```

### Per-column quantization (`SQu2`, `SQu4`, `SQu8` submodules)

Each column (vector) keeps its own `min`/scale, computed from its own extrema.
```@docs
ScalarQuant.SQu2
ScalarQuant.SQu2.quantize
ScalarQuant.SQu2.SQu2Database
ScalarQuant.SQu4
ScalarQuant.SQu4.quantize
ScalarQuant.SQu4.SQu4Database
ScalarQuant.SQu8
ScalarQuant.SQu8.quantize
ScalarQuant.SQu8.SQu8Database
ScalarQuant.SQu2.SQu2Vec
ScalarQuant.SQu4.SQu4Vec
ScalarQuant.SQu8.SQu8Vec
```

### Global (database-wide) quantization (`SQgu2`, `SQgu4`, `SQgu8` submodules)

All columns share a single `min`/scale, letting the distance kernels compare the packed
codes directly with SIMD, without any per-element dequantization. Each submodule offers an
allocating `quantize` and an in-place `quantize!(vout, v, minmax)` for loops that reuse
their output buffer.
```@docs
ScalarQuant.sqglobalscale
ScalarQuant.sqautorange
ScalarQuant.sqrange
ScalarQuant.SQgu2
ScalarQuant.SQgu2.quantize
ScalarQuant.SQgu2.quantize!
ScalarQuant.SQgu2.SqL2
ScalarQuant.SQgu4
ScalarQuant.SQgu4.quantize
ScalarQuant.SQgu4.quantize!
ScalarQuant.SQgu4.SqL2
ScalarQuant.SQgu8
ScalarQuant.SQgu8.quantize
ScalarQuant.SQgu8.quantize!
ScalarQuant.SQgu8.SqL2
```

### A database that keeps its quantization parameters (`GlobalQuantDatabase`)

`SQgu*.quantize` returns a bare matrix of codes and leaves `min`/`max` to the caller, so
stored codes cannot be dequantized and can only be compared against codes from the same run.
`GlobalQuantDatabase` keeps the pair, and the per-vector code sums an order-preserving cosine
needs; it is the global-family `QuantDatabase`, so every distance above applies to it and it
grows like any other.
```@docs
ScalarQuant.GlobalQuantDatabase
```

### The quantizers as an encoder for the asymmetric graph (`SQEncoder`)

Objects are quantized once and stored as codes, queries are prepared once (`SQQuery`: the
`Float32` query with its sums and an integer image), and the distances above evaluate one
against the other through one integer dot product per pair; an optional rotation is applied to both
sides first. It uses the `AbstractEstimator` interface the `AsymmetricSearchGraph` navigates
with, but carries no error model. The quantizer is named by its module (`SQgu4`, `SQu8`, ...),
the rotation by the object that applies it (`Projections.qr(dim, dim)`,
`Projections.RandomizedHadamard(dim)`) or `nothing`, the default.
```@docs
ScalarQuant.SQEncoder
ScalarQuant.RangePolicy
ScalarQuant.AutoRange
ScalarQuant.ExtremaRange
ScalarQuant.FixedRange
ScalarQuant.CalibratedRange
ScalarQuant.HistogramRange
ScalarQuant.RefinedRange
ScalarQuant.ExactRange
ScalarQuant.resolverange
ScalarQuant.calibrate
ScalarQuant.defaultk
ScalarQuant.SQQuery
ScalarQuant.sqcodes
ScalarQuant.quantizer
```

## RaBitQ (`RaBitQ` submodule)

The RaBitQ estimator (Gao & Long, 2024) as an `AbstractEstimator`: sign bits of the rotated
vector plus three scalars per object, an unbiased estimate of the cosine with a per-object
error bound, and a two-level variant that keeps a fallback beside the bits and re-evaluates
from it, inside the estimate, when the bound cannot rule an object out.
```@docs
RaBitQ
RaBitQ.AbstractRaBitQ
RaBitQ.RaBitQCode
RaBitQ.RaBitQQuery
RaBitQ.rabitqcodes
RaBitQ.estimatecos
RaBitQ.errorbound
RaBitQ.RaBitQRefined
RaBitQ.AbstractFallback
RaBitQ.RaBitQExactFallback
RaBitQ.RaBitQVectorFallback
RaBitQ.refinethreshold
```

## Random projections (`Projections` submodule)

!!! note "The hyperplanes pass through the origin"
    The sketches built from a linear map cut the space with hyperplanes through the origin.
    This covers [`Projections.bitsketch`](@ref) and [`Projections.QuantSketch`](@ref) over
    [`Projections.RandomProjections`](@ref) (built with [`Projections.gaussian`](@ref) or
    [`Projections.qr`](@ref)), and `index!(...; :bitsketch)` with `method=:gaussian` or
    `method=:qr`.
    - These sketches may work better on centered data. Centering means subtracting the
      mean of the collection.
    - This package does not center the data. Centering is the responsibility of the user.
    - The queries and every later insertion must have the same mean subtracted.

    See [Centering the data for linear sketches](@ref) below for when centering may help.

```@docs
Projections.RandomProjections
Projections.gaussian
Projections.qr
Projections.outdim
Projections.indim
Projections.transform
Projections.transform!
Projections.bitsketch
Projections.packsigns
Projections.packsigns!
```

### Centering the data for linear sketches

This section explains when to center the data before a linear sketch. A *linear sketch* here
is a sketch computed from a linear map: [`Projections.bitsketch`](@ref) or
[`Projections.QuantSketch`](@ref) over a [`Projections.RandomProjections`](@ref) or a
[`Projections.HadamardProjection`](@ref). *Centering* means subtracting the mean `μ` of the
collection from every vector.

**Why it can matter.** Each component of a linear sketch records the side of a hyperplane
through the origin. Suppose the mean of the data is large compared with its spread. Then most
objects fall on the same side of many hyperplanes. Those bits lose entropy, and the Hamming
distances between sketches bunch together. In a [`Projections.QuantSketch`](@ref) with 2, 4 or
8 bits, the single quantization range also spends levels on the offset.

**Who centers.** No function of this package centers the data. Centering is the
responsibility of the user, for two reasons. First, the user must subtract the same `μ` from
the data, from the queries and from every later insertion. Second, centering changes some
distances, as the next paragraph explains.

**Distances that are not translation invariant.** A distance is *translation invariant* when
adding the same vector to both arguments does not change it. The *L2 family* here means L2,
squared L2, L1, L∞ and the other Minkowski distances. All of them are translation
invariant. Cosine and inner product are not. For cosine and inner product, centering the
stored vectors changes the distances. The alternative is to center
only for the sketch: compute `sketch(x - μ)` and keep the original vectors for the distance.
The Hamming distance between sketches then follows the angle between centered vectors, not
the angle between the original vectors.

```julia
using SimilaritySearch, Statistics
using SimilaritySearch.Projections: bitsketch

μ = vec(mean(X; dims=2))
B, R = bitsketch(:qr, 256, X .- μ)    # the sketch of the centered vectors
bq = bitsketch(R, q .- μ)             # a query, centered with the same μ
```

The same `B` can build a graph over the original vectors with
`index!(idx, ctx, :bitsketch; method=:external, nbits=256, sketch=B)`. The sketch is
used only to build the graph, so the queries of `idx` need no centering.

**What does not need centering.** Do not center by reflex. Three cases need no centering:
- [`Projections.PCAProjection`](@ref) centers by itself. Its `transform` calls the `predict`
  of MultivariateStats, which subtracts the fitted mean. The exception is a model fitted
  with `mean=0`.
- The metric hyperplanes ([`Projections.DistantHyperplanes`](@ref),
  [`Projections.AnchoredDistantHyperplanes`](@ref), [`Projections.RandomHyperplanes`](@ref))
  depend only on distances between an object and pairs of anchors. The anchors are objects
  of the collection, so centering moves them with the data. A translation-invariant
  distance (the L2 family) then gives the same sketch with or without centering.
- A plain [`Projections.transform`](@ref) followed by an L2-family distance does not depend
  on the mean either, because the map is linear.

**A cheap diagnostic.** Sketch a sample of the data, and compute for each bit the share of
objects that set it to 1. A bit whose share is far from 0.5, for example
`abs(p - 0.5) > 0.4`, is nearly constant. A large share of such bits means that the sketch
spends bits on the offset.

```julia
nbits = 256
B, R = bitsketch(:qr, nbits, X[:, 1:10_000])    # a sample of 10 000 objects
p = [mean((B[((i - 1) >> 6) + 1, j] >> ((i - 1) & 63)) & 1 for j in axes(B, 2)) for i in 1:nbits]
mean(abs.(p .- 0.5) .> 0.4)    # the share of nearly constant bits
```

**Measured so far.** The effect depends on the collection. Many embeddings from recent models
are nearly centered already, and they may not benefit from centering. The test assigned every
object to its nearest centers in Hamming distance. The sketches had 384 sign bits after a
`qr` rotation. There were K ∈ {256, 1024, 4096} centers, and each object took its k ∈ {1, 2,
3} nearest ones. The test measured the share of the 10 nearest neighbors of an object that
fall in its k cells.
- The collections were `ccnews` and `yahooaq` from SISAP 2025, both normalized embeddings.
  The norm of the mean is 0.27 on `ccnews` and 0.12 on `yahooaq`.
- On both, centering changed the share by at most 0.03.
- On these two collections, no bit was nearly constant without centering.
- Collections with a large mean are not measured yet. Examples are non-negative features,
  such as SIFT or histograms, and embeddings with a strong common component. Centering may
  help there, but there are no numbers.

Whether to center is a decision of the user, on their own data. `PCAProjection` is the
exception, since it centers by itself.

## Hadamard projection (`Projections.HadamardProjection`) and the rotations

A projection computed with the fast Walsh-Hadamard transform (an in-place butterfly,
`Projections.fwht!`, with no plan behind it) instead of a dense random matrix. Uses the same `outdim`/`indim`/`transform`/`transform!`/`bitsketch` generic
functions documented above for `RandomProjections`. `RandomizedHadamard` makes a random
rotation of it (a random sign per coordinate first, and the norm preserved), and `Rotation`
is what the estimators (`ScalarQuant.SQEncoder`, `RaBitQ`) take as theirs.

!!! note "The hyperplanes pass through the origin"
    `HadamardProjection` and `RandomizedHadamard` are linear maps, so a sketch built from
    them cuts the space with hyperplanes through the origin.
    - These sketches may work better on centered data.
    - This package does not center the data. Centering is the responsibility of the user.
    - The queries and every later insertion must have the same mean subtracted.

    See [Centering the data for linear sketches](@ref) for when centering may help.

```@docs
Projections.HadamardProjection
Projections.fwht!
Projections.RandomizedHadamard
Projections.Rotation
```

## PCA projection (`Projections.PCAProjection`)

A projection fitted from data, via [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl)'s
`PCA`, instead of a random or structured rotation. Uses the same
`outdim`/`indim`/`transform`/`transform!`/`bitsketch` generic functions documented above
for `RandomProjections`; unlike those, its matrix `transform` has no `minbatch` (a single
vectorized call into MultivariateStats already covers every column).

```@docs
Projections.PCAProjection
```

## Hyperplane bit sketches (`Projections` submodule)

Binary sketch generators for *any* metric space -- not just floating-point vectors under
`transform` above: an object is encoded by which side of a set of hyperplanes, pairs of
anchor objects compared through the space's own distance function, it falls on. Each of
these carries its own [`distance`](@ref) (Hamming, over the packed sketch) and supports
[`Projections.outdim`](@ref)/[`Projections.bitsketch`](@ref) like the projections above.
See the [bit sketches tutorial](@ref "Quantization and Bit Sketches") for a worked example.

```@docs
Projections.DistantHyperplanes
Projections.AnchoredDistantHyperplanes
Projections.RandomHyperplanes
```

## Multi-bit sketches (`Projections.QuantSketch`)

The same sketch models as above, but keeping an `m`-bit unsigned code per component
(`m = 2, 4, 8`, via [`ScalarQuant.SQgu2`](@ref)/[`ScalarQuant.SQgu4`](@ref)/[`ScalarQuant.SQgu8`](@ref))
instead of a single sign bit -- so a sketch records *how far* an object sits from each
hyperplane, not merely on which side. `nbits=1` is supported too and reproduces
[`Projections.bitsketch`](@ref) exactly, so a sweep over `1, 2, 4, 8` bits runs through one
API. Applies to both families: for a rotation the encoded value is the projected
coordinate, for a metric hyperplane it is the signed margin
`d(obj, b) - d(obj, a)` -- see [`Projections.sketchvalues!`](@ref).

```@docs
Projections.QuantSketch
Projections.quantsketch
Projections.sketchvalues!
Projections.sketchbits
Projections.sketchsize
Projections.hyperplanewidths
```

## Sketch-based search pipeline (`Projections.SketchedSearch`)

Encode, index, retrieve candidates cheaply, re-score them exactly -- packaged as an
ordinary `AbstractSearchIndex`, so `search`/`searchbatch` work on it unchanged and its
results are ids into the original database with true distances.

```@docs
Projections.SketchedSearch
Projections.exhaustivesketchindex
```
