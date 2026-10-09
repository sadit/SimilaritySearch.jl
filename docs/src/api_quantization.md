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
ScalarQuant.SymmetricRange
ScalarQuant.ExtremaRange
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

## Hadamard projection (`Projections.HadamardProjection`) and the rotations

A projection computed with the fast Walsh-Hadamard transform (an in-place butterfly,
`Projections.fwht!`, with no plan behind it) instead of a dense random matrix. Uses the same `outdim`/`indim`/`transform`/`transform!`/`bitsketch` generic
functions documented above for `RandomProjections`. `RandomizedHadamard` makes a random
rotation of it (a random sign per coordinate first, and the norm preserved), and `Rotation`
is what the estimators (`ScalarQuant.SQEncoder`, `RaBitQ`) take as theirs.

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
