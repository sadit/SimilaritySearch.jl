```@meta
CurrentModule = SimilaritySearch
```

# Multi-Bit Sketches: Spending More Than One Bit per Hyperplane

The [previous section](quantization_and_bitsketches.md) built *bit* sketches: each hyperplane
of a fitted model contributes one bit, recording which side of it an object falls on. Fitting
those hyperplanes is the expensive part of the construction, and a sign bit discards most of
what they computed. It records *which side*, and never *how far*.

[`Projections.QuantSketch`](@ref) keeps that distance instead, at 2, 4 or 8 bits per
hyperplane. The model is **not** resized: the same `nbits` hyperplanes are fitted either way,
and the sketch simply occupies `nbits * width` bits. Precision is bought with memory, at no
extra model-fitting cost.

---

## One value, two encodings

Both encodings are a function of the same real-valued vector,
[`Projections.sketchvalues!`](@ref):

| model family | what the value is |
| :--- | :--- |
| rotations ([`Projections.RandomProjections`](@ref), [`Projections.HadamardProjection`](@ref), [`Projections.PCAProjection`](@ref)) | the projected coordinate |
| metric hyperplanes ([`Projections.DistantHyperplanes`](@ref), [`Projections.AnchoredDistantHyperplanes`](@ref), [`Projections.RandomHyperplanes`](@ref)) | the signed margin $d(x, b) - d(x, a)$ |

`bitsketch` keeps the sign of that value; `QuantSketch` quantizes it. The sign convention is shared. A `QuantSketch` of width 1 therefore reproduces
[`Projections.bitsketch`](@ref) exactly. A sweep over 1, 2, 4 and 8 bits then runs through a
single API, instead of comparing two encoders that were written independently.

```julia
# SimilaritySearch v1.5
using SimilaritySearch
const P = SimilaritySearch.Projections

X = randn(Float32, 64, 2_000)
db = MatrixDatabase(X)
model = P.gaussian(64, 256)          # 256 hyperplanes, fitted once

for width in (1, 2, 4, 8)
    qs = P.QuantSketch(model, width, db)
    codes = P.quantsketch(qs, db)
    println("width=", width,
            "  bits/sketch=", 256 * width,
            "  bytes/vector=", sizeof(codes.matrix) ÷ length(db),
            "  distance=", typeof(P.distance(qs)))
end
```

At width 1 the codes are `UInt64` words, compared with `Dist.Bits.Hamming`. At widths 2, 4 and 8
they are packed `UInt8` codes, compared with the matching `ScalarQuant.SQgu*.SqL2`. That kernel
reads them without dequantizing anything.

---

## Bootstrapping a `SearchGraph` from wider codes

[`index!`](@ref)`(idx, ctx, :bitsketch)` builds a graph's topology in sketch space and copies
it back, which is far cheaper than building it under the original distance. The `width`
keyword decides how many bits each hyperplane's value survives with:

```julia
# SimilaritySearch v1.5
using SimilaritySearch

X = randn(Float32, 64, 2_000)
db = MatrixDatabase(X)
dist = Dist.SqL2()

G = SearchGraph(dist, db)
ctx = SearchGraphContext()
index!(G, ctx, :bitsketch; method=:gaussian, nbits=512, width=4)

optimize_index!(G, ctx, MinRecall(0.9))
println("built from 512 hyperplanes at 4 bits each = ", 512 * 4, " bits per sketch")
```

`width=1` is the historical path and stays bit-for-bit what it was. `method=:external`, which
takes a precomputed sketch, supports only `width=1`: a matrix of bits carries no quantization
range, so there is nothing to compare wider codes with.

| `width` | sketch size | compared with |
| :--- | :--- | :--- |
| 1 | `nbits` bits | `Dist.Bits.Hamming` |
| 2 | `2 * nbits` bits | `ScalarQuant.SQgu2.SqL2` |
| 4 | `4 * nbits` bits | `ScalarQuant.SQgu4.SqL2` |
| 8 | `8 * nbits` bits | `ScalarQuant.SQgu8.SqL2` |

---

## Choosing between more hyperplanes and more bits

Both spend memory, and they are not interchangeable. A wider code refines the resolution of a
hyperplane that has already been fitted; a new hyperplane adds an independent direction. The useful question is which resource is scarce. If recall stops improving as `nbits` grows, the
model has no more informative directions, and the remaining error is in `width`. If the sketch
is already coarse in *direction*, add hyperplanes first.

Everything on this page compared code against code: the sketch of the query against the
sketches stored, which is the **symmetric** mode and the cheapest one. The next section,
[Asymmetric Search: raw queries against codes](asymmetric.md), keeps the storage encoded but
evaluates the raw query against it, so the encoding error is paid on one side only; after it,
[Sketched Search](sketchedsearch.md) wraps the symmetric encode, index, retrieve and re-score
steps into one ordinary search index.
