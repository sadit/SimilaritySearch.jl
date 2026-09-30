```@meta
CurrentModule = SimilaritySearch
```

# Asymmetric Search: Raw Queries Against Codes

The last two sections encoded vectors into codes -- scalar quantization, bit sketches,
multi-bit sketches -- and then compared **code against code**: an `ExhaustiveSearch` over a
quantized database, a `SearchGraph` whose topology was bootstrapped in sketch space,
`Dist.Bits.Hamming` between two sketches. That is the **symmetric** mode. Both sides of every
distance evaluation are encoded, so the encoding error is paid twice, once on each side. It is
the cheapest mode at every step, and the only one available when the codes are all there is.

This section is the other mode. An [`AsymmetricSearchGraph`](@ref) keeps its storage encoded
but inserts and searches with the objects in their **raw** form: every distance the graph
evaluates is a raw query against a stored code, so the edges are chosen on the exact side of
the comparison and the error is paid once. Two encoders ship with the package:
[`ScalarQuant.SQEncoder`](@ref), the quantizers of the previous sections as the encoder of
such a graph, and the [`RaBitQ`](@ref) estimators, sign bits with an error bound per object.

| | symmetric: `SearchGraph` over codes, `index!(:bitsketch)`, `SketchedSearch` | asymmetric: `AsymmetricSearchGraph` |
| :--- | :--- | :--- |
| what is stored | codes | codes |
| what is inserted | codes (the database encodes on `push_item!`) | raw objects, encoded once by the distance |
| what a query is | a code, or a raw vector the kernel accepts | a raw object, prepared once by `encodequery` |
| what is evaluated | code against code | raw against code; code against code only among a new item's candidates |
| where the error goes | both sides, and into the edges | one side; the edges are chosen on the raw form |
| when | the codes are all there is; the cheapest option | the raw objects are available at insertion and query time |

The way of working is a property of the instance, fixed when it is built: a `SearchGraph`
never sees a raw object, an `AsymmetricSearchGraph` never inserts or searches with a code,
and both are `AbstractSearchGraph`s with the same search interface.

---

## The distance is the encoder

The graph knows nothing about codes. Its distance is an [`AbstractEstimator`](@ref), one plain
type whose parameters are fields, and the graph only ever calls three things on it:

- [`encode`](@ref)`(est, obj)`: what the storage receives for a raw object;
- [`encodequery`](@ref)`(est, q)`: what a raw query becomes, applied **once per query and once
  per inserted item** -- a rotation costs `dim²`, and applying it inside every evaluation would
  dwarf the evaluation itself;
- `evaluate(est, q, stored)`, the raw query against a stored code, and `evaluate(est, a, b)`
  between two stored codes, which the neighborhood filters use to compare a new item's
  candidates among themselves.

Everything the model needs to give its codes meaning travels in the estimator or in the code,
so a graph and its distance serialize together, and a model that can bound its own error
re-evaluates *inside* `evaluate` when it must, transparently to the graph.

---

## `SQEncoder`: the quantizers as the encoder

[`ScalarQuant.SQEncoder`](@ref) packages the scalar quantizers of
[Quantization and Bit Sketches](quantization_and_bitsketches.md) for the asymmetric graph. It
goes through the estimator interface because that is what the graph navigates with, but it is
a codification with no error to exploit: nothing is bounded, nothing is re-evaluated, and its
`evaluate` is `ScalarQuant.SqL2` (or `L2`, `L1`, `NormCosine`, `Cosine`), the mixed kernel for
a `Float32` query against packed codes and the integer kernel between two codes.

The quantizer is named by its module, which says the family and the width at once, and the
storage the graph grows is [`ScalarQuant.sqcodes`](@ref)`(enc)`, a `QuantDatabase` in dense
blocks that takes the encoder's codes as they are:

```julia
# SimilaritySearch v1.5
using SimilaritySearch, SimilaritySearch.ScalarQuant
using SimilaritySearch.ScalarQuant: SQEncoder, sqcodes

X = randn(Float32, 64, 5_000)
queries = MatrixDatabase(randn(Float32, 64, 50))
k = 10
gold, _ = searchbatch(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)
recall(ids) = sum(length(intersect(Set(ids[:, j]), Set(gold[:, j]))) for j in 1:size(gold, 2)) / length(gold)
ctx = SearchGraphContext(; reporters=[])

enc = SQEncoder(ScalarQuant.SQgu4, X)          # 4 bits per coordinate, one range fitted on a sample of X
asym = AsymmetricSearchGraph(enc, sqcodes(enc))
append_items!(asym, ctx, MatrixDatabase(X))   # raw vectors in; stored as codes; edges chosen raw-against-codes
ids, _ = searchbatch(asym, ctx, queries, k)   # raw queries in
println("asymmetric, 4 bits: recall@10 = ", recall(ids), "  bytes/vector = ", 64 ÷ 2 + 8)

# the symmetric graph over the same codes: codes against codes on both sides
sym = SearchGraph(ScalarQuant.SqL2(), sqcodes(enc))
append_items!(sym, ctx, MatrixDatabase(X))    # the database quantizes each vector on the way in
ids, _ = searchbatch(sym, ctx, queries, k)
println("symmetric,  4 bits: recall@10 = ", recall(ids))
```

On i.i.d. Gaussian vectors at 4 bits the two tie: the codes carry the query's neighbors
either way. The difference is in the edges, and it shows on real data below 8 bits (see the
end of this page); with a rotation in front, the symmetric graph over `sqcodes(enc)` would
have to receive rotated vectors, `encodequery(enc, v)`, since its database quantizes with a
range fitted on the rotated coordinates.

| module | range | bits per coordinate | bytes per vector at 384-d |
| :--- | :--- | :--- | :--- |
| `SQgu8` | one for the whole database, fitted by `sqautorange` on a sample (or given as `minmax`) | 8 | 392 |
| `SQgu4` | global | 4 | 200 |
| `SQgu2` | global | 2 | 104 |
| `SQu8` | one per vector, from its own extrema; needs no data to build | 8 | 400 |
| `SQu4` | per vector | 4 | 208 |
| `SQu2` | per vector | 2 | 112 |

The per-vector modules take only the dimension, `SQEncoder(ScalarQuant.SQu8, 64)`, and the
global ones take a matrix to fit their range on or an explicit `minmax`.

### Rotating first, or not

`SQEncoder` accepts a rotation between the quantizer and the data,
`SQEncoder(quant, rotation, X)`: a [`Projections.Rotation`](@ref) -- `Projections.qr(dim, dim)`,
an orthogonal matrix costing `dim²` per vector, or `Projections.RandomizedHadamard(dim)`, a
random sign per coordinate followed by the Walsh-Hadamard transform, `dim log dim` per vector
and `dim` a power of two -- or `nothing`. **The default is `nothing`**: the two-argument forms
above rotate nothing.

What a rotation buys is a *shared scale*. A global range is one `min`/scale for every
coordinate of every vector, and it rests on the coordinates having roughly the same spread;
when one coordinate is a hundred times wider than another, the range fitted on all of them
spends its levels on the wide one and flattens the narrow ones. A random rotation mixes every
coordinate into every other, so the rotated coordinates share one scale whatever the original
ones had. When the coordinates already share a scale -- normalized text and image embeddings do
-- it changes nothing and costs its flops per query and per inserted item. Measured on the
SISAP 2025 `ccnews` embeddings (603,664 x 384; issue #86), a QR rotation moved recall@10 by
less than 0.01 at every width and both families, and added 30-50 µs to each query. On
synthetic data the same encoder reads the other way as soon as the scales diverge (5,000
Gaussian vectors in 64-d, 100 queries, exhaustive over the codes, recall@10):

| coordinates | `SQgu4`, no rotation | `SQgu4`, QR | `SQu4`, no rotation | `SQu4`, QR |
| :--- | :--- | :--- | :--- | :--- |
| the same scale | 0.83 | 0.84 | 0.87 | 0.88 |
| scales from 1 to 100 | 0.79 | 0.85 | 0.88 | 0.89 |
| one coordinate 50x the rest | 0.12 | 0.28 | 0.43 | 0.67 |

So: leave the default when the coordinates share a scale, which is what a range fitted by
`sqautorange` assumes; rotate when they do not and the memory budget calls for the global
family; and note that the per-vector family, whose range follows each vector, is the other
remedy for uneven scales, at 8 more bytes per vector and no rotation. At 2 bits the per-vector
family collapses (0.65 against 0.82 of exhaustive recall on `ccnews`), rotated or not.

```julia
# SimilaritySearch v1.5
using SimilaritySearch, SimilaritySearch.ScalarQuant
using SimilaritySearch.ScalarQuant: SQEncoder, sqcodes
using SimilaritySearch: encodequery
using SimilaritySearch.Projections: RandomizedHadamard
const P = SimilaritySearch.Projections

scale = Float32.(range(1, 100; length=64))              # coordinates of very different spread
X = randn(Float32, 64, 5_000) .* scale
Q = randn(Float32, 64, 50) .* scale
k = 10
gold, _ = searchbatch(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), MatrixDatabase(Q), k)
recall(ids) = sum(length(intersect(Set(ids[:, j]), Set(gold[:, j]))) for j in 1:size(gold, 2)) / length(gold)

for (name, rot) in (("no rotation", nothing), ("QR", P.qr(64, 64)), ("randomized Hadamard", RandomizedHadamard(64)))
    enc = SQEncoder(ScalarQuant.SQgu4, rot, X)
    db = sqcodes(enc, X)                                   # every column encoded, in parallel
    # the encoder's own ceiling: an exhaustive scan of the codes with queries prepared by encodequery
    qs = VectorDatabase([encodequery(enc, q) for q in eachcol(Q)])
    ids, _ = searchbatch(ExhaustiveSearch(enc, db), GenericContext(), qs, k)
    println(rpad(name, 20), " 4 bits, exhaustive over the codes: recall@10 = ", recall(ids))
end
```

The object is rotated once, in `encode`, and the query once, in `encodequery`; nothing is
rotated inside an evaluation. An `AsymmetricSearchGraph` does both for you; the exhaustive scan
above prepares the queries by hand because `ExhaustiveSearch` takes whatever it is given.

---

## `RaBitQ`: an estimator with an error bound

[`RaBitQ`](@ref) (Gao & Long, 2024) is a genuine estimator. It stores, per object, the sign
bits of the rotated vector -- one bit per coordinate -- plus three scalars: the projection of
the unit vector onto its own sign vector, which normalizes the estimate, the norm, and the
half-width of the estimate's confidence interval. Against a raw query, rotated and normalized
once, the estimate of the cosine is one signed sum over the bits (a SIMD kernel, 67 ns per
384-d pair), unbiased, with [`RaBitQ.errorbound`](@ref) known per object. Between two stored
codes, which the neighborhood filters need, it is the SimHash estimate over the Hamming
distance of the bits.

Here the rotation is **required**, `RaBitQCosine(rotation)`: the estimate is unbiased and its
bound holds because the sign vector is taken in a uniformly random basis.

```julia
# SimilaritySearch v1.5
using SimilaritySearch, SimilaritySearch.RaBitQ, SimilaritySearch.ScalarQuant
using SimilaritySearch: encode
const P = SimilaritySearch.Projections

X = randn(Float32, 64, 5_000); X ./= sqrt.(sum(abs2, X; dims=1))      # unit vectors: cosine order == L2 order
Q = randn(Float32, 64, 50); Q ./= sqrt.(sum(abs2, Q; dims=1))
queries = MatrixDatabase(Q)
k = 10
gold, _ = searchbatch(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), queries, k)
recall(ids) = sum(length(intersect(Set(ids[:, j]), Set(gold[:, j]))) for j in 1:size(gold, 2)) / length(gold)
ctx = SearchGraphContext(; reporters=[])

est = RaBitQCosine(P.qr(64, 64))              # or RaBitQL2(rot) for the Euclidean distance
code = encode(est, X[:, 1])                    # RaBitQCode: 64 bits in one UInt64 word, c, norm, err
println("bytes per vector: ", 8 * length(code.bits) + 12, "   error bound of this object: ", errorbound(est, code))

G = AsymmetricSearchGraph(est, rabitqcodes(est))
append_items!(G, ctx, MatrixDatabase(X))       # raw vectors in, rotated once each, bits stored
ids, _ = searchbatch(G, ctx, queries, k)       # raw queries in, rotated once each
println("RaBitQ bits, 20 bytes per vector: recall@10 = ", recall(ids))

# two levels: a fallback beside the bits, consulted inside the estimate only when the bits'
# lower bound (estimate minus error) is within τ, the scale of the k-th neighbor's distance
τ = refinethreshold(est, X, k)
fallback = RaBitQVectorFallback(ScalarQuant.SQgu8, est, X)   # the rotated unit vector at 8 bits; or RaBitQExactFallback()
ref = RaBitQRefined(est, fallback; τ)
G2 = AsymmetricSearchGraph(ref, rabitqcodes(ref))
append_items!(G2, ctx, MatrixDatabase(X))
ids, _ = searchbatch(G2, ctx, queries, k)
println("bits + 8-bit fallback at τ = ", round(τ; digits=3), ": recall@10 = ", recall(ids))
```

The bits alone are weak on i.i.d. Gaussian vectors, and the printed numbers say why: in 64-d
the true cosines spread about `1/sqrt(64) = 0.125`, against an error bound near 0.18, so the
estimate cannot separate the neighbors from the rest. On data with structure the same bits
reach 0.69 of exhaustive recall@10 (`ccnews`), and the fallback lifts either case.

The two-level [`RaBitQ.RaBitQRefined`](@ref) is the shape a probabilistic estimator takes in
this design: the bits navigate, and when the bound cannot rule an object out the distance is
re-evaluated from the fallback -- [`RaBitQ.RaBitQExactFallback`](@ref), the rotated vector in
`Float32` or `Float16`, or [`RaBitQ.RaBitQVectorFallback`](@ref), the same vector through one of
the quantizer modules -- *inside the same `evaluate`*. There is no re-ranking pass after the
search and no raw data the graph goes back to: the correction is a re-evaluation, and the graph
only ever saw a distance. `τ = Inf` re-evaluates everything; [`RaBitQ.refinethreshold`](@ref)
reads a `τ` off a sample of `k`-th neighbor distances.

---

## Which graph, and which encoder

Measured on `ccnews` (issue #86), the fixed-beam columns being the comparable ones:

- At 8 bits the asymmetric `SQEncoder` graph matches the `Float32` graph at the same beam, at
  a quarter of the memory; the symmetric graph over the same codes is 30-40% faster per query
  and loses 0.002.
- Below 8 bits the asymmetric edges are better (searched in exact precision, 0.90 against 0.88
  at 4 bits and 0.94 against 0.88 at 2), but a query evaluated against codes cannot cash it:
  0.81 at 4 bits and 0.65 at 2 through either graph. The asymmetric graph pays when its
  distance re-evaluates what the codes alone cannot resolve.
- The RaBitQ bits, 60 bytes per 384-d vector, sit with the 2-bit `SQEncoder` (104 bytes) in
  recall. The fallback pays in an exhaustive scan (227 to 63 ms per query at the same recall)
  but not through an in-RAM graph, whose neighborhood filters compare candidates by the bits.

The symmetric pipeline has its own packaged form, next: [Sketched Search](sketchedsearch.md)
encodes, indexes the codes, retrieves more candidates than asked for and **re-scores them
afterwards** under the real distance -- a post-search pass over the raw data, which is exactly
what the asymmetric graph does not have and does not need.
