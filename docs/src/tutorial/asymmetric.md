```@meta
CurrentModule = SimilaritySearch
```

# Asymmetric Search: Raw Queries Against Codes

The last two sections encoded vectors into codes. The encoders were scalar quantization, bit
sketches and multi-bit sketches. Every comparison was then **code against code**: an
`ExhaustiveSearch` over a quantized database, a `SearchGraph` whose topology was bootstrapped in
sketch space, `Dist.Bits.Hamming` between two sketches.

That is the **symmetric** mode. Both sides of a distance evaluation are encoded, so the encoding
error is paid twice. This mode is the cheapest at every step. It is also the only one available
when the raw objects are gone and only the codes remain.

This section describes the other mode. An [`AsymmetricSearchGraph`](@ref) stores codes, but it
inserts and searches with the objects in their **raw** form. Every distance it evaluates is a raw
query against a stored code. The error is therefore paid once instead of twice, and the edges are
chosen with the exact side of the comparison.

The package provides two encoders. [`ScalarQuant.SQEncoder`](@ref) uses the quantizers of the
previous sections. The [`RaBitQ`](@ref) estimators store sign bits and an error bound for each
object.

| | symmetric: `SearchGraph` over codes, `index!(:bitsketch)`, `SketchedSearch` | asymmetric: `AsymmetricSearchGraph` |
| :--- | :--- | :--- |
| what is stored | codes | codes |
| what is inserted | codes (the database encodes on `push_item!`) | raw objects, encoded once by the distance |
| what a query is | a code, or a raw vector the kernel accepts | a raw object, prepared once by `encodequery` |
| what is evaluated | code against code | raw against code; code against code only among a new item's candidates |
| where the error goes | both sides, and into the edges | one side; the edges are chosen on the raw form |
| when | the codes are all there is; the cheapest option | the raw objects are available at insertion and query time |

The mode is a property of the instance and is fixed when the graph is built. A `SearchGraph`
never sees a raw object. An `AsymmetricSearchGraph` never inserts or searches with a code. Both
are `AbstractSearchGraph` and both answer the same search interface.

---

## The distance is the encoder

The graph knows nothing about codes. Its distance is an [`AbstractEstimator`](@ref). That is one
plain type whose parameters are fields. The graph calls only three things on it:

- [`encode`](@ref)`(est, obj)`: what the storage receives for a raw object;
- [`encodequery`](@ref)`(est, q)`: what a raw query becomes. It is applied **once per query and
  once per inserted item**. A rotation costs `dim²` operations, so applying it inside every
  evaluation would cost more than the evaluation;
- `evaluate(est, q, stored)` compares a raw query against a stored code, and
  `evaluate(est, a, b)` compares two stored codes. The neighborhood filters need the second form
  to compare the candidates of a new item among themselves.

Everything the model needs to interpret its codes is stored in the estimator or in the code
itself. A graph and its distance therefore serialize together. A model that can bound its own
error re-evaluates the distance *inside* `evaluate` when it needs to. The graph does not observe
this.

---

## `SQEncoder`: the quantizers as the encoder

[`ScalarQuant.SQEncoder`](@ref) packages the scalar quantizers of
[Quantization and Bit Sketches](quantization_and_bitsketches.md) for the asymmetric graph. It uses
the estimator interface because that is the interface the graph navigates with. It is a
codification and not an estimator: it bounds no error and it re-evaluates nothing. Its `evaluate`
is `ScalarQuant.SqL2`, or `L2`, `L1`, `NormCosine`, `Cosine`. It uses two kernels: a mixed kernel
for a `Float32` query against packed codes, and an integer kernel between two codes.

The quantizer is named by its module. The module name gives the family and the width. The storage
the graph grows is [`ScalarQuant.sqcodes`](@ref)`(enc)`. That is a `QuantDatabase` in dense blocks,
and it takes the encoder's codes without changing them:

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

On i.i.d. Gaussian vectors at 4 bits the two modes give the same recall. The codes carry the
query's neighbors in both cases. The difference is in the edges, and it appears on real data below
8 bits. The end of this page gives those measurements.

A rotation changes what the symmetric graph must receive. Its database quantizes with a range
fitted on the rotated coordinates, so it has to be given rotated vectors, `encodequery(enc, v)`.

| module | range | bits per coordinate | bytes per vector at 384-d |
| :--- | :--- | :--- | :--- |
| `SQgu8` | one for the whole database, fitted by `sqautorange` on a sample (or given as `minmax`) | 8 | 392 |
| `SQgu4` | global | 4 | 200 |
| `SQgu2` | global | 2 | 104 |
| `SQu8` | one per vector, from its own extrema; needs no data to build | 8 | 400 |
| `SQu4` | per vector | 4 | 208 |
| `SQu2` | per vector | 2 | 112 |

The per-vector modules take only the dimension: `SQEncoder(ScalarQuant.SQu8, 64)`. The global ones
take a matrix to fit their range on, or an explicit `minmax`.

### Rotating first, or not

`SQEncoder` accepts a rotation between the quantizer and the data:
`SQEncoder(quant, rotation, X)`. The rotation is a [`Projections.Rotation`](@ref) or `nothing`.
Two rotations are available. `Projections.qr(dim, dim)` is an orthogonal matrix and costs `dim²`
operations per vector. `Projections.RandomizedHadamard(dim)` applies a random sign to each
coordinate and then the Walsh-Hadamard transform; it costs `dim log dim` per vector and requires
`dim` to be a power of two. **The default is `nothing`**, so the two-argument forms above rotate
nothing.

A rotation gives the coordinates a shared scale. A global range is one minimum and one scale for
every coordinate of every vector. It assumes that the coordinates have a similar spread. If one
coordinate is a hundred times wider than another, the range fitted on all of them spends its
levels on the wide coordinate and flattens the narrow ones.

A random rotation mixes every coordinate into every other one. The rotated coordinates then share
one scale, whatever the original coordinates had.

If the coordinates already share a scale, a rotation changes nothing and still costs its
operations on every query and on every inserted item. Normalized text and image embeddings are in
this case. Measured on the SISAP 2025 `ccnews` embeddings (603,664 x 384; issue #86), a QR rotation moved
recall@10 by less than 0.01 at every width and in both families. It added 30 to 50 µs to each
query.

On synthetic data the result reverses as soon as the scales diverge. The table below uses 5,000
Gaussian vectors in 64 dimensions, 100 queries, an exhaustive scan over the codes, and
recall@10.

| coordinates | `SQgu4`, no rotation | `SQgu4`, QR | `SQu4`, no rotation | `SQu4`, QR |
| :--- | :--- | :--- | :--- | :--- |
| the same scale | 0.83 | 0.84 | 0.87 | 0.88 |
| scales from 1 to 100 | 0.79 | 0.85 | 0.88 | 0.89 |
| one coordinate 50x the rest | 0.12 | 0.28 | 0.43 | 0.67 |

Three rules follow. Leave the default when the coordinates share a scale, which is what a range
fitted by `sqautorange` assumes. Rotate when they do not and the memory budget requires the global
family. Consider the per-vector family instead: its range follows each vector, so it also handles
uneven scales, at a cost of 8 more bytes per vector and no rotation at all.

The per-vector family fails at 2 bits. On `ccnews` it reaches 0.65 of exhaustive recall against
0.82, with or without a rotation.

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

The object is rotated once, in `encode`. The query is rotated once, in `encodequery`. Nothing is
rotated inside an evaluation. An `AsymmetricSearchGraph` performs both steps. The exhaustive scan
above prepares the queries explicitly, because `ExhaustiveSearch` uses whatever it is given.

---

## `RaBitQ`: an estimator with an error bound

[`RaBitQ`](@ref) (Gao & Long, 2024) is an estimator. For each object it stores the sign bits of
the rotated vector, one bit per coordinate, and three scalars. The first scalar is the projection
of the unit vector onto its own sign vector, which normalizes the estimate. The second is the
norm. The third is the half-width of the confidence interval of the estimate.

The query is rotated and normalized once. The estimate of the cosine is then one signed sum over
the bits. It uses a SIMD kernel and takes 67 ns for a pair of 384 dimensions. The estimate is
unbiased, and [`RaBitQ.errorbound`](@ref) gives its bound for each object. Between two stored
codes, which the neighborhood filters need, the estimate is the SimHash estimate over the Hamming
distance of the bits.

Here the rotation is **required**: `RaBitQCosine(rotation)`. The estimate is unbiased and its
bound holds only because the sign vector is taken in a uniformly random basis.

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

The bits alone are weak on i.i.d. Gaussian vectors, and the printed numbers explain why. In 64
dimensions the true cosines spread by about `1/sqrt(64) = 0.125`, while the error bound is near
0.18. The estimate therefore cannot separate the neighbors from the rest. On data with structure
the same bits reach 0.69 of exhaustive recall@10 on `ccnews`. The fallback improves both cases.

[`RaBitQ.RaBitQRefined`](@ref) has two levels. The bits navigate the graph. When the bound cannot
exclude an object, the distance is re-evaluated from the fallback, *inside the same `evaluate`*.
Two fallbacks are available. [`RaBitQ.RaBitQExactFallback`](@ref) keeps the rotated vector in
`Float32` or `Float16`. [`RaBitQ.RaBitQVectorFallback`](@ref) keeps the same vector through one of
the quantizer modules.

There is no re-ranking pass after the search, and the graph never returns to raw data. The
correction is a re-evaluation, and the graph only ever observes a distance. `τ = Inf` re-evaluates
everything. [`RaBitQ.refinethreshold`](@ref) computes a `τ` from a sample of `k`-th neighbor
distances.

---

## Which graph, and which encoder

The measurements below come from `ccnews` (issue #86). The comparable columns are the ones with a
fixed beam.

- At 8 bits the asymmetric `SQEncoder` graph reaches the recall of the `Float32` graph at the same
  beam, and uses a quarter of the memory. The symmetric graph over the same codes is 30 to 40%
  faster per query and loses 0.002 of recall.
- Below 8 bits the asymmetric edges are better. Searched in exact precision they give 0.90 against
  0.88 at 4 bits, and 0.94 against 0.88 at 2 bits. A query evaluated against codes cannot use that
  advantage: both graphs give 0.81 at 4 bits and 0.65 at 2 bits. The asymmetric graph is worth its
  cost when its distance re-evaluates what the codes alone cannot resolve.
- The RaBitQ bits use 60 bytes for a vector of 384 dimensions and reach the recall of the 2-bit
  `SQEncoder`, which uses 104 bytes. The fallback is worth its cost in an exhaustive scan, where it
  reduces the time from 227 to 63 ms per query at the same recall. It is not worth its cost through
  an in-RAM graph, because the neighborhood filters there compare candidates by the bits.

The symmetric pipeline has its own packaged form. The next section,
[Sketched Search](sketchedsearch.md), encodes the data, indexes the codes, retrieves more
candidates than requested, and **re-scores them afterwards** under the real distance. That is a
pass over the raw data after the search. The asymmetric graph does not have such a pass and does
not need one.
