```@meta
CurrentModule = SimilaritySearch
```

# Quantization and Bit Sketches

Storing millions of high-dimensional vectors as `Float32` uses a large amount of memory and
bandwidth. `SimilaritySearch.jl` provides three strategies that reduce both:

1. **Scalar Quantization ([`ScalarQuant`](@ref))**: Maps continuous floating-point coordinates to low-bit integer representations using column-wise affine scaling.
2. **Projection-Based Bit Sketches ([`Projections.bitsketch`](@ref))**: Projects high-dimensional continuous vectors onto binary signatures via a random ([`Projections.RandomProjections`](@ref), [`Projections.HadamardProjection`](@ref)) or data-fitted ([`Projections.PCAProjection`](@ref)) rotation (SimHash / Locality Sensitive Hashing), enabling fast Hamming distance evaluations.
3. **Hyperplane Bit Sketches ([`Projections.DistantHyperplanes`](@ref),
   [`Projections.RandomHyperplanes`](@ref))**: Encode objects of *any* metric space, not only
   floating-point vectors. Each bit records which side of a hyperplane the object falls on. The
   hyperplanes are defined through the distance function of the space itself.

---

## Scalar Quantization (`ScalarQuant`)

Scalar quantization approximates each coordinate $x_i \in \mathbb{R}$ of a vector by mapping it to a discrete integer grid of $b$ bits:

$$q_i = \text{round}\left( \frac{x_i - \min(X)}{\text{scale}} \right)$$

The `ScalarQuant` module provides multiple bit-depth representations:
- **`SQu8` (8-bit)**: Compresses `Float32` vectors by a factor of 4$\times$, storing each coordinate in a single `UInt8` alongside column-wise scale and offset parameters.
- **`SQu4` (4-bit)**: Compresses by 8$\times$.
- **`SQu2` (2-bit)**: Compresses by 16$\times$.

Each of them keeps one `min` and scale pair *per column*, in `E`, next to the packed codes in
`Q`. Those two fields are everything a database contains. A stored database can therefore be
rebuilt without the original matrix: `SQu8Database(E, Q)` quantizes nothing and recomputes what
it derives.

This matters at scale. Re-quantizing a 64 x 730,320 projection to recover 46.7 MB of codes would
first rebuild 187 MB of `Float32` values.

### Example: Quantization and Search with `SQu8`

```julia
using SimilaritySearch
using SimilaritySearch.ScalarQuant

# 1. Generate synthetic continuous dataset
dim = 32
n = 10_000
X = rand(Float32, dim, n)

# 2. Quantize dataset to 8 bits per coordinate
db_sq = ScalarQuant.SQu8.quantize(X)

# 3. Construct an exact search index. The distance must be the quantizer's own: it reads the
#    packed codes directly, while `Dist.SqL2()` is for plain Float32 vectors and has no method
#    for a quantized one.
dist = Dist.SqL2()                       # kept for the plain-vector examples further down
qdist = ScalarQuant.SQu8.SqL2()          # what a quantized database is searched with
idx = ExhaustiveSearch(qdist, db_sq)
ctx = GenericContext()

# 4. Execute queries using unquantized Float32 vectors
queries = rand(Float32, dim, 5)
queries_db = MatrixDatabase(queries)

# The distance function evaluates asymmetric distances between quantized dataset vectors and Float32 queries
knns = searchbatch(idx, ctx, queries_db, 10)
```

Scalar quantization reduces the memory footprint. Asymmetric distance computation keeps the
nearest-neighbor ranking close to the exact one.

### Global quantization, stored parameters, and raw queries

The `SQgu2`, `SQgu4` and `SQgu8` variants share **one** `min` and scale pair across the whole
dataset. Their kernels are cheaper for that reason: a shared scale cancels in a difference, so
codes are compared as integers.

`quantize` returns a bare `Matrix{UInt8}` and leaves the pair to the caller. Codes stored without
the pair cannot be dequantized. They can only be compared against codes from the same run.

`GlobalQuantDatabase` keeps the pair, and the per-vector code sums:

```julia
using SimilaritySearch
using SimilaritySearch.ScalarQuant

Xg = randn(Float32, 64, 10_000)
gdb = ScalarQuant.GlobalQuantDatabase(8, Xg; minmax=extrema(Xg))

qg = randn(Float32, 64)                                  # a *raw* query, not quantized
gidx = ExhaustiveSearch(ScalarQuant.SQu8.SqL2(), gdb)
gres = search(gidx, GenericContext(), qg, knnqueue(KnnSorted, 10))

# and cosine, which needs the stored sums
cidx = ExhaustiveSearch(ScalarQuant.Cosine(), gdb)
cres = search(cidx, GenericContext(), ScalarQuant.quantize(gdb, qg), knnqueue(KnnSorted, 10))
```

Indexing it yields the same `SQVec` that the per-column quantizers produce. A globally quantized
vector *is* a per-column one whose scale is shared. Every distance in `ScalarQuant` therefore
applies, and each one takes its best path: an exact integer pass between two stored vectors, and
a mixed kernel against a plain `Float32` query.

### Growing a quantized database

Both families are a `QuantDatabase`: the parameters plus some database of code vectors. That
inner database is a `MatrixDatabase` when a matrix is quantized in one call, and it can be any
other database.

Start from an empty growable storage. The database then quantizes each vector as it arrives, with
the parameters it was created with. A `SearchGraph` can therefore build over it one item at a
time, and a `MMapMatrixDatabase` keeps the codes on disk across processes:

```julia
# SimilaritySearch v1.5
using SimilaritySearch, SimilaritySearch.ScalarQuant

X = randn(Float32, 64, 10_000)
mm = extrema(X)
gdb = GlobalQuantDatabase(8, BlockMatrixDatabase(64, UInt8), mm; dim=64)  # empty; 64 bytes per 8-bit vector
G = SearchGraph(ScalarQuant.SqL2(), gdb)
ctx = SearchGraphContext(; reporters=[])
append_items!(G, ctx, MatrixDatabase(X))                # quantized on the way in
res = search(G, ctx, randn(Float32, 64), knnqueue(KnnSorted, 10))   # a raw query, mixed kernel
gdb == GlobalQuantDatabase(8, X; minmax=mm)             # true: the same codes, byte for byte

# the per-vector family grows the same way
pdb = ScalarQuant.SQu4.SQu4Database(ScalarQuant.SQMinC[], BlockMatrixDatabase(32, UInt8); dim=64)
push_item!(pdb, X[:, 1])
```

`GlobalQuantDatabase(8, MMapMatrixDatabase(path), mm)` reopens a database whose codes were pushed
into an mmap file. `E` travels with it, and so do the sums `Sa` and `Saa` if they were kept.
Giving those sums back skips the single pass over the codes that would otherwise recompute
them.

### Symmetric and asymmetric graphs over quantized storage

A graph stored as codes can work in two ways. The way is a property of the instance.

A `SearchGraph` over a `QuantDatabase` is the **symmetric** graph. It inserts and searches with
the objects as the database stores them. Both sides of every evaluation are codes, and the exact
integer kernel is used.

An `AsymmetricSearchGraph` over the same database also stores codes, but it inserts and searches
with the **raw** objects. Each raw object is evaluated against the stored codes by the mixed
kernel. Each new item therefore picks its neighbors by its exact distance, and the quantization
error does not enter the edges.

A query can be passed raw to either graph. The symmetric one also accepts it as codes,
`quantize(database(G), q)`.

```julia
# SimilaritySearch v1.5
using SimilaritySearch, SimilaritySearch.ScalarQuant

X = randn(Float32, 64, 10_000)
mm = extrema(X)
ctx = SearchGraphContext(; reporters=[])
q = randn(Float32, 64)

sym = SearchGraph(ScalarQuant.SqL2(), GlobalQuantDatabase(4, BlockMatrixDatabase(32, UInt8), mm; dim=64))
append_items!(sym, ctx, MatrixDatabase(X))                   # edges chosen on codes
res_raw = search(sym, ctx, q, knnqueue(KnnSorted, 10))       # query in Float32, against codes
res_codes = search(sym, ctx, ScalarQuant.quantize(database(sym), q), knnqueue(KnnSorted, 10))

asym = AsymmetricSearchGraph(ScalarQuant.SqL2(), GlobalQuantDatabase(4, BlockMatrixDatabase(32, UInt8), mm; dim=64))
append_items!(asym, ctx, MatrixDatabase(X))                  # edges chosen on Float32 vs codes; storage at 4 bits
res_asym = search(asym, ctx, q, knnqueue(KnnSorted, 10))
```

Quantization error at insertion time stays in the topology permanently. At query time it can
still be corrected by re-ranking.

For a fixed graph, a `Float32` query never gives a worse result than a quantized one. The same
edges searched over the full-precision vectors never give a worse result than either. The tests
assert exactly that ordering.

On the SISAP 2025 `ccnews` benchmark (issue #86) the asymmetric edges are better below 8 bits. A
query evaluated against codes cannot use that advantage. Use the asymmetric graph for queries
that will be re-scored in higher precision, and the symmetric graph, which is cheaper at every
step, for the rest.

That re-evaluation belongs to the distance. The graph only evaluates `dist(q, stored)`. `dist`
receives everything the model has: the raw query, the encoded object together with whatever the
model stored beside the code, and the model's own parameters.

The distances of the scalar quantizers are the simple case. They produce an estimate that needs
no correction.

A distance that is an *estimator* has an error of its own. A sketch compared against a raw query
is one example, and a RaBitQ-style code is another. Such a distance is an `AbstractEstimator`. It
declares through `encode(est, obj)` what the storage receives, and through `encodequery(est, q)`
what a raw query becomes. A rotation is applied there, once per query instead of once per
evaluation. Inside its `evaluate` the estimator bounds its error and re-evaluates the distance
when it needs to. The graph does not observe this.

An estimator is one plain type with its parameters as fields, so the graph and the information
that gives its codes meaning serialize together. The package provides two estimators,
`ScalarQuant.SQEncoder` and `RaBitQ`. They have their own section,
[Asymmetric Search: raw queries against codes](asymmetric.md), after the sketches below.

## Bit Sketches: Binary Random Projections

Bit sketches map continuous vectors $x \in \mathbb{R}^d$ into compact binary signatures $b \in \{0, 1\}^m$ using random hyperplane projections:

$$b_i = \begin{cases} 1 & \text{if } \langle r_i, x \rangle \ge 0 \\ 0 & \text{if } \langle r_i, x \rangle < 0 \end{cases}$$

where $R = [r_1, \dots, r_m]^T$ is a random projection matrix (e.g., drawn from a standard Gaussian distribution $\mathcal{N}(0, I)$).

Binary signatures are packed into arrays of `UInt64` words. The **Hamming distance** then
approximates the angular similarity. It counts the differing bits with the hardware `POPCNT`
instruction.

### Example: Generating and Querying Bit Sketches

```julia
using SimilaritySearch
using SimilaritySearch.Projections: bitsketch

# 1. Project dataset vectors into 256-bit sketches (4 × UInt64 words per vector)
B, R = bitsketch(:gaussian, 256, X)
db_bits = MatrixDatabase(B)

# 2. Construct an exact search index using binary Hamming distance
dist_bits = Dist.Bits.Hamming()
idx_bits = ExhaustiveSearch(dist_bits, db_bits)

# 3. Project query vectors using the same projection matrix R
bq = bitsketch(R, queries)
queries_bits_db = MatrixDatabase(bq)

# 4. Execute batch search over the binary representations
knns_bits = searchbatch(idx_bits, ctx, queries_bits_db, 10)
```

Bit sketches give high throughput and low memory for high-dimensional embeddings. They can also
serve as a first filtering stage before a re-ranking pass in full precision.

### PCA-Fitted Bit Sketches

`bitsketch` works with any rotation that implements [`Projections.transform`](@ref). The random
matrix `R` above can therefore be replaced by a rotation *fitted from data*,
[`Projections.PCAProjection`](@ref), without changing the rest of the pipeline.

`RandomProjections` and `HadamardProjection` do not depend on the data. A `PCAProjection` does:
it depends on the sample it was fitted from. Reuse the same object, and not a newly built one, to
sketch anything that will be compared against an already-sketched dataset:

```julia
using SimilaritySearch.Projections: PCAProjection, bitsketch

p = PCAProjection(X, 256)         # fit 256 principal directions from X
B_pca = bitsketch(p, X)
bq_pca = bitsketch(p, queries)    # same p, so sketches stay comparable to B_pca's columns
```

---

## Hyperplane Bit Sketches for Generic Metric Spaces

All the bit sketches above require the dataset to live in $\mathbb{R}^d$. They `transform`
(rotate or project) raw coordinate vectors before packing the signs into bits. Some objects
support only a distance function, and then there is nothing to rotate. The prime-factor sets of
this tutorial, under the Dice distance, are such a case (see the [Quickstart](index.md)).
[`Projections.DistantHyperplanes`](@ref), [`Projections.AnchoredDistantHyperplanes`](@ref),
and [`Projections.RandomHyperplanes`](@ref) sketch *any* `SemiMetric`/`AbstractDatabase`
instead. A hyperplane here is a pair of anchor objects $(i, j)$ taken from the dataset. An object
$x$ is encoded by the side of that hyperplane it falls on:

$$b = \begin{cases} 1 & \text{if } d(x, i) \le d(x, j) \\ 0 & \text{otherwise} \end{cases}$$

- **[`DistantHyperplanes`](@ref Projections.DistantHyperplanes)** samples many candidate anchor
  pairs and discards the uninformative ones, which are those with low entropy over a data sample.
  It then keeps a mutually diverse subset with [`fft`](@ref). Diversity is measured under a
  flip-invariant Hamming distance, because swapping the two anchors of a pair describes the same
  hyperplane.
- **[`AnchoredDistantHyperplanes`](@ref Projections.AnchoredDistantHyperplanes)** follows the same
  idea. It orients every candidate pair by the distance to a reference `anchor` object, so plain
  Hamming distance is enough during selection. The anchor is given explicitly or chosen
  automatically by an `anchorpolicy`.
- **[`RandomHyperplanes`](@ref Projections.RandomHyperplanes)** performs no search. The caller
  supplies the anchor pairs directly, for example a plain random sample. The fit is much cheaper
  and the sketch quality is lower.

All three expose the same [`distance`](@ref) (Hamming, over the packed sketch),
[`Projections.outdim`](@ref), and [`Projections.bitsketch`](@ref) used above.

### Example: Sketching Sets Under the Dice Distance

Reusing the prime-factor dataset `X` and Dice `dist` from the [Quickstart](index.md):

```julia
using SimilaritySearch
using SimilaritySearch.Projections: DistantHyperplanes, bitsketch

# henc/hsel are shrunk from their defaults to fit this tutorial's small n = 1000
m = DistantHyperplanes(dist, X, 64; henc=512, hsel=4096, verbose=false)
B = bitsketch(m, X)                          # a (1, 1000) MatrixDatabase{Matrix{UInt64}}

idx_bits = ExhaustiveSearch(distance(m), B)
bq = bitsketch(m, factors(1000))
res = knnqueue(ctx, 5)
search(idx_bits, ctx, bq, res)

[p.id for p in IdDistView(res)]   # 10, 20, 40, 50, 80 -- the exact-Dice result, from bit sketches alone
```

`AnchoredDistantHyperplanes` and `RandomHyperplanes` are drop-in replacements for `m` in the
snippet above; only their construction differs (see their docstrings for the extra keyword
arguments each one takes).

In the next section, [Multi-Bit Sketches](multibit_sketches.md), we keep more than one bit per
hyperplane. It is the same fitted model, with the magnitude it was already computing.
