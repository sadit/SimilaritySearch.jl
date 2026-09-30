```@meta
CurrentModule = SimilaritySearch
```

# Quantization and Bit Sketches

In large-scale similarity search, storing millions of high-dimensional vectors in standard single-precision floating-point format (`Float32`) presents memory and bandwidth bottlenecks. `SimilaritySearch.jl` provides three compression and acceleration strategies:

1. **Scalar Quantization ([`ScalarQuant`](@ref))**: Maps continuous floating-point coordinates to low-bit integer representations using column-wise affine scaling.
2. **Projection-Based Bit Sketches ([`Projections.bitsketch`](@ref))**: Projects high-dimensional continuous vectors onto binary signatures via a random ([`Projections.RandomProjections`](@ref), [`Projections.HadamardProjection`](@ref)) or data-fitted ([`Projections.PCAProjection`](@ref)) rotation (SimHash / Locality Sensitive Hashing), enabling fast Hamming distance evaluations.
3. **Hyperplane Bit Sketches ([`Projections.DistantHyperplanes`](@ref), [`Projections.RandomHyperplanes`](@ref))**: Encode objects of *any* metric space -- not just floating-point vectors -- by which side of a set of hyperplanes, defined directly through the space's own distance function, they fall on.

---

## Scalar Quantization (`ScalarQuant`)

Scalar quantization approximates each coordinate $x_i \in \mathbb{R}$ of a vector by mapping it to a discrete integer grid of $b$ bits:

$$q_i = \text{round}\left( \frac{x_i - \min(X)}{\text{scale}} \right)$$

The `ScalarQuant` module provides multiple bit-depth representations:
- **`SQu8` (8-bit)**: Compresses `Float32` vectors by a factor of 4$\times$, storing each coordinate in a single `UInt8` alongside column-wise scale and offset parameters.
- **`SQu4` (4-bit)**: Compresses by 8$\times$.
- **`SQu2` (2-bit)**: Compresses by 16$\times$.

Each of them keeps one `min`/scale pair *per column*, in `E`, next to the packed codes in `Q`.
Those two fields are everything a database is, so a stored one can be rebuilt without the
original matrix -- `SQu8Database(E, Q)` quantizes nothing and recomputes what it derives. That
matters at scale: re-quantizing a 64 x 730,320 projection to recover 46.7 MB of codes would
first rebuild 187 MB of `Float32`.

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

Scalar quantization substantially reduces memory footprint while maintaining high fidelity in nearest-neighbor rankings through asymmetric distance computation.

### Global quantization, stored parameters, and raw queries

The `SQgu2`/`SQgu4`/`SQgu8` variants share **one** `min`/scale pair across the whole dataset.
That makes their kernels cheaper -- a shared scale cancels in a difference, so codes are
compared as integers -- but `quantize` hands back a bare `Matrix{UInt8}` and leaves the pair
with the caller. Codes stored without it cannot be dequantized at all, and can only ever be
compared against codes from the same run.

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

Indexing it yields the same `SQVec` the per-column quantizers produce -- a globally quantized
vector *is* a per-column one whose scale happens to be shared -- so every distance in
`ScalarQuant` applies, and each takes its best path: an exact integer pass between two stored
vectors, and the mixed kernels against a plain `Float32` query.

### Growing a quantized database

Both families are a `QuantDatabase`: the parameters plus *some* database of code vectors,
which is `MatrixDatabase` when a matrix is quantized in one shot and can be any other one.
Start from an empty growable storage and the database quantizes each vector on the way in
with the parameters it was created with, so a `SearchGraph` builds over it one item at a
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

`GlobalQuantDatabase(8, MMapMatrixDatabase(path), mm)` reopens a database whose codes were
pushed into an mmap file; `E` and, if kept, the sums `Sa`/`Saa` travel with it, and a
`Sa`/`Saa` handed back skips the one pass over the codes that otherwise recomputes them.

### Symmetric and asymmetric graphs over quantized storage

A graph stored as codes can work in two ways, and the way is a property of the instance. A
`SearchGraph` over a `QuantDatabase` is the **symmetric** graph: it inserts and searches
with the objects as the database stores them, codes against codes on both sides, through
the exact integer kernel. An `AsymmetricSearchGraph` over the same database stores codes
too, but inserts and searches with the **raw** objects, evaluated against the stored codes
by the mixed kernel: each new item picks its neighbors by its exact distance, so the
quantization error is not baked into the edges. A query can be passed raw to either graph;
the symmetric one also takes it as codes, `quantize(database(G), q)`.

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

Quantization error at insertion time is baked into the topology for good; at query time it
is recoverable by re-ranking. For a fixed graph, a `Float32` query never does worse than a
quantized one, and the same edges searched over the full-precision vectors never do worse
than either; the tests assert exactly that ordering. On the SISAP 2025 `ccnews` benchmark
(issue #86) the asymmetric edges are better below 8 bits, but a query evaluated against
codes cannot cash the difference, so the asymmetric graph is for queries that will be
re-scored in higher precision and the symmetric one, cheaper at every step, for the rest.

That re-evaluation belongs to the distance. The graph only ever evaluates
`dist(q, stored)`, and `dist` receives everything a model has: the raw query, the encoded
object with whatever the model kept beside the code, and its own parameters. The scalar
quantizers' distances are the no-op case, an estimate that needs no correction. A
distance that is an *estimator* with an error of its own -- a sketch against a raw query, a
RaBitQ-style code -- is an `AbstractEstimator`: it says through `encode(est, obj)` what the
storage receives and through `encodequery(est, q)` what a raw query becomes (a rotation,
applied once per query rather than per evaluation), and inside its `evaluate` it bounds its
error and re-evaluates when it must, transparently to the graph. One plain type with its parameters as fields, so the
graph and what gives its codes meaning serialize together. The two estimators that ship with
the package, `ScalarQuant.SQEncoder` and `RaBitQ`, have their own section,
[Asymmetric Search: raw queries against codes](asymmetric.md), after the sketches below.

## Bit Sketches: Binary Random Projections

Bit sketches map continuous vectors $x \in \mathbb{R}^d$ into compact binary signatures $b \in \{0, 1\}^m$ using random hyperplane projections:

$$b_i = \begin{cases} 1 & \text{if } \langle r_i, x \rangle \ge 0 \\ 0 & \text{if } \langle r_i, x \rangle < 0 \end{cases}$$

where $R = [r_1, \dots, r_m]^T$ is a random projection matrix (e.g., drawn from a standard Gaussian distribution $\mathcal{N}(0, I)$).

Binary signatures are packed into arrays of `UInt64` words. In this binary representation, angular similarity is approximated by the **Hamming distance**, which evaluates bitwise differences via hardware-accelerated bit-population count (`POPCNT`) instructions.

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

Bit sketches provide a high-throughput, low-memory indexing option for high-dimensional embedding search, and can be used as a coarse-filtering stage prior to full-precision re-ranking.

### PCA-Fitted Bit Sketches

`bitsketch` works with any rotation that implements [`Projections.transform`](@ref), so the
random matrix `R` above can be swapped for a rotation *fitted from data* --
[`Projections.PCAProjection`](@ref) -- without touching the rest of the pipeline. Unlike
`RandomProjections`/`HadamardProjection`, a `PCAProjection` depends on the sample it was
fitted from, so the same object (not a freshly-built one) must be reused to sketch
anything compared against an already-sketched dataset:

```julia
using SimilaritySearch.Projections: PCAProjection, bitsketch

p = PCAProjection(X, 256)         # fit 256 principal directions from X
B_pca = bitsketch(p, X)
bq_pca = bitsketch(p, queries)    # same p, so sketches stay comparable to B_pca's columns
```

---

## Hyperplane Bit Sketches for Generic Metric Spaces

The bit sketches above all require the dataset to live in $\mathbb{R}^d$: they `transform`
(rotate/project) raw coordinate vectors before packing signs into bits. When objects only
support a distance function -- e.g. this tutorial's running prime-factor sets under the
Dice distance (see the [Quickstart](index.md)) -- there is nothing to rotate.
[`Projections.DistantHyperplanes`](@ref), [`Projections.AnchoredDistantHyperplanes`](@ref),
and [`Projections.RandomHyperplanes`](@ref) sketch *any* `SemiMetric`/`AbstractDatabase`
instead: an object $x$ is encoded by which side of a hyperplane -- a pair of anchor objects
$(i, j)$ from the dataset -- it falls on:

$$b = \begin{cases} 1 & \text{if } d(x, i) \le d(x, j) \\ 0 & \text{otherwise} \end{cases}$$

- **[`DistantHyperplanes`](@ref Projections.DistantHyperplanes)** samples many candidate
  anchor pairs, discards the uninformative ones (low entropy over a data sample), and keeps
  a mutually diverse subset via [`fft`](@ref) -- diverse under a flip-invariant Hamming
  distance, since swapping a pair's two anchors describes the exact same hyperplane.
- **[`AnchoredDistantHyperplanes`](@ref Projections.AnchoredDistantHyperplanes)** is the
  same idea, but orients every candidate pair by distance to a reference `anchor` object
  (given explicitly, or picked automatically per an `anchorpolicy`) instead, so plain
  Hamming distance is enough during selection.
- **[`RandomHyperplanes`](@ref Projections.RandomHyperplanes)** skips the search entirely:
  the caller supplies the anchor pairs directly (e.g. a plain random sample), trading
  sketch quality for a much cheaper fit.

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
hyperplane -- the same fitted model, with the magnitude it was already computing.
