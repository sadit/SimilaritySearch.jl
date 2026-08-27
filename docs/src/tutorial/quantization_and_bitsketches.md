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

# 3. Construct an exact search index using Squared Euclidean distance
dist = Dist.SqL2()
idx = ExhaustiveSearch(dist, db_sq)
ctx = GenericContext()

# 4. Execute queries using unquantized Float32 vectors
queries = rand(Float32, dim, 5)
queries_db = MatrixDatabase(queries)

# The distance function evaluates asymmetric distances between quantized dataset vectors and Float32 queries
knns = searchbatch(idx, ctx, queries_db, 10)
```

Scalar quantization substantially reduces memory footprint while maintaining high fidelity in nearest-neighbor rankings through asymmetric distance computation.

---

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
