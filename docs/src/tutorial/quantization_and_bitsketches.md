```@meta
CurrentModule = SimilaritySearch
```

# Quantization and Bit Sketches

In large-scale similarity search, storing millions of high-dimensional vectors in standard single-precision floating-point format (`Float32`) presents memory and bandwidth bottlenecks. `SimilaritySearch.jl` provides two vector compression and acceleration strategies:

1. **Scalar Quantization ([`ScalarQuant`](@ref))**: Maps continuous floating-point coordinates to low-bit integer representations using column-wise affine scaling.
2. **Bit Sketches ([`Projections.bitsketch`](@ref))**: Projects high-dimensional continuous vectors onto binary signatures via random hyperplane projections (SimHash / Locality Sensitive Hashing), enabling fast Hamming distance evaluations.

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
