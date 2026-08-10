```@meta
CurrentModule = SimilaritySearch
```

# Quantization and Bit Sketches

In scenarios where memory constraints are a primary concern or where accelerated exact searches are required, `SimilaritySearch.jl` provides mechanisms to compress vectors into representations with a smaller memory footprint: **Scalar Quantization** and **Bit Sketches**. 

This section demonstrates how to apply these techniques to reduce the dataset size while performing nearest neighbor queries using `ExhaustiveSearch`.

## Scalar Quantization (SQu8)

Scalar quantization maps each coordinate of a vector to a lower-precision integer representation. The `ScalarQuant` module provides different bit-depths, including 8-bit (`SQu8`), 4-bit (`SQu4`), and 2-bit (`SQu2`). 

Using 8-bit quantization (`SQu8`) reduces 32-bit `Float32` vectors by a factor of four, storing one `UInt8` per coordinate, along with the minimum value and scaling factor per column to allow approximate dequantization.

The following example illustrates how to quantize a dataset and execute a search:

```julia
using SimilaritySearch
using SimilaritySearch.ScalarQuant

# Generate synthetic data
dim = 32
n = 10_000
X = rand(Float32, dim, n)

# Quantize the dataset to 8 bits per coordinate
db_sq = ScalarQuant.SQu8.quantize(X)

# Construct an exact exhaustive search index using SqL2 distance
dist = Dist.SqL2()
idx = ExhaustiveSearch(dist, db_sq)
ctx = GenericContext()

# Search using raw Float32 queries
queries = rand(Float32, dim, 5)
queries_db = MatrixDatabase(queries)

# The distance function handles the evaluation between the SQu8Vec and Float32 query pair
knns = searchbatch(idx, ctx, queries_db, 10)
```

Storing the dataset as `SQu8` reduces the overall memory requirements. This compression involves a trade-off, introducing a measured decrease in numerical precision during distance evaluations.

## Bit Sketches

For further reduction of the memory footprint and to accelerate searches, continuous vectors can be projected into binary signatures, referred to as Bit Sketches. A bit sketch applies a random projection matrix to the data and encodes the sign of the resulting projection into bits (stored as `UInt64`). 

Searching across bit sketches involves bitwise distances (such as the Hamming distance), which require fewer CPU cycles to compute compared to standard floating-point operations.

The following code demonstrates how to compute bit sketches and query them:

```julia
using SimilaritySearch
using SimilaritySearch.Projections: bitsketch

# 1. Sketch the database
# Apply a gaussian random projection to map the 32-dimensional vectors to 256 bits (four UInt64 words)
B, R = bitsketch(:gaussian, 256, X)

# B is a Matrix{UInt64} of size (4, 10_000)
db_bits = MatrixDatabase(B)

# 2. Build the exact index using a binary distance metric
dist_bits = Dist.Bits.Hamming()
idx_bits = ExhaustiveSearch(dist_bits, db_bits)

# 3. Sketch the queries using the same rotation matrix R
# Queries must be projected into the same binary space as the database
bq = bitsketch(R, queries)
queries_bits_db = MatrixDatabase(bq)

# 4. Search
knns_bits = searchbatch(idx_bits, ctx, queries_bits_db, 10)
```

In this pipeline, `bitsketch` performs the linear projection of the data with `R` and the packing of the resulting signs into `UInt64` arrays. This approach yields a compact representation and can accelerate the exhaustive search process by simplifying the distance calculations.
