```@meta
CurrentModule = SimilaritySearch
```

# Bichromatic Operations and Metric Joins

Monochromatic search operations evaluate distances among objects within a single dataset $A$. In contrast, **bichromatic** operations evaluate distances across the Cartesian product $A \times B$ of two distinct datasets $A, B \subseteq \mathcal{U}$ under a shared distance metric $d$.

`SimilaritySearch.jl` implements bichromatic algorithms in the `Bichromatic` submodule (re-exported at the top level):
- [`bichromatic_closestpair`](@ref): Finds the globally closest pair $(a, b) \in A \times B$.
- [`bichromatic_kclosestpairs`](@ref): Finds the $k$ closest pairs in $A \times B$.
- [`bichromatic_metricjoin`](@ref): Performs an adaptive metric join between $A$ and $B$, handling regions of non-uniform density.

---

## Dataset Setup

Consider a set of four reference centers $A$ (e.g., facility locations) and six query points $B$ (e.g., client locations) in $\mathbb{R}^2$:

```julia
using SimilaritySearch, Distances

dist = Dist.L2()

# Reference points A: four corners of a 10x10 square
A = MatrixDatabase(Float32[0 10 0 10; 0 0 10 10])

# Query points B: points located near the corners, a center point, and a point near corner 1
B = MatrixDatabase(Float32[1 9 1 9 5 0.5; 1 1 9 9 5 0.5])
```

---

## Minimal Pair: `bichromatic_closestpair`

To find the globally closest pair $(a, b) \in A \times B$, index dataset $A$ and query with dataset $B$:

```julia
idxA = ExhaustiveSearch(dist, A)
ctx = GenericContext()

i, j, d = bichromatic_closestpair(idxA, ctx, B)
# Result: (1, 6, 0.70710677f0)
```

The algorithm returns:
- `i`: Index of the matching object in $A$ (Center 1 at $(0,0)$).
- `j`: Index of the matching object in $B$ (Point 6 at $(0.5, 0.5)$).
- `d`: Distance between $A[i]$ and $B[j]$ ($d = \sqrt{0.5^2 + 0.5^2} \approx 0.7071$).

---

## Top-$k$ Pairs: `bichromatic_kclosestpairs`

To retrieve the $k$ closest pairs across $A \times B$:

```julia
pairs = bichromatic_kclosestpairs(idxA, ctx, B; k=5)
for p in pairs
    println("A ID: ", p[1], " | B ID: ", p[2], " | Distance: ", p[3])
end
```

The output contains the minimal pair followed by the remaining nearest cross-dataset associations.

---

## Monochromatic Equivalence and the `samedata` Keyword

Monochromatic closest pair search ([`closestpair`](@ref)) is a special case of `bichromatic_closestpair` where $B = A$ and self-comparisons ($i = j$) are excluded:

```julia
closestpair(idxA, ctx)
bichromatic_closestpair(idxA, ctx, database(idxA))
```

### Identity vs. Equality (`samedata`)

Self-match filtering is determined by the `samedata` keyword, which defaults to object identity (`database(idxA) === B`).

If $B$ is a distinct container with identical values to $A$, `database(idxA) === B` evaluates to `false`, causing the algorithm to report distance $0.0$ self-matches. To exclude self-matches when passing distinct database objects with matching contents, set `samedata=true` explicitly:

```julia
B2 = MatrixDatabase(copy(A.matrix))  # Distinct container holding identical coordinates
bichromatic_closestpair(idxA, ctx, B2; samedata=true)
```

---

## High-Level Convenience Signatures

For direct execution without manually instantiating an index, pass the metric and datasets directly:

```julia
# Exact bichromatic search
i, j, d = bichromatic_closestpair(dist, A, B)

# Approximate bichromatic search with target recall
i, j, d = bichromatic_closestpair(dist, A, B; recall=0.9)
```

When `recall < 1.0`, an approximate `SearchGraph` is constructed internally and calibrated to the requested recall target.

---

## Adaptive Metric Joins: `bichromatic_metricjoin`

Standard distance joins filter pairs by a fixed global radius $r$:

$$\text{Join}(A, B, r) = \{ (a, b) \in A \times B \mid d(a, b) \le r \}$$

However, in datasets with heterogeneous density distributions, a single global threshold $r$ produces suboptimal results: dense regions generate excessive matches while sparse regions yield no matches.

[`bichromatic_metricjoin`](@ref) computes an **adaptive metric join**. It estimates local density around each center $a \in A$ from the distribution of its $k$ candidate neighbors in $B$, dynamically calculating localized distance cutoffs:

```julia
using Random
Random.seed!(42)

# Generate synthetic non-uniform clusters
dense   = 0.0f0   .+ 5.0f0  .* randn(Float32, 2, 40)    # High-density cluster near (0,0)
sparse_ = 100.0f0 .+ 15.0f0 .* randn(Float32, 2, 8)     # Low-density cluster near (100,100)
near2   = [100.0f0, 0.0f0] .+ 5.0f0 .* randn(Float32, 2, 10)
near3   = [0.0f0, 100.0f0] .+ 5.0f0 .* randn(Float32, 2, 10)

Awide = MatrixDatabase(Float32[0 100 0 100; 0 0 100 100])
Bwide = MatrixDatabase(Float32.(hcat(dense, sparse_, near2, near3)))   # 68 total query points

idxAwide = ExhaustiveSearch(dist, Awide)
ctxwide = GenericContext()

pairs = bichromatic_metricjoin(idxAwide, ctxwide, Bwide; k=8)
```

By adapting cutoffs to local point densities, `bichromatic_metricjoin` achieves consistent match rates across both dense and sparse regions.

---

In the next section, [Parallelism and Multithreading](parallelism.md), we discuss the `@BATCHES` multithreading framework and thread-safety considerations.
