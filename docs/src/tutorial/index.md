```@meta
CurrentModule = SimilaritySearch
```

# Tutorial

This tutorial introduces `SimilaritySearch.jl` systematically from foundational concepts to advanced indexing strategies. To ensure reproducibility and eliminate external dependencies, the examples throughout this guide use small, self-contained synthetic datasets that can be verified analytically.

Across several sections, we explore different mathematical representations of integers:
- As sets of distinct prime factors (set metrics).
- As factor sequences with multiplicity (sequence and edit distances).
- As binary divisibility signatures (Hamming distances).
- As prime-gap vectors (continuous geometric distances in $\mathbb{R}^d$).

This recurring theme demonstrates how different distance functions and data representations interact with search algorithms.

## Tutorial Structure

We recommend reading the tutorial in the following order:

1. **[Tutorial Overview and Quickstart](index.md)** (this page) -- Installation and a basic working example.
2. **[Databases: Data Abstraction with `AbstractDatabase`](databases.md)** -- The database abstraction layer and memory layout considerations.
3. **[Distance Functions and Metric Spaces](distances.md)** -- Vector, set, sequence, and binary distances, with a theoretical discussion on graph navigability in continuous versus discrete spaces.
4. **[SearchGraph: Approximate Proximity Graphs](searchgraph.md)** -- Construction, query execution, hyperparameter optimization (`optimize_index!`), and graph rebuilding.
5. **[Radius Queries: Range-Bounded Search](radius_search.md)** -- Retrieving all elements within a fixed distance threshold using `RadiusSorted` and `RadiusHeap`.
6. **[Dataset Operations: Selection, All-kNN, and Closest Pairs](operations.md)** -- Advanced algorithms including Farthest First Traversal (`fft`), all-pairs $k$-NN (`allknn`), closest pair search, and near-duplicate elimination (`neardup`).
7. **[Bichromatic Operations and Metric Joins](bichromatic.md)** -- Closest pairs and metric joins between two distinct datasets ($A \times B$).
8. **[Parallelism and Multithreading](parallelism.md)** -- The `@BATCHES` execution model, context thread safety, and concurrency best practices.
9. **[Index Persistence and Serialization](persistence.md)** -- Serializing indexes with JLD2 and decoupling graph topology from dataset storage.
10. **[Logging and Observation Channels](logging.md)** -- Informational reporting (`reporters`) versus structural event listening (`observers`) for incremental graph tracking.
11. **[Inverted Files and Posting List Intersections](invertedfiles.md)** -- Inverted indexing (`InvertedFile`, `DictInvertedFile`) for sparse vectors, set metrics, and Maximum Inner Product Search (MIPS).
12. **[Quantization and Bit Sketches](quantization_and_bitsketches.md)** -- Compressing vectors to reduce memory usage and accelerate distance evaluations via `ScalarQuant` and `bitsketch`.

---

## Installation

Install `SimilaritySearch.jl` using the Julia package manager:

```julia
] add SimilaritySearch
```

All examples in this tutorial require only `SimilaritySearch` and `Distances.jl` (which provides the generic `evaluate` interface for distance objects):

```julia
using SimilaritySearch, Distances
```

---

## Quickstart: A Five-Minute Example

In similarity search, an index organizes a dataset $X \subseteq \mathcal{U}$ under a distance function $d: \mathcal{U} \times \mathcal{U} \to \mathbb{R}_{\ge 0}$ to answer nearest-neighbor queries efficiently.

For initial exploration, we use [`ExhaustiveSearch`](@ref). `ExhaustiveSearch` evaluates distances against every object sequentially, guaranteeing exact results with minimal setup overhead.

In this example, we index integers $i \in \{1, \dots, 1000\}$. Each integer is represented by the **set of its distinct prime factors**, and we measure similarity using the **Dice distance**:

```julia
using SimilaritySearch, Distances

"""
    factors(n::Integer) -> Vector{Int32}

Computes the sorted vector of distinct prime factors of `n`.
For example, `factors(60) == Int32[2, 3, 5]` because 60 = 2² · 3 · 5.
"""
function factors(n::Integer)
    f = Int32[]
    m = n
    d = Int32(2)
    while d * d <= m
        if m % d == 0
            push!(f, d)
            while m % d == 0
                m ÷= d
            end
        end
        d += 1
    end
    m > 1 && push!(f, m)
    isempty(f) ? Int32[1] : f  # Use 1 as a placeholder for n = 1
end

# 1. Prepare the dataset
n = 1000
X = VectorDatabase([factors(i) for i in 1:n])   # X[i] contains the prime factors of integer i
dist = Dist.Sets.Dice()

# 2. Instantiate the index and search context
idx = ExhaustiveSearch(dist, X)
ctx = GenericContext()

# 3. Execute a 5-nearest-neighbor query for the query integer 1000 (factors: {2, 5})
res = knnqueue(ctx, 5)                 # Pre-allocated result buffer for k = 5
search(idx, ctx, factors(1000), res)

# 4. Inspect the results
for p in IdDistView(res)
    println("ID: ", p.id, " => Factors: ", factors(p.id), " | Distance: ", p.dist)
end
```

### Interpretation of Results

The query retrieves the five numbers whose prime factor sets have the highest overlap with $\{2, 5\}$ (the factors of $1000 = 2^3 \cdot 5^3$).

Objects with distance `0.0` (such as $1000, 500 = 2^2 \cdot 5^3, 200 = 2^3 \cdot 5^2$) share the exact same set of distinct prime factors $\{2, 5\}$. Because the Dice distance operates purely on set membership, multiplicity is disregarded.

### Batch Queries

To process multiple query vectors simultaneously, use [`searchbatch`](@ref):

```julia
queries = VectorDatabase([factors(i) for i in (7, 60, 97, 360, 999)])
knns = searchbatch(idx, ctx, queries, 5)   # Returns a (5, 5) Matrix of IdDist elements
```

Here, `knns[:, j]` contains the 5 nearest neighbors for `queries[j]`.

---

In the next section, we explore [Databases: Data Abstraction with `AbstractDatabase`](databases.md) to understand why `SimilaritySearch.jl` defines a database interface rather than using raw arrays directly.
