```@meta
CurrentModule = SimilaritySearch
```

# `SearchGraph`: Approximate Proximity Graphs

[`SearchGraph`](@ref) is an approximate nearest neighbor search index based on a navigable proximity graph. It provides sub-linear query times on continuous metric spaces by traversing an adjacency network of data points.

!!! warning "Requirement for Continuous Metric Spaces"
    As detailed in [Distance Functions and Metric Spaces](distances.md), graph-based search requires a navigable continuous distance gradient. For discrete metrics (such as Jaccard, Hamming, or edit distances with high tie frequencies), use [`ExhaustiveSearch`](@ref) or [`InvertedFile`](@ref) instead.

---

## Synthetic Continuous Dataset: Prime Gap Windows

To demonstrate `SearchGraph` on a continuous space without external dependencies, we construct vectors from sliding windows of logarithmic prime gaps.

Let $p_1 < p_2 < \dots < p_n$ be consecutive prime numbers. The gap $g_i = p_{i+1} - p_i$ is transformed logarithmically as $y_i = \log_2(g_i)$. A sliding window of width $w = 5$ generates feature vectors $x_i = [y_i, y_{i+1}, \dots, y_{i+w-1}]^T \in \mathbb{R}^5$:

```julia
using SimilaritySearch, Distances

function primes_upto(n::Integer)
    sieve = trues(n)
    sieve[1] = false
    for p in 2:isqrt(n)
        sieve[p] && (sieve[p*p:p:n] .= false)
    end
    findall(sieve)
end

function prime_gap_windows(n::Integer, w::Integer)
    P = primes_upto(n)
    gaps = Float32.(log2.(diff(P)))    # Compute log2 of prime gaps
    m = length(gaps) - w
    M = Matrix{Float32}(undef, w, m)
    for i in 1:m
        M[:, i] .= view(gaps, i:i+w-1)  # Extract window of width w
    end
    M
end

# Generate 17,978 5-dimensional vectors from primes up to 200,000
M = prime_gap_windows(200_000, 5)
X = MatrixDatabase(M)
```

In this continuous space, vectors with low squared Euclidean distance (`Dist.SqL2`) correspond to similar local growth dynamics in prime number distributions.

---

## Index Construction and Querying

Instantiate a `SearchGraph` using the positional constructor `(dist, db)` and build the graph using [`index!`](@ref):

```julia
dist = Dist.SqL2()
G = SearchGraph(dist, X)     # Positional constructor: SearchGraph(dist, db)
ctx = SearchGraphContext()
index!(G, ctx)                # Constructs the proximity graph across all items in X

# Execute a 5-NN query
res = knnqueue(ctx, 5)
search(G, ctx, X[1], res)

for p in IdDistView(res)
    println("ID: ", p.id, " | Distance: ", p.dist)
end
```

Unlike [`ExhaustiveSearch`](@ref), which performs $O(n)$ distance evaluations per query, `SearchGraph` traverses a small subset of the graph, offering substantial speedups on large datasets at the expense of an approximation factor.

---

## Tuning Search Quality: `optimize_index!`

Approximate nearest neighbor indexes exhibit a trade-off between search throughput and search accuracy (recall).

To measure empirical search recall, compare the approximate results against an exact index baseline using [`macrorecall`](@ref):

```julia
# 1. Build exact baseline
E = ExhaustiveSearch(dist, X)
ectx = GenericContext()

# 2. Select query sample
Q = X[1:50]
gold   = searchbatch(E, ectx, Q, 5)   # Exact nearest neighbors
approx = searchbatch(G, ctx, Q, 5)    # Approximate nearest neighbors

# 3. Calculate macro-averaged recall
current_recall = macrorecall(gold, approx)
```

The function [`optimize_index!`](@ref) automatically calibrates internal search hyperparameters (such as beam search width) to satisfy a target recall constraint:

```julia
optimize_index!(G, ctx, MinRecall(0.9))   # Optimize parameters to achieve ≥ 90% recall
```

!!! tip "Evaluation Best Practice: Held-Out Queries"
    To avoid statistical overfitting during parameter optimization, provide a separate, held-out query set via the `queries` keyword of `optimize_index!` rather than evaluating on the training data.

[`MinRecall`](@ref) is not the only quality target: see [`MaxMatchError`: A Distance-Based Alternative to `MinRecall`](matcherror.md) for a tie-tolerant alternative that suits discretized/quantized spaces (e.g. bit sketches) better.

---

## Incremental Graph Growth

When backed by a growable container such as [`BlockMatrixDatabase`](@ref) or [`VectorDatabase`](@ref), a `SearchGraph` can incorporate new data points dynamically after initial construction using [`append_items!`](@ref):

```julia
# Create a growable graph index
db = BlockMatrixDatabase(M)
G = SearchGraph(dist, db)
ctx = SearchGraphContext()
index!(G, ctx)

# Append additional vectors
more = MatrixDatabase(prime_gap_windows(210_000, 5)[:, end-500:end])
append_items!(G, ctx, more)
length(G)  # Reflects the updated total object count
```

---

## Global Graph Rebuilding: `rebuild`

During incremental construction, element $i$ is connected only to the subset of preceding elements $\{1, \dots, i-1\}$. As a result, early elements may possess lower-quality connectivity than elements inserted later.

The [`rebuild`](@ref) function computes a global proximity graph by allowing all vertices to consider the complete dataset simultaneously:

```julia
G2 = rebuild(G, ctx)   # Returns a new, optimized SearchGraph; G remains unmodified
```

Rebuilding performs a complete reconstruction pass and is typically executed after completing large batch insertions.

---

## Traversal Mechanics: `BeamSearch` and Hints

The traversal algorithm governing graph navigation is stored in `G.algo` (defaulting to [`BeamSearch`](@ref)). During query execution:
1. **Entry Point Selection (Hints)**: The algorithm selects initial entry vertices determined by `ctx.hints_callback` (defaulting to [`RandomHints`](@ref)).
2. **Beam Exploration**: `BeamSearch` maintains a priority queue of size $b$ containing the most promising visited candidates. At each step, it expands the neighborhood of candidate vertices, updating the beam until no closer neighbor is found.

Hyperparameter tuning via [`optimize_index!`](@ref) adjusts the beam parameters in-place to achieve the requested accuracy.

---

In the next section, [Radius Queries: Range-Bounded Search](radius_search.md), we examine how to retrieve all neighbors within a distance threshold $r$ rather than a fixed count $k$.
