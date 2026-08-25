```@meta
CurrentModule = SimilaritySearch
```

# Dataset Operations: Selection, All-kNN, and Closest Pairs

In addition to individual query search, `SimilaritySearch.jl` provides global metric operations over datasets. These algorithms operate generically on any [`AbstractDatabase`](@ref) and distance metric.

---

## Center Selection: Farthest First Traversal (`fft`)

[`fft`](@ref) (Farthest First Traversal) selects a subset of $k$ representative centers $C = \{c_1, \dots, c_k\} \subseteq X$ such that the selected points are mutually well-separated in the metric space. This is widely used for prototype selection, hierarchical clustering, and spatial partitioning.

Consider 24 points regularly spaced on the unit circle $S^1 \subset \mathbb{R}^2$, from which we extract $k = 6$ representative centers:

```julia
using SimilaritySearch, Distances

n = 24
X = Matrix{Float32}(undef, 2, n)
for i in 1:n
    θ = 2π * (i - 1) / n
    X[1, i] = cos(θ)
    X[2, i] = sin(θ)
end
db = MatrixDatabase(X)

R = fft(Dist.L2(), db, 6; verbose=false)
```

### Properties of `CenterSelection`

The returned [`CenterSelection`](@ref) struct `R` contains:
- `R.centers`: Vector of object identifiers in `db` selected as centers.
- `R.assign[i]`: Position in `R.centers` corresponding to the closest center for object $i$.
- `R.assigndist[i]`: Distance from object $i$ to its assigned center $R.centers[R.assign[i]]$.
- `R.separation`: The minimum pairwise distance between any two chosen centers:

$$\text{separation} = \min_{i \ne j} d(c_i, c_j)$$

- `R.covering`: The covering radius (maximum distance from any dataset element to its nearest center):

$$\text{covering} = \max_{x \in X} \min_{c \in C} d(x, c)$$

Farthest First Traversal guarantees that $\text{covering} \le \text{separation}$.

### Selection Algorithms: `dnet`, `randsel`, and `multirandsel`

The selection module provides alternative algorithms returning the same `CenterSelection` structure:

- [`randsel`](@ref): Samples $k$ centers uniformly at random without separation guarantees.
- [`dnet`](@ref): Partitions the dataset into clusters of size $\approx |X| / k$ and retains one representative per cluster. It is computationally efficient on large datasets. Note that `assign` records the absorbing cluster center rather than the globally nearest center.
- [`multirandsel`](@ref): A randomized heuristic that samples candidate pools at each step, choosing the candidate maximizing the sum of distances to already selected centers:

```julia
R = multirandsel(Dist.L2(), db, 6)
```

---

## All-Pairs $k$-Nearest Neighbors: `allknn`

[`allknn`](@ref) computes the $k$ nearest neighbors for every element $x_i \in X$ against the dataset $X$.

```julia
X = MatrixDatabase(rand(Float32, 4, 2000))
dist = Dist.L2()

# Exact all-kNN baseline (O(n²))
E = ExhaustiveSearch(dist, X)
ectx = GenericContext()
gold_ids, gold_dists = allknn(E, ectx, 8)

# Approximate all-kNN via SearchGraph (O(n log n))
G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)
optimize_index!(G, ctx, MinRecall(0.9))
approx_ids, approx_dists = allknn(G, ctx, 8)

# Evaluate approximate all-kNN recall
macrorecall(gold_ids, approx_ids)
```

The output matrices `gold_ids` and `gold_dists` have dimensions $(k, |X|)$, where column $i$ contains the $k$ nearest neighbors of object $i$.

---

## Closest Pair: `closestpair`

[`closestpair`](@ref) finds the pair of distinct objects $(u, v) \in X \times X$ ($u \ne v$) with the minimum pairwise distance:

```julia
i, j, d = closestpair(E, ectx)  # Returns (id_u, id_v, distance)
```

When evaluated with `SearchGraph`, `closestpair` leverages graph connectivity to locate close elements without evaluating all $O(n^2)$ pairs.

---

## Near-Duplicate Elimination: `neardup`

[`neardup`](@ref) computes an $\epsilon$-net over a dataset, clustering objects within distance $\epsilon$ of an earlier representative into a single duplicate group:

```julia
D = neardup(dist, X, 0.1)     # Distance threshold ϵ = 0.1

length(D.centers)             # Number of retained distinct representatives
D.centers[D.assign[7]]        # Center index covering object 7
D.assigndist[7]               # Distance to covering center (always ≤ ϵ)
```

To automatically select $\epsilon$ from empirical data, sample the pairwise distance distribution using [`distsample`](@ref):

```julia
using Statistics
ϵ = quantile(distsample(dist, X; samplesize=2^10), 0.01)
D = neardup(dist, X, ϵ)
```

---

## Neighborhood Pruning: Half-Space Proximal (`hsp_queries`)

[`hsp_queries`](@ref) refines an existing $k$-NN candidate set by applying the Half-Space Proximal (HSP) criterion. A candidate $v$ is pruned if there exists another candidate $u$ closer to the query such that $v$ falls in the half-space dominated by $u$:

```julia
ids, dists = allknn(E, ectx, 16)
hsp_ids, hsp_dists, hsp = hsp_queries(dist, X, X, ids, dists)
```

The HSP criterion reduces graph degree while preserving navigability, which is used internally during `SearchGraph` construction.

---

In the next section, [Bichromatic Operations and Metric Joins](bichromatic.md), we generalize closest pairs and similarity matching to pairs of distinct datasets.
