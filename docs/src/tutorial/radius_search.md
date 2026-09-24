```@meta
CurrentModule = SimilaritySearch
```

# Radius Queries: Range-Bounded Search

In similarity search, queries are broadly categorized into two formulations:
1. **$k$-Nearest Neighbor ($k$-NN) Queries**: Find the $k$ closest elements to a query point $q$, where the search radius expands dynamically until $k$ items are found.
2. **Radius (Range) Queries**: Find all elements within a fixed distance threshold $r$ of a query point $q$:

$$B_d(q, r) = \{ x \in X \mid d(q, x) \le r \}$$

The result cardinality $|B_d(q, r)|$ is variable and depends on local point density.

`SimilaritySearch.jl` implements radius queries through specialized result queues: [`RadiusSorted`](@ref) and [`RadiusHeap`](@ref).

---

## Dataset Setup

We reuse the continuous prime-gap window dataset from the previous section:

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
    gaps = Float32.(log2.(diff(P)))
    m = length(gaps) - w
    M = Matrix{Float32}(undef, w, m)
    for i in 1:m
        M[:, i] .= view(gaps, i:i+w-1)
    end
    M
end

X = MatrixDatabase(prime_gap_windows(200_000, 5))
dist = Dist.SqL2()
```

---

## Radius Queries with `ExhaustiveSearch`

[`RadiusSorted`](@ref) and [`RadiusHeap`](@ref) are fully compatible with the generic `search` interface:

```julia
E = ExhaustiveSearch(dist, X)
ectx = GenericContext()

# Retrieve all items within squared Euclidean distance r = 0.05
res = RadiusSorted(0.05f0)
search(E, ectx, X[1], res)

println("Found ", length(res), " elements within radius 0.05:")
for p in IdDistView(res)
    println("ID: ", p.id, " | Distance: ", p.dist)
end
```

---

## Radius Queries with `SearchGraph`

The syntax is the same, but what happens underneath is not, and the difference matters:

```julia
G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)

res = RadiusSorted(0.05f0)
search(G, ctx, X[1], res)         # kmin defaults to 8
println("Found ", length(res), " elements within radius 0.05 using SearchGraph")
```

A graph search cannot be driven by a radius container on its own. Such a container rejects
every candidate outside the ball, so it is still **empty** when the beam needs somewhere to
start, and its covering radius is the constant `r` rather than a threshold that tightens as
the search improves -- a beam that starts outside the ball would have no admissible child and
would stop on its first expansion. (Both failure modes were real: until v1.5 this call
segfaulted whenever no entry point fell inside the ball, which is the normal case for a small
radius.)

The search therefore navigates with an internal container that keeps the ball **plus a reserve
of at least `kmin` nearest items, even when those fall outside it**, and copies only the
in-ball part into your `RadiusSorted`/`RadiusHeap`. The reserve is what restores both a
starting point and a shrinking threshold; it never reaches the result.

```julia
# a bigger reserve navigates better and costs more; the floor of a radius query is roughly
# what a k-NN query with k = kmin costs
res = RadiusSorted(0.05f0)
search(G, ctx, X[1], res; kmin=32)
```

!!! warning "The graph answer is approximate"
    Over a `SearchGraph` the result is a **subset** of the true ball, and its completeness is
    governed by the same `BeamSearch` parameters as a `k`-NN search. `ExhaustiveSearch` stays
    exact. This matters for algorithms that read a cardinality rather than a ranking --
    `dbscan`, for one, decides whether a point is a core point by counting its
    $\epsilon$-neighbors, and an incomplete ball can silently demote a core point to noise.

---

## Tuning for a radius workload

An index tuned for `k`-NN is not tuned for balls, and until v1.5 there was no way to tune for
them at all. [`optimize_index!`](@ref) takes a `radius` keyword: the gold standard becomes each
query's true ball -- of whatever size, empty included -- and candidate configurations are
scored against it.

```julia
optimize_index!(G, ctx, MaxMatchError(; maxerror=0.01f0); radius=0.05f0, kmin=8)
```

[`MaxMatchError`](@ref) is the only goal that applies. It compares distances rank by rank and
charges a fixed penalty for each ball member the search failed to reach, which is exactly ball
incompleteness; the recall-based goals go through `macrorecall`, which divides by the size of
the gold set, and a small radius routinely leaves queries whose true ball is empty. Passing
`MinRecall` together with `radius` raises an `ArgumentError` rather than dividing by zero.

---

## Batch Radius Search

Because each query may return a different number of results, batch radius queries cannot use a fixed-dimension rectangular matrix. Instead, batch execution is performed using the vector overload of [`searchbatch!`](@ref), which accepts a vector of independent result containers:

```julia
Q = X[1:5]
knns = [RadiusSorted(0.05f0) for _ in 1:length(Q)]   # Pre-allocate one queue per query

searchbatch!(G, ctx, Q, knns)  # Or searchbatch!(E, ectx, Q, knns) for ExhaustiveSearch

for (i, res) in enumerate(knns)
    println("Query ", i, ": ", length(res), " elements found within radius")
end
```

---

## Comparison: `RadiusSorted` vs. `RadiusHeap`

Both data structures filter elements based on the condition $d(q, x) \le r$, but differ in their internal storage strategy:

| Container | Insertion Complexity | Read Complexity | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| [`RadiusSorted`](@ref) | $O(\log n)$ (Binary search insertion) | $O(1)$ (Already sorted) | Queries with small result sets where immediate sorted ordering is desired. |
| [`RadiusHeap`](@ref) | $O(1)$ amortized (Append) | $O(m \log m)$ (Lazy sort upon inspection) | High-density queries expected to accumulate many matches within the radius. |

---

In the next section, [Dataset Operations: Selection, All-kNN, and Closest Pairs](operations.md), we explore global dataset algorithms including selection methods, all-pairs $k$-NN, and near-duplicate removal.
