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

Radius queries execute with identical syntax on graph-based indexes:

```julia
G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)

res = RadiusSorted(0.05f0)
search(G, ctx, X[1], res)
println("Found ", length(res), " elements within radius 0.05 using SearchGraph")
```

On a `SearchGraph`, radius queries prune traversal when graph paths exceed the distance threshold $r$.

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
