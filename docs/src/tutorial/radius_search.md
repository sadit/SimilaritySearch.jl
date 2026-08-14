```@meta
CurrentModule = SimilaritySearch
```

# Radius queries: `RadiusSorted`, `RadiusHeap`

Every example so far has asked for a *fixed* number of neighbors (`k`). Sometimes what you
want instead is "every point within distance `r`, however many that turns out to be" -- a
**radius query**. [`RadiusSorted`](@ref) and [`RadiusHeap`](@ref) are growable result
containers for exactly that: unlike [`KnnSorted`](@ref)/[`KnnHeap`](@ref), they have no `k` at
all -- they accept an `(id, dist)` pair iff `dist <= r`, so the number of neighbors returned is
however many the data happens to contain within that radius.

This page reuses the prime-gap-windows dataset from [the previous page](searchgraph.md) --
still a genuinely continuous space, which radius queries need just as much as `SearchGraph`
does (see [the distances page](distances.md) for why):

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

## With `ExhaustiveSearch`

`RadiusSorted`/`RadiusHeap` work with any index whose `search` method is written generically
over result containers -- including [`ExhaustiveSearch`](@ref), no special-casing needed:

```julia
E = ExhaustiveSearch(dist, X)
ectx = GenericContext()

res = RadiusSorted(0.05f0)          # radius chosen arbitrarily for this example
search(E, ectx, X[1], res)
println(length(res), " neighbors within radius 0.05")
for p in viewitems(res)
    println(p.id, " ", p.dist)
end
```

## With `SearchGraph`

Same container, same `search` call, now walking the approximate proximity graph instead of
comparing against every point:

```julia
G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)

res = RadiusSorted(0.05f0)
search(G, ctx, X[1], res)
println(length(res), " neighbors within radius 0.05")
```

Since `SearchGraph` is approximate, this may miss a few of the points `ExhaustiveSearch` would
have found, or take a few extra exploration steps to reach far borderline ones -- the same
recall/approximation trade-off ordinary k-NN search has, just applied to "how many points are
within `r`" instead of "what are the k closest".

## Batch radius queries: the vector form of `searchbatch!`

Radius containers don't fit the fixed-`k` matrix layout `searchbatch!`/`searchbatch` normally
use (each query can return a different number of neighbors), so batch radius search goes
through the form of [`searchbatch!`](@ref) that takes a `Vector` of already-built containers,
one per query, instead:

```julia
Q = X[1:5]
knns = [RadiusSorted(0.05f0) for _ in 1:length(Q)]   # one independent container per query
searchbatch!(G, ctx, Q, knns)          # or searchbatch!(E, ectx, Q, knns) for ExhaustiveSearch

for (i, res) in enumerate(knns)
    println("query ", i, ": ", length(res), " neighbors within radius")
end
```

## `RadiusSorted` vs `RadiusHeap`

Both accept the same `(id, dist)` pairs under the same radius rule; they differ only in how
they keep that data:

- [`RadiusSorted`](@ref) keeps its items sorted by distance after every single push (bounded
  binary-search insertion), so [`viewitems`](@ref) is always ready with no extra work.
- [`RadiusHeap`](@ref) just appends on every push (`O(1)`) and only sorts lazily -- once, the
  first time you read it back (via [`viewitems`](@ref), [`nearest`](@ref), etc.) -- trading
  that one deferred sort for a cheaper build-up.

Reach for `RadiusHeap` when you expect a query to accumulate many matches and don't need
sorted order until you're done searching; `RadiusSorted` otherwise.

!!! note "Scope"
    Radius containers are only meant to be driven through `search` and the vector form of
    `searchbatch!` shown above. They have no fixed `k`, so they don't fit
    `GenericContext`/`SearchGraphContext`'s automatic `knnqueue(ctx, k)` construction, and
    they aren't supported by `ParallelExhaustiveSearch`'s parallel batch path or by
    `optimize_index!`'s recall calibration.

Next: [other things you can do with an index besides `search`](operations.md) -- `fft`,
`allknn`, `closestpair`, `neardup`.
