```@meta
CurrentModule = SimilaritySearch
```

# Beyond search: `fft`, `allknn`, `closestpair`, `neardup`

An index isn't only good for "give me the k nearest neighbors of this one query." This
page tours the other whole-dataset operations built on top of the same
[`AbstractDatabase`](@ref)/distance interface, again mostly with [`ExhaustiveSearch`](@ref)
since the data here is small.

## `fft`: picking a diverse, well-separated subset

[`fft`](@ref) (Farthest First Traversal) greedily selects `k` items that are as spread
out from each other as possible -- useful for choosing cluster centers, a representative
sample, or diverse candidates for anything downstream. A clean way to see it working: 24
points evenly spaced around a circle, and ask for 6 well-separated ones:

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
R.centers   # 6 identifiers into db -- roughly every 4th point around the circle
R.dmax      # the separation radius achieved (the smallest distance among the 6 centers)
```

`R.nn[i]` gives, for every point (not just the selected centers), which selected center
it ended up closest to -- so `fft` doubles as a quick way to partition a dataset around
`k` spread-out seeds.

## `allknn`: every object's own nearest neighbors, all at once

[`allknn`](@ref) computes the k nearest neighbors of *every* object in a database against
that same database (as opposed to `searchbatch`, which solves an externally given set of
queries) -- one call instead of a loop over every element as its own query. This is the
one operation on this page worth seeing with [`SearchGraph`](@ref) too, since comparing
against an exact `allknn` is a natural way to check an approximate index's quality:

```julia
X = MatrixDatabase(rand(Float32, 4, 2000))
dist = Dist.L2()

E = ExhaustiveSearch(dist, X)
ectx = GenericContext()
gold = allknn(E, ectx, 8)      # exact, O(n²) work -- fine at this scale

G = SearchGraph(dist, X)
ctx = SearchGraphContext()
index!(G, ctx)
optimize_index!(G, ctx, MinRecall(0.9))
approx = allknn(G, ctx, 8)      # approximate, much cheaper on large datasets

macrorecall(gold, approx)       # how close approx's neighbor sets are to gold's
```

Both `gold` and `approx` are `(8, 2000)` matrices of `IdDist` -- column `i` holds object
`i`'s own 8 nearest neighbors (note: an object is its own nearest neighbor at distance
`0`, and `allknn` keeps that self-reference rather than filtering it out).

## `closestpair`: the single closest pair in the whole dataset

[`closestpair`](@ref) finds the two objects with the smallest distance between them,
without checking every pair explicitly:

```julia
i, j, d = closestpair(E, ectx)   # (id, id, distance) of the closest pair in X
```

Passing a `SearchGraph` instead of an `ExhaustiveSearch` uses the graph structure to
avoid most pairwise comparisons -- much faster on large, continuous datasets, with the
same navigability caveat as regular search.

## `neardup`: collapsing near-duplicates

[`neardup`](@ref) walks through a dataset and, for every object, checks whether it's
within `ϵ` of something already kept -- if so, it's marked as a duplicate of that
earlier object instead of being kept itself. The simplest way to call it is the
two-argument form, which manages its own (empty, exact) index internally:

```julia
D = neardup(dist, X, 0.1)   # ϵ = 0.1
length(D.centers)            # how many distinct (non-duplicate) objects survived
D.map                        # D.centers, indexed 1:length(D.centers)
D.nn                         # for every object in X, the surviving object that "covers" it
```

If you already have an index you want `neardup` to fill (e.g. a `SearchGraph`, to use
approximate near-duplicate detection on a large dataset), pass it explicitly -- but it
must start **empty**:

```julia
empty_idx = ExhaustiveSearch(dist, VectorDatabase(Vector{Float32}[]))
D = neardup(empty_idx, ectx, X, 0.1)
```

## One more: `hsp_queries`

[`hsp_queries`](@ref) re-filters an already-computed k-NN matrix using the Half-Space
Proximal criterion -- a way to prune a neighborhood down to a smaller, more diverse set
of "true" neighbors (removing ones that are essentially redundant with a closer
neighbor). It's what `SearchGraph` uses internally to decide graph edges during
construction, but it's also directly usable on any k-NN matrix you already have lying
around:

```julia
knns = allknn(E, ectx, 16)                       # a generous k
hsp_matrix, hsp_knns = hsp_queries(dist, X, X, knns)
length.(hsp_knns)                                 # typically well under 16 per object
```

Next: [the parallelism model](parallelism.md) -- what `-t` actually buys you, and
concrete anti-patterns to avoid.
