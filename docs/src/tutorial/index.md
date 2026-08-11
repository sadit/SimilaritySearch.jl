```@meta
CurrentModule = SimilaritySearch
```

# Tutorial

!!! note "Authorship"
    Written by Eric S. Tellez with Claude (Anthropic). The prose was AI-drafted and
    human-reviewed, but not yet line-edited end to end -- as is typical of current AI
    models, some sentences here likely overstate things more confidently than warranted
    (a claim presented as a firm rule where reality is more nuanced, a benefit stated more
    emphatically than the evidence shown for it). Read matter-of-fact-sounding claims with
    a bit of caution and check them against the code/`?docstring` when it matters; this
    will get a proper editing pass over time.

This tutorial builds up `SimilaritySearch.jl` from first principles, using small,
self-contained synthetic datasets instead of large embedding collections (no downloads,
nothing random-looking-but-meaningless -- every example dataset here has a concrete
story you can check by hand). The running example across most pages is **numbers seen
through different lenses**: as sets of prime factors, as sequences of prime factors, as
divisibility bit patterns, and as prime-gap vectors. The same handful of integers keep
reappearing so you can compare how differently each distance function treats them.

Pages, in the order we recommend reading them:

1. **This page** -- installation and a five-minute quickstart.
2. [Databases: why not just a `Matrix`?](databases.md) -- the `AbstractDatabase`
   abstraction and why the library is built around it instead of raw arrays.
3. [A gallery of distances](distances.md) -- worked synthetic examples for vector,
   angular, set, sequence, and bit distances, and **why `SearchGraph` should not be used
   with discrete/combinatorial distances**.
4. [SearchGraph, in depth](searchgraph.md) -- the approximate index: how it works, when
   to reach for it, and (again, because it matters) when *not* to.
5. [Beyond search: fft, allknn, closestpair, neardup](operations.md) -- other things you
   can do with an index besides answering k-NN queries.
6. [Parallelism: what to expect, what not to do](parallelism.md) -- the `@BATCHES`-based
   threading model, its context objects, and concrete anti-patterns to avoid.
7. [Saving and loading indexes with JLD2](persistence.md) -- indexes are plain structs;
   here's the DIY save/load pattern now that this package doesn't ship its own.
8. [Logs, and capturing neighbors as they're built](logging.md) -- the logging system,
   custom loggers, and how (and how not) to use them to capture a `SearchGraph`'s
   neighbor lists incrementally during construction.
9. [Inverted files and posting list intersections](invertedfiles.md) -- the `InvertedFile`
   index for sparse vector, MIPS, and set search, and posting list intersection algorithms
   (`Intersections`).
10. [Quantization and Bit Sketches](quantization_and_bitsketches.md) -- techniques for compressing vectors into 
   smaller memory footprints and accelerating search using `ScalarQuant` and `bitsketch` with `ExhaustiveSearch`.

## Installation

```julia
] add SimilaritySearch
```

Everything in this tutorial only needs `SimilaritySearch` itself plus `Distances` (for
the `evaluate` function used to call distance objects directly) -- no extra packages, no
downloaded datasets.

```julia
using SimilaritySearch, Distances
```

## Five-minute quickstart

Throughout the tutorial we lean on [`ExhaustiveSearch`](@ref) for almost every worked
example: it is exact (no recall/approximation questions to worry about while you're
learning the API) and, on the small datasets used here, essentially instant.
[`SearchGraph`](@ref) gets its own dedicated page once the underlying concepts (databases,
distances) are in place.

Let's index a tiny, fully synthetic "space of numbers": we represent each integer by the
*set* of its distinct prime factors, and compare integers by how much those sets overlap
(a **Dice** distance -- two numbers are "close" if they share many small prime factors).

```julia
using SimilaritySearch, Distances

"""
Distinct prime factors of `n`, sorted -- e.g. `factors(60) == Int32[2, 3, 5]`
(60 = 2²·3·5). `Dist.Sets` distances expect their inputs as sorted vectors like this one.
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
    isempty(f) ? Int32[1] : f  # 1 has no prime factors; give it its own placeholder "factor"
end

n = 1000
X = VectorDatabase([factors(i) for i in 1:n])   # X[i] holds the prime factors of i
dist = Dist.Sets.Dice()

idx = ExhaustiveSearch(dist, X)
ctx = GenericContext()

res = knnqueue(ctx, 5)                 # a reusable result buffer for k=5
search(idx, ctx, factors(1000), res)   # which numbers "look like" 1000 = 2³·5³?

for p in viewitems(res)  # (id, dist) pairs, in storage order
    println(p.id, " => ", factors(p.id), "  dist=", p.dist)
end
```

Running this prints five numbers whose *set* of prime factors overlaps heavily with
`{2, 5}` (1000's factors) -- e.g. other numbers of the form `2^a · 5^b`. Note that a
result's `dist == 0.0` doesn't mean "the same number", it means "the same set of
distinct prime factors" -- e.g. 1000 and 500 both factor as `{2, 5}`, so they tie at
distance 0 despite being different integers. That's expected: Dice/Jaccard-style set
distances only see *which* primes divide a number, not how many times or what the
number itself is.

To search many queries at once instead of one at a time, use [`searchbatch`](@ref):

```julia
queries = VectorDatabase([factors(i) for i in (7, 60, 97, 360, 999)])
knns = searchbatch(idx, ctx, queries, 5)   # a (5, 5) matrix of IdDist
```

`knns[:, j]` holds the 5 nearest neighbors of `queries[j]`, sorted or not depending on
the `sorted` keyword (see [`searchbatch`](@ref)).

From here: [why does the library wrap `X` in a `VectorDatabase` instead of just using the
`Vector` directly?](databases.md) -- next page.
