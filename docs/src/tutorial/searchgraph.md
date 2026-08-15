```@meta
CurrentModule = SimilaritySearch
```

# `SearchGraph`, in depth

!!! warning "Read [the previous page's last section](distances.md) first if you haven't"
    `SearchGraph` is an **approximate** index built around a **navigable** proximity
    graph. It is a poor fit for discrete/combinatorial distances (set, sequence, and bit
    distances with many exact ties) -- use [`ExhaustiveSearch`](@ref) for those instead,
    regardless of dataset size. Every example on this page uses a genuinely continuous
    distance (`SqL2`) for exactly that reason.

## The example: prime *gaps*, not prime factors

The previous pages' prime-factor examples are discrete and not a good match for
`SearchGraph` (as just discussed). This page uses a different, still fully synthetic and
prime-themed, but genuinely continuous space instead: **windows of consecutive prime
gaps**. Take the sequence of gaps between consecutive primes, log-transform it (gaps grow
roughly logarithmically, so this keeps the scale manageable), and slide a fixed-width
window over it -- each window becomes one `Float32` vector:

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
    gaps = Float32.(log2.(diff(P)))    # gap sizes, log2-scaled
    m = length(gaps) - w
    M = Matrix{Float32}(undef, w, m)
    for i in 1:m
        M[:, i] .= view(gaps, i:i+w-1)  # window i: gaps[i], gaps[i+1], ..., gaps[i+w-1]
    end
    M
end

M = prime_gap_windows(200_000, 5)   # (5, 17978) -- 17978 overlapping 5-gap windows
X = MatrixDatabase(M)
```

Nearby columns of `X` describe similar *local patterns* in how prime gaps grow -- a
genuinely continuous notion of similarity, well suited to `SqL2`/`L2`/`Cosine`.

## Building and querying

```julia
dist = Dist.SqL2()
G = SearchGraph(dist, X)     # positional (dist, db) -- not a keyword constructor
ctx = SearchGraphContext()
index!(G, ctx)                # builds the graph: inserts every element of X

res = knnqueue(ctx, 5)
search(G, ctx, X[1], res)
for p in IdDistView(res)
    println(p.id, " ", p.dist)
end
```

Compared to [`ExhaustiveSearch`](@ref), which always compares a query against every
object, `SearchGraph` looks at only a small fraction of the dataset per query by walking
the proximity graph -- much faster on large datasets, at the cost of not being
guaranteed to find the *exact* nearest neighbors.

Every query above asks for a *fixed* number of neighbors (`k`). If what you actually want
is "every point within distance `r`, however many that turns out to be" instead, see
[the next page](radius_search.md) on [`RadiusSorted`](@ref)/[`RadiusHeap`](@ref) --
radius-bounded result containers that work with both `ExhaustiveSearch` and `SearchGraph`.

## How much accuracy are you losing? `optimize_index!`

"Approximate" is a knob, not a fixed cost. [`optimize_index!`](@ref) tunes the graph's
search parameters (`BeamSearch`'s beam size, etc.) to try to hit a target recall:

```julia
optimize_index!(G, ctx, MinRecall(0.9))   # aim for ~90% recall against a small internal gold standard
```

You can measure the *actual* recall yourself against an exact index, which is always a
good idea before trusting an approximate index in production:

```julia
E = ExhaustiveSearch(dist, X)
ectx = GenericContext()

Q = X[1:50]                                  # 50 queries (reusing dataset points here, for simplicity)
gold  = searchbatch(E, ectx, Q, 5)           # exact
approx = searchbatch(G, ctx, Q, 5)           # approximate
macrorecall(gold, approx)                     # somewhere around 0.8-0.95 -- construction is randomized, so exact runs vary
```

!!! tip
    If you tune with [`optimize_index!`](@ref) using the *same* queries you evaluate
    recall with afterwards, you're measuring overfitting, not real-world recall. Prefer
    a held-out query set for `optimize_index!`'s `queries` keyword when you can.

## Incremental construction

Unlike [`MatrixDatabase`](@ref) (fixed-size), a [`BlockMatrixDatabase`](@ref)- or
[`VectorDatabase`](@ref)-backed `SearchGraph` can grow after `index!`, via
[`append_items!`](@ref) (see [the databases page](databases.md) for why the backing
database type matters here):

```julia
db = BlockMatrixDatabase(M)     # growable, unlike MatrixDatabase
G = SearchGraph(dist, db)
ctx = SearchGraphContext()
index!(G, ctx)                   # index the initial batch

more = MatrixDatabase(prime_gap_windows(210_000, 5)[:, end-500:end])
append_items!(G, ctx, more)       # grow the graph with new windows
length(G)                         # original count + 501
```

## `rebuild`: a second look at the whole dataset

`SearchGraph` is built incrementally: the `i`-th object is connected using only the
`1..i-1` objects seen so far, which can leave early insertions with a worse neighborhood
than they'd get if the whole dataset had been available from the start. [`rebuild`](@ref)
recomputes the graph letting every object see the *entire* final dataset:

```julia
G2 = rebuild(G, ctx)   # returns a new SearchGraph; G is left untouched
```

This costs roughly as much as building from scratch, so it's a "do this once you're
done growing the index and want the best possible quality" step, not something to call
after every insertion.

## `BeamSearch`, hints, and the local search algorithm

`SearchGraph`'s traversal strategy is stored in `G.algo[]` (default: [`BeamSearch`](@ref)),
and `optimize_index!` mutates it in place to hit your target recall -- you generally
don't need to construct one by hand. If you're curious what it's doing: `BeamSearch`
keeps a small set ("beam") of the most promising candidates seen so far and expands
their graph neighbors each step, same idea as beam search in other contexts (e.g.
sequence decoding), just over a proximity graph instead of a sequence of tokens.
`SearchGraphContext`'s `hints_callback` controls how entry points into the graph are
chosen for queries that don't provide their own -- the default (`RandomHints`) is a
reasonable choice for most datasets.

Next: [radius queries with `RadiusSorted`/`RadiusHeap`](radius_search.md).
