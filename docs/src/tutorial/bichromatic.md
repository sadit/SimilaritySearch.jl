```@meta
CurrentModule = SimilaritySearch
```

# Two datasets, one distance: bichromatic closest pairs and joins

The [previous page](operations.md)'s [`closestpair`](@ref) finds the two closest objects
*within* a single indexed dataset. This page covers its **bichromatic** generalization:
given two distinct datasets `A` and `B` under the same distance, which objects of `A` are
closest to which objects of `B`? "Bichromatic" is metric-search jargon for exactly this
setup -- color the points of `A` one color and `B` another, and ask cross-color questions
instead of within-color ones. `SimilaritySearch.jl` exposes three such questions, in the
`Bichromatic` submodule (re-exported at the top level, so no
`using SimilaritySearch.Bichromatic` is needed):

- [`bichromatic_closestpair`](@ref) -- the single globally closest `A`-`B` pair.
- [`bichromatic_kclosestpairs`](@ref) -- the `k` globally closest `A`-`B` pairs.
- [`bichromatic_metricjoin`](@ref) -- every `A`-`B` pair "close enough" to count as a
  match, when neither a fixed radius nor a match count per element is known ahead of time.

[`closestpair`](@ref) and [`closestpairs`](@ref) from the previous page aren't separate
algorithms -- they're exactly the bichromatic functions called with `B = database(idx)`,
i.e. the case where the two "colors" happen to be the same dataset (self-matches
excluded). We'll see that equivalence directly below.

## Setup: warehouses and customers

Four warehouses at the corners of a square, and six customers scattered near them --
small enough to check every distance by hand (every warehouse-to-warehouse edge is
length `10`; a couple of customers sit right next to a corner, one sits stranded in the
middle):

```julia
using SimilaritySearch, Distances

dist = Dist.L2()

A = MatrixDatabase(Float32[0 10 0 10; 0 0 10 10])          # 4 warehouses, corners of a 10x10 square
B = MatrixDatabase(Float32[1 9 1 9 5 0.5; 1 1 9 9 5 0.5])  # 6 customers
```

`A`'s columns are `(0,0)`, `(10,0)`, `(0,10)`, `(10,10)`. `B`'s are `(1,1)`, `(9,1)`,
`(1,9)`, `(9,9)` -- one just inside each corner -- plus `(5,5)` (equidistant from every
corner, `≈7.07` away from each) and `(0.5,0.5)` (even closer to the first corner than
`(1,1)` is).

## `bichromatic_closestpair`: the single closest match

Index the smaller/"server" side (`A`, the warehouses) and query it with the other side
(`B`, the customers) -- this drives the same `(idx, ctx, q, res)` machinery as regular
`search`, internally, once per element of `B`:

```julia
idxA = ExhaustiveSearch(dist, A)
ctx = GenericContext()

i, j, d = bichromatic_closestpair(idxA, ctx, B)
# (1, 6, 0.70710677f0)
```

Warehouse `1` (`(0,0)`) and customer `6` (`(0.5,0.5)`) are the globally closest pair, at
distance `√0.5 ≈ 0.707` -- closer than the seemingly-obvious `(1,1)`/warehouse-`1` pair
(`√2 ≈ 1.414`). `bichromatic_closestpair` always iterates over `B` querying into `idxA`
(not the reverse), so pass whichever side you'd rather have indexed as `A`.

## `bichromatic_kclosestpairs`: the k closest matches, globally

Ask for the 5 closest pairs instead of just 1:

```julia
pairs = bichromatic_kclosestpairs(idxA, ctx, B; k=5)
for p in pairs
    println(p)
end
# (1, 6, 0.70710677f0)
# (4, 4, 1.4142135f0)
# (3, 3, 1.4142135f0)
# (2, 2, 1.4142135f0)
# (1, 1, 1.4142135f0)
```

The single closest pair from before comes first, then the four "obvious" corner-customer
pairs -- all tied at exactly `√2`, since every corner has a customer sitting exactly
`(1,1)` away from it. Ties are broken by insertion order, not by warehouse/customer id, so
don't rely on their relative order here -- only on the distance and the tie itself (same
caveat as the Jaccard ties in [A gallery of distances](distances.md)/
[Inverted Files](invertedfiles.md)). Customer `5`, the stranded `(5,5)` point, doesn't show
up at all among the 5 closest -- at `≈7.07` from every corner, it's nobody's close match.

## `closestpair`/`closestpairs` are the same-dataset special case

[`closestpair`](@ref)`(idx, ctx)` is defined as exactly
`bichromatic_closestpair(idx, ctx, database(idx))`, with self-matches excluded:

```julia
closestpair(idxA, ctx)
# (2, 1, 10.0f0)
bichromatic_closestpair(idxA, ctx, database(idxA))
# (2, 1, 10.0f0)
```

Both calls solve "the closest pair *among the warehouses themselves*" -- any two adjacent
corners, `10` apart (there's a 4-way tie here too; only one winner is returned).
[`closestpairs`](@ref) is [`bichromatic_kclosestpairs`](@ref)'s analogous same-dataset case.

## The `samedata` gotcha: "same values" isn't "the same dataset"

Both functions decide whether to exclude self-matches via a `samedata` keyword, defaulting
to `database(idxA) === B` -- an *identity* check, not a value check. Two distinct
`MatrixDatabase`s holding equal coordinates do **not** count as the same dataset by
default:

```julia
B2 = MatrixDatabase(copy(A.matrix))   # distinct object, same coordinates as A
database(idxA) === B2   # false -- different object, even though the values match
bichromatic_closestpair(idxA, ctx, B2)
# (1, 1, 0.0f0)  -- warehouse 1 "matched" to its own coordinates, at distance 0
```

If `B2` is genuinely meant to be treated as the same dataset (self-matches excluded),
force it explicitly:

```julia
bichromatic_closestpair(idxA, ctx, B2; samedata=true)
# (2, 1, 10.0f0) -- same answer as closestpair(idxA, ctx) above
```

## Building the index for you: the `(dist, A, B)` convenience form

Every function above also has a form that takes raw datasets instead of a pre-built index,
mirroring [`neardup`](@ref)'s two-argument convenience wrapper:

```julia
i, j, d = bichromatic_closestpair(dist, A, B)
# (1, 6, 0.70710677f0) -- same answer; `A` gets indexed (ExhaustiveSearch) internally
```

Pass `recall` below `1.0` to use an approximate ([`SearchGraph`](@ref)) index instead,
tuned toward that recall via [`optimize_index!`](@ref)/[`MinRecall`](@ref) internally --
worth it once `A` is large enough that indexing it exactly stops being cheap:

```julia
i, j, d = bichromatic_closestpair(dist, A, B; recall=0.9)
# (1, 6, 0.70710677f0) -- same answer on a dataset this small/easy
```

## `bichromatic_metricjoin`: matching without a fixed radius or count

The two functions above always return a fixed number of pairs (`1` or `k`). Sometimes what
you actually want is a **join**: every pair close enough to count as a match, however many
that turns out to be -- with no natural "closeness" threshold known ahead of time, and,
critically, no reason to expect the *same* threshold to make sense everywhere in `A`.

Simulate that: still four warehouses at the same corners (scaled up to a `100x100`
square), but now a **dense** cluster of 40 customers around warehouse 1 and a **sparse**
cluster of 8 around warehouse 4 (plus a lighter scatter of 10 near each of the other two)
-- deliberately uneven density, the situation a single global radius handles badly:

```julia
using Random
Random.seed!(42)

dense   = 0.0f0   .+ 5.0f0  .* randn(Float32, 2, 40)    # tight cluster around (0,0)
sparse_ = 100.0f0 .+ 15.0f0 .* randn(Float32, 2, 8)     # loose cluster around (100,100)
near2   = [100.0f0, 0.0f0] .+ 5.0f0 .* randn(Float32, 2, 10)
near3   = [0.0f0, 100.0f0] .+ 5.0f0 .* randn(Float32, 2, 10)

Awide = MatrixDatabase(Float32[0 100 0 100; 0 0 100 100])
Bwide = MatrixDatabase(Float32.(hcat(dense, sparse_, near2, near3)))   # 68 customers total

idxAwide = ExhaustiveSearch(dist, Awide)
ctxwide = GenericContext()

pairs = bichromatic_metricjoin(idxAwide, ctxwide, Bwide; k=8)
length(pairs)   # 61

counts = zeros(Int, length(idxAwide))
for (a, b, d) in pairs
    counts[a] += 1
end
counts   # [36, 9, 9, 7] -- warehouse 1 (dense, 40 customers) ... warehouse 4 (sparse, 8 customers)
```

`k=8` is a deliberately generous, overestimated guess -- `bichromatic_metricjoin` doesn't
return `k` pairs per customer; it uses those `k` candidates to work out a *separate*
cutoff radius for every warehouse (from how customers vote for their nearest warehouse;
see [`bichromatic_metricjoin`](@ref)'s docstring for the exact mechanics), then filters
against that per-warehouse cutoff. That's why the match *rate* comes out similarly high
for the dense corner (`36/40 = 90%`) and the sparse one (`7/8 = 87.5%`), even though their
raw distances-to-nearest-warehouse sit on completely different scales -- a single global
radius tuned to suit one corner would systematically over- or under-match the other.

Next: [the parallelism model](parallelism.md) -- what `-t` actually buys you, and
concrete anti-patterns to avoid.
