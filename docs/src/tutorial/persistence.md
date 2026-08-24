```@meta
CurrentModule = SimilaritySearch
```

# Saving and loading indexes with JLD2

Building a [`SearchGraph`](@ref) is real work -- `index!` does a full pass of
approximate-neighbor search per inserted object. You don't want to redo that every time
you restart Julia. This package used to ship its own `saveindex`/`loadindex` functions,
but they were **removed** (they added a JLD2 dependency and a layer of indirection for
little benefit): every index in this package is a plain Julia `struct`, and
[JLD2.jl](https://github.com/JuliaIO/JLD2.jl) already serializes plain structs directly,
with no help needed. This page shows the DIY replacement.

```julia
] add JLD2 Accessors
```

(Neither is a dependency of your own project by default -- `JLD2` for saving/loading,
`Accessors` for the `@reset` macro used below to swap out a struct field. `Accessors` is
already a dependency *of* `SimilaritySearch`, but that doesn't make it `using`-able in
your own code without adding it yourself.)

## The basic pattern

```julia
using SimilaritySearch, JLD2

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
G = SearchGraph(Dist.SqL2(), X)
ctx = SearchGraphContext()
index!(G, ctx)

jldsave("graph.jld2"; G)
```

Loading it back gives you an index that behaves identically -- same graph, same database,
same search results:

```julia
G2 = load_object("graph.jld2")
res1 = knnqueue(ctx, 5); search(G, ctx, X[1], res1)
res2 = knnqueue(ctx, 5); search(G2, ctx, X[1], res2)
collect(IdView(res1)) == collect(IdView(res2))   # true
```

`jldsave(path; G)` saves the variable under the name `"G"`; `load_object` is the
matching one-object convenience reader. [`ExhaustiveSearch`](@ref)/
`SimilaritySearch.Exact.ParallelExhaustiveSearch` persist exactly the same way -- they're
plain structs too, nothing SearchGraph-specific is needed.

## Don't bother saving the context

Only save the *index* (`G`), not the [`SearchGraphContext`](@ref)/[`GenericContext`](@ref)
you built it with. The context is just configuration and scratch caches (`vstates`,
`beams`, the logger) -- there's nothing in it worth persisting, and recreating one fresh
after loading (`SearchGraphContext()`) gives you the exact same thing a saved-and-loaded
one would, with less to go wrong.

## Saving without the database

If your database is huge and already lives somewhere else (or you'd rather store it in a
more compact format than JLD2's default), swap it out for a placeholder before saving,
using the plain positional constructor:

```julia
placeholder = MatrixDatabase(zeros(Float32, 0, 0))
G_nodb = SearchGraph(G.dist, placeholder, G.adj, G.hints, G.algo, G.len)
jldsave("graph_nodb.jld2"; G_nodb)
```

and reattach the real database after loading:

```julia
loaded = load_object("graph_nodb.jld2")
G3 = SearchGraph(loaded.dist, X, loaded.adj, loaded.hints, loaded.algo, loaded.len)
```

This is the same trick the [databases page](databases.md) keeps coming back to: nothing
about `SearchGraph` cares what concrete database type it holds, so swapping one in or out
after the fact is just constructing a new `SearchGraph` value with a different `db`
field -- no special "detach the database" API needed.

## Note: external datasets you already have stored elsewhere

The pattern above generalizes to a common real case: your database isn't just "big", it's
already stored somewhere else entirely -- an HDF5/`.h5` file, a memory-mapped array, a
dataset another pipeline step manages -- and re-saving a copy of it inside the graph's
JLD2 file would be pure waste. Rather than rebuilding a whole new `SearchGraph` by hand
(as above), `Accessors.@reset` does the same field swap more directly, and is the
approach this package's own (now-removed) `saveindex` used internally:

```julia
using Accessors

G_nodb = @reset G.db = MatrixDatabase(zeros(Float32, 2, 2))   # tiny placeholder, not the real data
jldsave("graph_external.jld2"; G_nodb)
```

`@reset G.db = ...` returns a *new* `SearchGraph` with only the `db` field replaced --
`G` itself is untouched. The placeholder just needs to be *some* valid, cheap database of
the right element type; its contents are never used since the real dataset lives
elsewhere.

**On loading, you must `@reset` the real dataset back in yourself** -- nothing in the
saved file can do this for you, since the real data was never saved there in the first
place:

```julia
loaded = load_object("graph_external.jld2")
X_real = MatrixDatabase(prime_gap_windows(200_000, 5))   # e.g. re-derived, or read back from your .h5 file
loaded = @reset loaded.db = X_real
```

If your real dataset is itself stored in an HDF5 file, this is exactly where you'd read
it -- e.g. `X_real = MatrixDatabase(your_hdf5_loader("data.h5"))` -- instead of
recomputing it. Either way, the loaded graph is unusable for search until this step
happens; its adjacency structure refers to positions `1:length(loaded)`, which must line
up with whatever database you reattach.

## One caveat: struct changes across versions

JLD2 stores enough type information to reconstruct a struct, but if this package's
internal struct definitions change between the version you saved with and the version
you load with (a field added/removed/retyped), loading can fail or need migration code.
This mostly matters for long-lived saved indexes across major version upgrades of
`SimilaritySearch` itself -- not a concern for the everyday "save now, load in the next
session" use case above.

Next: [reporting, observing, and capturing neighbors as they're built](logging.md).
