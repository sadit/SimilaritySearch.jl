```@meta
CurrentModule = SimilaritySearch
```

# Databases: why not just a `Matrix`?

Every index in this package (`SearchGraph`, `ExhaustiveSearch`, ...) is generic over an
[`AbstractDatabase`](@ref) -- never over a `Matrix` or `Vector` directly. This page
explains why that indirection exists, using the previous page's prime-factor sets as a
running example (where a plain `Matrix` genuinely cannot work at all) plus a couple of
other concrete cases.

## The problem a `Matrix` can't solve

A `Matrix{Float32}` is a fine way to store `n` fixed-size numeric vectors: column `i` is
object `i`. But nothing about nearest-neighbor search actually *requires* "fixed-size
numeric vector" -- the previous page's objects were `Vector{Int32}` of *varying length*
(1000 has three prime factors, 997 -- prime -- has one). There is no `Matrix` that can
store that. You need *some* collection of arbitrary Julia objects, indexed by position,
that supports `length`, `getindex`, and (for growable indexes) `push!`-like growth.
That collection is exactly what [`VectorDatabase`](@ref) wraps:

```julia
julia> db = VectorDatabase([Int32[2,3,5], Int32[7], Int32[2,2,3]]);

julia> length(db), db[1], db[3]
(3, Int32[2, 3, 5], Int32[2, 2, 3])
```

`VectorDatabase` can hold *any* Julia type this way -- strings, sets, custom structs --
not just numeric vectors:

```julia
julia> words = VectorDatabase(["kitten", "sitting", "mitten", "bitten"]);

julia> words[2]
"sitting"
```

## Zero-copy wrapping when you *do* have a matrix

When your data genuinely is a dense numeric matrix, [`MatrixDatabase`](@ref) wraps it
**without copying**: `db.matrix === X` after `db = MatrixDatabase(X)`. Crucially, it
wraps *any* `AbstractMatrix`, not just `Matrix` -- which means the exact same index code
works unmodified over storage backends you'd never want to hand-write distance code for
yourself:

```julia
julia> using SparseArrays

julia> X = sprand(Float32, 200, 500, 0.05);  # a 200-dim, 500-object *sparse* dataset

julia> db = MatrixDatabase(X);

julia> db[1] isa AbstractVector  # a sparse column view -- SearchGraph/ExhaustiveSearch don't care
true
```

Nothing in `SearchGraph`'s or `ExhaustiveSearch`'s code path knows or cares that `X` is
sparse -- `getindex`/`length`/iteration all just work because `SparseMatrixCSC <:
AbstractMatrix`. If distances were instead hard-wired to expect `Matrix{Float32}`
columns, every alternative storage layout (sparse, a memory-mapped array, a custom
fixed-point encoding...) would need its own copy of every algorithm. The database
abstraction is what lets `SearchGraph`/`ExhaustiveSearch`/`fft`/`allknn`/... be written
*once*, against the `AbstractDatabase` interface, and reused for all of them.

## The rest of the family

- [`BlockMatrixDatabase`](@ref) -- like `MatrixDatabase`, but *growable*: it allocates
  dense blocks of columns internally, so you can `push_item!`/`append_items!` into it
  without the reallocate-and-copy-everything cost a single growing `Matrix` would incur.
  Use this over `VectorDatabase` when your objects genuinely are fixed-size numeric
  vectors and you still need incremental growth.
- [`SubDatabase`](@ref) -- a zero-copy *view* over a subset of another database (what
  `db[indices]`/`view(db, indices)`/`rand(db, k)` return). No copying happens; it just
  remaps indices into the original database.

```julia
julia> sample = rand(words, 2);  # a SubDatabase: 2 random elements of `words`, no copy

julia> sample isa SubDatabase
true
```

## Takeaway

Pick the database wrapper based on what your objects *are* and whether you need growth,
not based on habit:

| Your data is...                                   | Use                    |
|-----------------------------------------------------|------------------------|
| Fixed-size numeric vectors, static                   | [`MatrixDatabase`](@ref) (wraps `Matrix`, or any `AbstractMatrix`, including sparse) |
| Fixed-size numeric vectors, needs to grow             | [`BlockMatrixDatabase`](@ref) |
| Variable-length or non-numeric objects (sets, strings, sequences), static or growable | [`VectorDatabase`](@ref) |
| A subset/sample of an existing database               | [`SubDatabase`](@ref) (returned automatically by indexing/`rand`) |

Next: [a tour of the distance functions](distances.md) these databases get combined
with, including the sets, sequences, and bit-pattern examples used throughout the rest
of the tutorial.
