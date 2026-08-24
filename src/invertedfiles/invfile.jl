# This file is part of InvertedFiles.jl

using LinearAlgebra, SparseArrays
export AbstractInvertedFile, InvertedFile

"""
    abstract type AbstractInvertedFile <: AbstractSearchIndex end

Abstract inverted file; the concrete data structure is [`InvertedFile`](@ref).
"""
abstract type AbstractInvertedFile <: AbstractSearchIndex end

"""
    length(idx::AbstractInvertedFile)

Number of indexed elements (i.e., objects with postings already built). This can be less than
`length(database(idx))` if `db` was grown (e.g. via `push_item!(database(idx), obj)` or
`append_items!(database(idx), items)` directly) without a following [`index!`](@ref) call to
catch up -- mirrors `SearchGraph`'s `length`/`len` contract.
"""
Base.length(idx::AbstractInvertedFile) = idx.len[]

"""
    struct InvertedFile{DistType<:PreMetric, AdjType<:AbstractAdjList, DbType<:AbstractDatabase} <: AbstractInvertedFile

A general-purpose inverted index: a sparse matrix-like representation mapping component
dimensions (or set elements/tokens) to identifiers (`AdjType`'s element type is `UInt32`, plain
token/set membership; other concrete adjacency element types, e.g. a compressed encoding, can be
added by extending [`getcontainer`](@ref), `internal_push!`, and [`sort_postinglist!`](@ref)). It
always keeps the original indexed object in `db`.

# Fields

- `dist`: distance function used at search time (e.g. `Dist.Sets.Jaccard()`, `Dist.NormCosine()`).
- `adj`: posting lists (non-zero id-elements, in rows).
- `sizes`: number of non-zero values in each element (non-zero values in columns); resized/populated
  only up to `len[]`.
- `db`: the original indexed objects, one per identifier; always populated by `push_item!`/`append_items!`,
  but may hold more objects than have actually been indexed -- see `len`.
- `len`: number of objects already indexed (postings built); may be less than `length(database(idx))`
  if `db` was grown directly without a following [`index!`](@ref) call to catch up.

For a handful of distances (the set metrics in `Dist.Sets`, see
[`InvertedFiles.has_exact_fastpath`](@ref)) the score computed while merging posting lists is already
exact, at O(1) cost. For any other distance (including `Dist.NormCosine`), every merge candidate is
instead scored by evaluating `dist` directly against the objects stored in `db`, so results for that
path are exact too — the number of such evaluations (hence cost) is controlled by the `t`-threshold
parameter of `search`; raise `t` above the default `1` to bound the number of real evaluations per query.
"""
struct InvertedFile{DistType<:PreMetric, AdjType<:AbstractAdjList, DbType<:AbstractDatabase} <: AbstractInvertedFile
    dist::DistType
    adj::AdjType
    sizes::Vector{UInt32}
    db::DbType
    len::Ref{Int64}
end

function Base.show(io::IO, invfile::InvertedFile; prefix="", indent="\t")
    println(io, prefix, "InvertedFile:")
    prefix = indent * prefix
    println(io, prefix, "dist: ", invfile.dist)
    println(io, prefix, "length: ", length(invfile))
    println(io, prefix, "adj: ", typeof(invfile.adj))
    println(io, prefix, "db: ", typeof(invfile.db))
end

distance(idx::InvertedFile) = idx.dist
database(idx::InvertedFile) = idx.db

"""
    InvertedFile(vocsize::Integer, dist::PreMetric=Dist.Sets.Jaccard(); db::AbstractDatabase=VectorDatabase(Any[]))

Creates an empty `InvertedFile` with plain token/set-membership posting lists (`AdjType`'s element type is
`UInt32`), for the given vocabulary size and distance function `dist` (typically one of the set metrics in
`Dist.Sets`, e.g. `Jaccard`, `Dice`, `Intersection`, `CosineSet`, `RogersTanimoto`; or any other `PreMetric`
— e.g. `Dist.NormCosine()` for sparse-vector/MIPS-style cosine search — via the generic direct-evaluate
fallback).

# Arguments
- `vocsize`: the vocabulary size of the index
- `dist`: the distance function to be used in searches

# Keyword arguments
- `db`: the database that will receive a copy of every indexed object (must support `push_item!`/`append_items!`
  for incremental construction, e.g. a `VectorDatabase`); defaults to an empty, untyped `VectorDatabase`.
  If `db` is passed already non-empty, call [`index!`](@ref) once before searching to build postings
  for its contents.
"""
function InvertedFile(vocsize::Integer, dist::PreMetric=Dist.Sets.Jaccard(); db::AbstractDatabase=VectorDatabase(Any[]))
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    InvertedFile(dist, resize!(AdjList(UInt32), vocsize), UInt32[], db, Ref(Int64(0)))
end

function getcontainer(idx::AbstractInvertedFile, ctx::InvertedFileContext)
    Q = getcontainer(idx.adj, ctx)
    empty!(Q)
    Q
end

getcontainer(adj::AbstractAdjList, ctx) = ctx.buffer[ctx.batchid]

function getcontainer(adj::StaticAdjList, ctx)
    Q = [PostingList(neighbors(adj, 1), zero(UInt32))]
    empty!(Q)
    sizehint!(Q, 32)
    Q
end

function getpositions(k::Integer, ctx::InvertedFileContext)
    P = ctx.positions[ctx.batchid]
    resize!(P, k)
    fill!(P, 1)
    P
end

"""
    identiterator(obj)

Iterator over the plain ids/keys in `obj`, for callers that only need to know *which* ids/keys
are present (e.g. `InvertedFile` building/re-sorting/searching its posting lists, which never
need a weight: the handful of distances with an exact fast path score from intersection size
and set sizes alone, and any other distance is evaluated directly against the full objects
kept in `db` -- see [`InvertedFile`](@ref)). Dense `Vector`s are not accepted directly --
convert to a `SparseVector` first (e.g. via `SparseArrays.sparse`) so the reduction to
non-zero components is explicit in the caller's code.
"""
identiterator(obj::SparseVector) = obj.nzind
identiterator(obj::SparseVecView) = obj.nzind
identiterator(obj::Set) = (convertident(u) for u in obj)
identiterator(obj::SortedIntSet) = (convertident(u) for u in obj)
identiterator(obj) = (convertident(u) for u in obj)

"""
    identiterator(dist::PreMetric, obj)

Distance-aware id-only iterator for `obj`. Defaults to the distance-agnostic
[`identiterator(obj)`](@ref) dispatch tree above; overload this for a specific `(DistType, ObjType)`
pair when the same native object type must generate different *candidate ids* depending on which
distance the enclosing index is built for (e.g. a shingle-based candidate encoding for a sequence
distance).
"""
identiterator(::PreMetric, obj) = identiterator(obj)

"""
    convertident(u)

Converts an element of an [`identiterator`](@ref) fallback into a plain key/id, discarding any paired
weight if present as a `Pair`.
"""
convertident(u::Pair) = u.first
convertident(u) = u

"""
    const DictInvertedFile{DistType, KeyType, DbType} = InvertedFile{DistType, AdjDict{KeyType, UInt32}, DbType}

A dictionary-backed inverted file mapping posting list keys of type `KeyType` (e.g., `String`,
`Vector{UInt8}`, `NTuple`, `Int`) to document identifiers (`UInt32`). Empty or non-existent
posting lists are never stored in memory or disk, enabling use over arbitrary or massive key spaces.

# Constructors
- `DictInvertedFile(::Type{KeyType}, dist::PreMetric=Dist.Sets.Jaccard(); db::AbstractDatabase=VectorDatabase(Any[]), hint_size::Integer=0)`
- `DictInvertedFile(dist::PreMetric=Dist.Sets.Jaccard(); KeyType::Type=Any, db::AbstractDatabase=VectorDatabase(Any[]), hint_size::Integer=0)`
"""
const DictInvertedFile{DistType, KeyType, DbType} = InvertedFile{DistType, AdjDict{KeyType, UInt32}, DbType}

function DictInvertedFile(::Type{KeyType}, dist::PreMetric=Dist.Sets.Jaccard(); db::AbstractDatabase=VectorDatabase(Any[]), hint_size::Integer=0) where KeyType
    InvertedFile(dist, AdjDict(KeyType, UInt32; n=hint_size), UInt32[], db, Ref(Int64(0)))
end

function DictInvertedFile(dist::PreMetric=Dist.Sets.Jaccard(); KeyType::Type=Any, db::AbstractDatabase=VectorDatabase(Any[]), hint_size::Integer=0)
    InvertedFile(dist, AdjDict(KeyType, UInt32; n=hint_size), UInt32[], db, Ref(Int64(0)))
end

"""
    append_items!(idx, ctx, items)

Appends all `items` elements into the index `idx`. It work in parallel using all available threads.
Grows `database(idx)` then delegates the actual indexing work to [`index!`](@ref), which is the sole
emitter of the `:add!` log event for this batch -- this function itself does not log, per the
exactly-once contract documented on [`OBSERVE`](@ref).

# Arguments:
- `idx`: The inverted index
- `items`: The database of sparse objects, it can be only indices if each object is a list of integers or a set of integers,
    `SparseVector`s, among other combinations (see [`identiterator`](@ref) for the exact set of
    natively supported object types; dense vectors are not accepted directly — convert with
    `SparseArrays.sparse` first).
- `n`: The number of items to insert (defaults to all)
"""
function append_items!(idx::AbstractInvertedFile, ctx::InvertedFileContext, items::AbstractDatabase, n=length(items))
    append_items!(idx.db, n == length(items) ? items : view(items, 1:n))
    index!(idx, ctx)
    idx
end

"""
    push_item!(idx::AbstractInvertedFile, ctx::InvertedFileContext, obj)

Inserts a single element into the index. This operation is not thread-safe.

# Arguments
- `idx`: The inverted index
- `ctx`: the index's context
- `obj`: The object to be indexed

"""
function push_item!(idx::AbstractInvertedFile, ctx::InvertedFileContext, obj, objID=length(idx) + 1)
    nz = internal_push_object!(idx, ctx, objID, obj)
    for tokenID in identiterator(idx.dist, obj)
        N = neighbors(idx.adj, tokenID)
        N === nothing && continue
        sort_postinglist!(idx.adj, N)
    end
    push!(idx.sizes, nz)
    push_item!(idx.db, obj)
    idx.len[] += 1
    OBSERVE(ctx, :add!, idx, objID, objID)
    @inform ctx "add! sp=$objID ep=$objID" index=idx
    idx
end

function internal_push_object!(idx::AbstractInvertedFile, ctx::InvertedFileContext, objID::Integer, obj)
    nz = 0
    @inbounds for tokenID in identiterator(idx.dist, obj)
        (tokenID isa Number && tokenID == 0) && continue  # object 0 is a sentinel
        nz += 1
        internal_push!(idx, ctx, tokenID, objID)
    end

    nz
end

internal_push!(idx::InvertedFile{<:Any,<:AbstractAdjList{UInt32}}, ctx::InvertedFileContext, tokenID, objID) =
    add!(idx.adj, tokenID, (objID,))

"""
    sort_postinglist!(adj::AbstractAdjList, N)

Sorts a single posting list `N` (as returned by `neighbors(adj, tokenID)`) back into the order the
merge/search algorithms rely on: ascending by id, for plain token adjacency (`UInt32`). Override for
a different concrete adjacency element type (e.g. a compressed encoding).
"""
sort_postinglist!(::AbstractAdjList{UInt32}, N) = sort!(N)

"""
    index!(idx::AbstractInvertedFile, ctx::InvertedFileContext)

Builds postings for every object already present in `database(idx)` but not yet indexed, i.e.
the block `database(idx)[length(idx)+1 : length(database(idx))]`. It is a no-op (nothing is
logged) if `db` has not grown past `length(idx)`. Mirrors `SearchGraph`'s `index!`: grow
`database(idx)` first (e.g. `push_item!(database(idx), obj)` / `append_items!(database(idx),
items)`), then call `index!(idx, ctx)` to catch up. `push_item!`/`append_items!` on `idx` itself
already call this internally, so it only needs to be called explicitly when `db` was grown
directly. This is the sole emitter of the `:add!` log event for the batch it indexes -- see the
exactly-once contract documented on [`OBSERVE`](@ref).
"""
function index!(idx::AbstractInvertedFile, ctx::InvertedFileContext)
    sp = idx.len[] + 1
    n = length(database(idx))
    if sp <= n
        _index_block!(idx, ctx, sp, n)
        idx.len[] = n
        OBSERVE(ctx, :add!, idx, sp, n)
        @inform ctx "add! sp=$sp ep=$n" index=idx
    end
    idx
end

"""
    _index_block!(idx::AbstractInvertedFile, ctx::InvertedFileContext, sp::Int, n::Int)

Per-type hook for [`index!`](@ref): builds postings and any bookkeeping (e.g. `sizes`/`doclens`)
for `database(idx)[sp:n]`, resizing bookkeeping vectors as needed. The caller (`index!`) updates
`idx.len[]` afterward.
"""
function _index_block!(idx::InvertedFile, ctx::InvertedFileContext, sp::Int, n::Int)
    resize!(idx.sizes, n)
    minbatch = getminbatch(n - sp + 1)

    @BATCHES minbatch scheduler=ctx.scheduler for objID in sp:n
        idx.sizes[objID] = internal_push_object!(idx, ctx, objID, idx.db[objID])
    end

    keys_vec = collect(eachindex(idx.adj))
    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:length(keys_vec)
        N = neighbors(idx.adj, keys_vec[i])
        N === nothing && continue
        sort_postinglist!(idx.adj, N)
    end

    idx
end
