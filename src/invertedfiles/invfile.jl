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

Number of indexed elements
"""
Base.length(idx::AbstractInvertedFile) = length(idx.sizes)

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
- `sizes`: number of non-zero values in each element (non-zero values in columns).
- `db`: the original indexed objects, one per identifier; always populated by `push_item!`/`append_items!`.

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
"""
function InvertedFile(vocsize::Integer, dist::PreMetric=Dist.Sets.Jaccard(); db::AbstractDatabase=VectorDatabase(Any[]))
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    InvertedFile(dist, resize!(AdjList(UInt32), vocsize), UInt32[], db)
end

function getcontainer(idx::AbstractInvertedFile, ctx::InvertedFileContext)
    Q = getcontainer(idx.adj, ctx)
    empty!(Q)
    Q
end

getcontainer(adj::AdjList{UInt32}, ctx) = ctx.buffer[ctx.batchid]

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

Iterator over the plain integer ids in `obj`, for callers that only need to know *which* ids
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

Converts an element of an [`identiterator`](@ref) fallback into a plain id, discarding any paired
weight.
"""
convertident(u::Integer) = u
convertident(u::Tuple) = u[1] # assert length(u) = 2
convertident(u::Vector) = u[1] # assert length(u) = 2
convertident(u::Pair) = u[1]

"""
    append_items!(idx, ctx, items)

Appends all `items` elements into the index `idx`. It work in parallel using all available threads.

# Arguments:
- `idx`: The inverted index
- `items`: The database of sparse objects, it can be only indices if each object is a list of integers or a set of integers,
    `SparseVector`s, among other combinations (see [`identiterator`](@ref) for the exact set of
    natively supported object types; dense vectors are not accepted directly — convert with
    `SparseArrays.sparse` first).
- `n`: The number of items to insert (defaults to all)
"""
function append_items!(idx::AbstractInvertedFile, ctx::InvertedFileContext, items::AbstractDatabase, n=length(items))
    startID = length(idx)
    _parallel_append!(idx, ctx, items, startID, n)
    LOG(ctx.logger, :append_items!, idx, ctx, startID, length(idx))
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
    LOG(ctx.logger, :push_item!, idx, ctx, objID, objID)
    idx
end

function internal_push_object!(idx::AbstractInvertedFile, ctx::InvertedFileContext, objID::Integer, obj)
    nz = 0
    @inbounds for tokenID in identiterator(idx.dist, obj)
        tokenID == 0 && continue  # object 0 is a centinel
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

function _parallel_append!(idx::AbstractInvertedFile, ctx::InvertedFileContext, items::AbstractDatabase, startID::Int, n::Int)
    internal_parallel_prepare_append!(idx, startID + n)
    minbatch = getminbatch(n)

    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:n
        objID = i + startID
        idx.sizes[objID] = internal_push_object!(idx, ctx, objID, items[i])
    end

    append_items!(idx.db, n == length(items) ? items : view(items, 1:n))

    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:length(idx.adj)
        N = neighbors(idx.adj, i)
        N === nothing && continue
        sort_postinglist!(idx.adj, N)
    end

    idx
end

function internal_parallel_prepare_append!(idx::AbstractInvertedFile, new_size::Integer)
    resize!(idx.sizes, new_size)
end
