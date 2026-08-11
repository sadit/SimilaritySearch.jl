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

getcontainer(adj::AdjList{UInt32}, ctx) = ctx.cont_u32[ctx.batchid]

function getcontainer(adj::StaticAdjList, ctx)
    Q = [PostingList(neighbors(adj, 1), zero(UInt32), 0.0f0)]
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
    sparseiterator(db, i)

Creates an iterator for indices and values of the `i`-th db's element (e.g., column).
Several specializations are provided.
"""
function sparseiterator(db::MatrixDatabase{<:SparseMatrixCSC}, i)
    sparseiterator(db.matrix, i)
end

function sparseiterator(X::SparseMatrixCSC, i)
    r = nzrange(X, i)
    rows = rowvals(X)
    vals = nonzeros(X)
    zip(view(rows, r), view(vals, r))
end

function sparseiterator(vec::SubArray{<:AbstractFloat, 1, <:SparseMatrixCSC})  # to efficiently support views
    _, i = vec.indices
    sparseiterator(vec.parent, i)
end

sparseiterator(db::MatrixDatabase{<:Matrix}, i) = enumerate(view(db.matrix, i))
sparseiterator(db::AbstractDatabase, i) = sparseiterator(db[i])

"""
    sparseiterator(obj)

`(id, weight)` iterator for `obj` for generic databases.
"""
sparseiterator(obj::AbstractVector{<:AbstractFloat}) = enumerate(obj)
sparseiterator(obj::SparseVector) = zip(obj.nzind, obj.nzval)
sparseiterator(obj::SparseVecView) = zip(obj.nzind, obj.nzval)
sparseiterator(obj::Set) = (convertpair(u) for u in obj)
sparseiterator(obj::SortedIntSet) = (convertpair(u) for u in obj)
sparseiterator(obj) = (convertpair(u) for u in obj)

"""
    sparseiterator(dist::PreMetric, obj)

Distance-aware `(id, weight)` iterator for `obj`. Defaults to the distance-agnostic
[`sparseiterator(obj)`](@ref) dispatch tree above; overload this for a specific `(DistType, ObjType)`
pair when the same native object type must be tokenized/weighted differently depending on which distance
the enclosing index is built for (e.g. a different candidate-generation encoding for a sequence distance).
"""
sparseiterator(::PreMetric, obj) = sparseiterator(obj)

"""
    convertpair(u)

Converts an element of an `sparseiterator` into an usable pair.
"""
convertpair(u::Integer) = (u, 1)
convertpair(u::Tuple) = u # assert length(u) = 2
convertpair(u::Vector) = u # assert length(u) = 2
convertpair(u::Pair) = u

"""
    append_items!(idx, ctx, items; tol=1e-6)

Appends all `items` elements into the index `idx`. It work in parallel using all available threads.

# Arguments:
- `idx`: The inverted index
- `items`: The database of sparse objects, it can be only indices if each object is a list of integers or a set of integers,
    sparse matrices, dense matrices, among other combinations.
- `n`: The number of items to insert (defaults to all)

# Keyword arguments:
- `tol`: controls what is a zero (i.e., weights < tol will be ignored).
"""
function append_items!(idx::AbstractInvertedFile, ctx::InvertedFileContext, items::AbstractDatabase, n=length(items); tol::Float64=1e-6)
    startID = length(idx)
    parallel_append!(idx, ctx, items, startID, n, tol)
    LOG(ctx.logger, :append_items!, idx, ctx, startID, length(idx))
    idx
end

"""
    push_item!(idx::AbstractInvertedFile, ctx::InvertedFileContext, obj; tol=1e-6)

Inserts a single element into the index. This operation is not thread-safe.

# Arguments
- `idx`: The inverted index
- `ctx`: the index's context
- `obj`: The object to be indexed

# Keyword arguments
- `tol`: controls what is a zero (i.e., `weight < tol` will be ignored)
"""
function push_item!(idx::AbstractInvertedFile, ctx::InvertedFileContext, obj, objID=length(idx) + 1; tol=1e-6)
    nz = internal_push_object!(idx, ctx, objID, obj, tol)
    for (tokenID, _) in sparseiterator(idx.dist, obj)
        N = neighbors(idx.adj, tokenID)
        N === nothing && continue
        sort_postinglist!(idx.adj, N)
    end
    push!(idx.sizes, nz)
    push_item!(idx.db, obj)
    LOG(ctx.logger, :push_item!, idx, ctx, objID, objID)
    idx
end

function internal_push_object!(idx::AbstractInvertedFile, ctx::InvertedFileContext, objID::Integer, obj, tol::Float64)
    nz = 0
    @inbounds for (tokenID, weight) in sparseiterator(idx.dist, obj)
        weight < tol && continue
        tokenID == 0 && continue  # object 0 is a centinel
        nz += 1
        internal_push!(idx, ctx, tokenID, objID, weight)
    end

    nz
end

internal_push!(idx::InvertedFile{<:Any,<:AbstractAdjList{UInt32}}, ctx::InvertedFileContext, tokenID, objID, _) =
    add!(idx.adj, tokenID, (objID,))

"""
    sort_postinglist!(adj::AbstractAdjList, N)

Sorts a single posting list `N` (as returned by `neighbors(adj, tokenID)`) back into the order the
merge/search algorithms rely on: ascending by id, for plain token adjacency (`UInt32`). Override for
a different concrete adjacency element type (e.g. a compressed encoding).
"""
sort_postinglist!(::AbstractAdjList{UInt32}, N) = sort!(N)

function parallel_append!(idx::AbstractInvertedFile, ctx::InvertedFileContext, items::AbstractDatabase, startID::Int, n::Int, tol::Float64)
    internal_parallel_prepare_append!(idx, startID + n)
    minbatch = getminbatch(n)

    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:n
        objID = i + startID
        idx.sizes[objID] = internal_push_object!(idx, ctx, objID, items[i], tol)
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
