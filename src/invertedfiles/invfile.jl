# This file is part of InvertedFiles.jl

using LinearAlgebra, SparseArrays
export AbstractInvertedFile

"""
    abstract type AbstractInvertedFile <: AbstractSearchIndex end

Abstract inverted file, actual data structures are [`WeightedInvertedFile`](@ref) and [`BinaryInvertedFile`](@ref)
"""
abstract type AbstractInvertedFile <: AbstractSearchIndex end

"""
    length(idx::AbstractInvertedFile)

Number of indexed elements
"""
Base.length(idx::AbstractInvertedFile) = length(idx.sizes)
Base.show(io::IO, idx::AbstractInvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.adj)), n=$(length(idx))}")
database(idx::AbstractInvertedFile) = nothing

function getcontainer(idx::AbstractInvertedFile, ctx::InvertedFileContext)
    Q = getcontainer(idx.adj, ctx)
    empty!(Q)
    Q
end

getcontainer(adj::AdjList{UInt32}, ctx) = ctx.cont_u32[ctx.batchid]
getcontainer(adj::AdjList{IdWeight}, ctx) = ctx.cont_iw[ctx.batchid]
getcontainer(adj::AdjList{IdIntWeight}, ctx) = ctx.cont_iiw[ctx.batchid]

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
sparseiterator(obj::Set) = (convertpair(u) for u in obj)
sparseiterator(obj::SortedIntSet) = (convertpair(u) for u in obj)
sparseiterator(obj) = (convertpair(u) for u in obj)

"""
    convertpair(u)

Converts an element of an `sparseiterator` into an usable pair.
"""
convertpair(u::Integer) = (u, 1)
convertpair(u::Tuple) = u # assert length(u) = 2
convertpair(u::Vector) = u # assert length(u) = 2
convertpair(u::Pair) = u
convertpair(u::IdWeight) = (u.id, u.weight)
convertpair(u::IdIntWeight) = (u.id, u.weight)

#=function SimilaritySearch.index!(idx::AbstractInvertedFile, ctx::InvertedFileContext; tol=1e-6)
    startID = length(idx)
    db = database(idx)
    n = length(db) - startID
    n == 0 && return idx
    parallel_append!(idx, ctx, db, startID, n, tol, true)
end=#

"""
    append_items!(idx, ctx, items; tol=1e-6)

Appends all `items` elements into the index `idx`. It work in parallel using all available threads.

# Arguments:
- `idx`: The inverted index
- `items`: The database of sparse objects, it can be only indices if each object is a list of integers or a set of integers (useful for `BinaryInvertedFile`),
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
    for (tokenID, _) in sparseiterator(obj)
        N = neighbors(idx.adj, tokenID)
        N === nothing && continue
        sort!(N)
    end
    push!(idx.sizes, nz)
    !isnothing(idx.db) && push_item!(idx.db, obj)
    LOG(ctx.logger, :push_item!, idx, ctx, objID, objID)
    idx
end

function internal_push_object!(idx::AbstractInvertedFile, ctx::InvertedFileContext, objID::Integer, obj, tol::Float64)
    nz = 0
    @inbounds for (tokenID, weight) in sparseiterator(obj)
        weight < tol && continue
        tokenID == 0 && continue  # object 0 is a centinel
        nz += 1
        internal_push!(idx, ctx, tokenID, objID, weight)
    end

    nz
end

function parallel_append!(idx, ctx::InvertedFileContext, db::AbstractDatabase, startID::Int, n::Int, tol::Float64)
    internal_parallel_prepare_append!(idx, startID + n)
    minbatch = getminbatch(n)

    @BATCHES minbatch scheduler=ctx.scheduler for i in 1:n
        objID = i + startID
        idx.sizes[objID] = internal_push_object!(idx, ctx, objID, db[i], tol)
    end

    if idx isa BinaryInvertedFile
        @BATCHES minbatch scheduler=ctx.scheduler for i in 1:length(idx.adj)
            N = neighbors(idx.adj, i)
            N === nothing && continue
            sort!(N)
        end
    elseif idx isa WeightedInvertedFile
        @BATCHES minbatch scheduler=ctx.scheduler for i in 1:length(idx.adj)
            N = neighbors(idx.adj, i)
            N === nothing && continue
            sort!(N, by=p -> p.id)
        end
    else
        throw(ArgumentError("Unknown invertedfile type $(typeof(idx))"))
    end

    idx
end

function internal_parallel_prepare_append!(idx::AbstractInvertedFile, new_size::Integer)
    resize!(idx.sizes, new_size)
end
