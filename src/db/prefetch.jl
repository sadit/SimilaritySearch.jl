# This file is a part of SimilaritySearch.jl

"""
    prefetch_item(db::AbstractDatabase, i::Integer)

Asks the hardware to start bringing the storage of item `i` into the caches, without touching
it: the graph search calls it on the unvisited neighbours of a vertex *before* it evaluates
any of them, so their cache misses overlap instead of being paid one after another
(`beamsearch_inner_beam`). The default does nothing; databases with contiguous storage
prefetch the first `PREFETCH_LINES` cache lines of the item (the hardware streamer follows
from there) and whatever per-item side data their distance reads.
"""
@inline prefetch_item(::AbstractDatabase, ::Integer) = nothing

"At most this many 64-byte lines are prefetched per item: enough for 512 bytes of codes or floats."
const PREFETCH_LINES = 8

# `llvm.prefetch(ptr, rw=0 read, locality=3 keep in every level, cache type=1 data)`; the tuple
# form of `llvmcall` takes a module and its entry function (opaque pointers: LLVM 15+, Julia 1.10+)
const _PREFETCH_IR = """
declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
define void @prefetch_entry(ptr %p) alwaysinline {
    call void @llvm.prefetch.p0(ptr %p, i32 0, i32 3, i32 1)
    ret void
}
"""
@inline function _prefetch(p::Ptr)
    Base.llvmcall((_PREFETCH_IR, "prefetch_entry"), Cvoid, Tuple{Ptr{Int8}}, Ptr{Int8}(p))
end

"Prefetches the first lines of `nbytes` bytes at `p`."
@inline function _prefetch_bytes(p::Ptr, nbytes::Integer)
    q = Ptr{Int8}(p)
    for off in 0:64:min(nbytes, 64 * PREFETCH_LINES) - 1
        _prefetch(q + off)
    end
end

@inline function prefetch_item(db::MatrixDatabase{M}, i::Integer) where {T,M<:DenseMatrix{T}}
    m = db.matrix
    rows = size(m, 1)
    _prefetch_bytes(pointer(m, (i - 1) * rows + 1), rows * sizeof(T))
end

# the storage `sqcodes`/`SQEncoder` build and every `push_item!`-grown database use: blocks of
# 2^NumBits columns, item `i` at column `j` of block `b`
@inline function prefetch_item(db::BlockMatrixDatabase{Dim,NumType,NumBits}, i::Integer) where {Dim,NumType,NumBits}
    b, j = _get_block_and_pos(NumBits, i)
    @inbounds m = db.blocks[b]
    _prefetch_bytes(pointer(m, (j - 1) * Dim + 1), Dim * sizeof(NumType))
end

# a view onto another database: the parent's item
@inline prefetch_item(db::SubDatabase, i::Integer) = prefetch_item(db.parent, db.map[i])

# memory-mapped matrix: the same layout as a MatrixDatabase, backed by the file's pages
@inline function prefetch_item(db::MMapMatrixDatabase{Dim,NumType}, i::Integer) where {Dim,NumType}
    _prefetch_bytes(pointer(db.data, (i - 1) * Dim + 1), Dim * sizeof(NumType))
end

# one heap object per item (sets, strings, variable-length vectors): reaching the data costs the
# reference and the object's header first, two dependent loads the second pass then finds in
# cache; the data lines are what gets prefetched
@inline function prefetch_item(db::VectorDatabase, i::Integer)
    @inbounds v = db.vecs[i]
    _prefetch_object(v)
end
@inline _prefetch_object(v::Union{Vector,String}) = _prefetch_bytes(pointer(v), sizeof(v))
@inline _prefetch_object(::Any) = nothing
