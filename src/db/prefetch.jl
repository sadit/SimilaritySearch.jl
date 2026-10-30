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

"""
    prefetchable(db::AbstractDatabase) -> Bool

Whether [`prefetch_item`](@ref) does anything for `db`. The graph search skips its prefetch pass
when it does not, so a database without contiguous storage pays nothing.
"""
@inline prefetchable(::AbstractDatabase) = false

"An item of up to this many 64-byte lines (512 bytes: codes, sketches, Float16 at 256 dimensions) is prefetched whole, into every cache level."
const PREFETCH_LINES = 8
"""
A larger item only has this many leading lines prefetched, into L2 and beyond: the hardware streamer follows, and
L1 stays with the query, the beam and the visited words (768-byte Float16 vectors: 1.12x at recall 0.90).

The best value depends on the hardware and on the number of threads, so it is left at 4. Measured on 1.6.9 with a
100k subset of wikipedia-bge-m3 (1024-byte 8-bit codes, the same graph and visits for every value; Xeon Silver
4216, 32 KB L1d and 1 MB L2 per core, two hyperthreads per core), QPS at recall 0.9 relative to 4 lines:
- 62 threads: 8 lines +8.5%, 16 lines -10%;
- 1 thread: 8 lines -4%, 16 lines +15.6%.
With one thread the search waits on latency, and bringing the whole item wins. With 62 threads, two hyperthreads
share each L1 and L2 and all of them share the memory bandwidth, and bringing the whole item saturates it. A
different processor, cache size or thread count may move the best value either way.
"""
const PREFETCH_LINES_LARGE = 4
"Items above this size are not prefetched at all: 1536-byte Float32 vectors lost 8-13% with 8 lines into L1 and 4-11% with 4 lines into L2 (the search is half compute there, and the pass costs more than the misses it hides)."
const PREFETCH_MAX_BYTES = 1024

# `llvm.prefetch(ptr, rw=0 read, locality, cache type=1 data)`; locality 3 keeps the line in every
# level, 2 in L2 and beyond. The tuple form of `llvmcall` takes a module and its entry function.
# The entry receives the address as an `i64` and converts it with `inttoptr`, the form every
# supported Julia accepts and inlines (issue #130). Julia 1.10 (LLVM 15) parses typed pointers
# only, so it gets `i8*` and the intrinsic `p0i8`. Julia 1.11 accepted an entry taking `ptr`
# but did not inline it, leaving a call per prefetched line.
const _PREFETCH_PTR, _PREFETCH_SUFFIX = VERSION < v"1.11" ? ("i8*", "p0i8") : ("ptr", "p0")
_prefetch_ir(entry, locality) = """
declare void @llvm.prefetch.$_PREFETCH_SUFFIX($_PREFETCH_PTR, i32, i32, i32)
define void @$entry(i64 %a) alwaysinline {
    %p = inttoptr i64 %a to $_PREFETCH_PTR
    call void @llvm.prefetch.$_PREFETCH_SUFFIX($_PREFETCH_PTR %p, i32 0, i32 $locality, i32 1)
    ret void
}
"""
const _PREFETCH_IR = _prefetch_ir("prefetch_entry", 3)
const _PREFETCH_IR_L2 = _prefetch_ir("prefetch_entry_l2", 2)
@inline function _prefetch(p::Ptr)
    Base.llvmcall((_PREFETCH_IR, "prefetch_entry"), Cvoid, Tuple{Ptr{Int8}}, Ptr{Int8}(p))
end
@inline function _prefetch_l2(p::Ptr)
    Base.llvmcall((_PREFETCH_IR_L2, "prefetch_entry_l2"), Cvoid, Tuple{Ptr{Int8}}, Ptr{Int8}(p))
end

"Prefetches an item of `nbytes` bytes at `p`: whole into L1 when it fits `PREFETCH_LINES`, its first `PREFETCH_LINES_LARGE` lines into L2 otherwise."
@inline function _prefetch_bytes(p::Ptr, nbytes::Integer)
    q = Ptr{Int8}(p)
    if nbytes <= 64 * PREFETCH_LINES
        for off in 0:64:nbytes - 1
            _prefetch(q + off)
        end
    else
        for off in 0:64:64 * PREFETCH_LINES_LARGE - 1
            _prefetch_l2(q + off)
        end
    end
end

@inline prefetchable(db::MatrixDatabase{M}) where {T,M<:DenseMatrix{T}} = size(db.matrix, 1) * sizeof(T) <= PREFETCH_MAX_BYTES
@inline function prefetch_item(db::MatrixDatabase{M}, i::Integer) where {T,M<:DenseMatrix{T}}
    m = db.matrix
    rows = size(m, 1)
    _prefetch_bytes(pointer(m, (i - 1) * rows + 1), rows * sizeof(T))
end

# the storage `sqcodes`/`SQEncoder` build and every `push_item!`-grown database use: blocks of
# 2^NumBits columns, item `i` at column `j` of block `b`
@inline prefetchable(::BlockMatrixDatabase{Dim,NumType}) where {Dim,NumType} = Dim * sizeof(NumType) <= PREFETCH_MAX_BYTES
@inline function prefetch_item(db::BlockMatrixDatabase{Dim,NumType,NumBits}, i::Integer) where {Dim,NumType,NumBits}
    b, j = _get_block_and_pos(NumBits, i)
    @inbounds m = db.blocks[b]
    _prefetch_bytes(pointer(m, (j - 1) * Dim + 1), Dim * sizeof(NumType))
end

# a view onto another database: the parent's item
@inline prefetchable(db::SubDatabase) = prefetchable(db.parent)
@inline prefetch_item(db::SubDatabase, i::Integer) = prefetch_item(db.parent, db.map[i])

# memory-mapped matrix: the same layout as a MatrixDatabase, backed by the file's pages
@inline prefetchable(::MMapMatrixDatabase{Dim,NumType}) where {Dim,NumType} = Dim * sizeof(NumType) <= PREFETCH_MAX_BYTES
@inline function prefetch_item(db::MMapMatrixDatabase{Dim,NumType}, i::Integer) where {Dim,NumType}
    _prefetch_bytes(pointer(db.data, (i - 1) * Dim + 1), Dim * sizeof(NumType))
end

# one heap object per item (sets, strings, variable-length vectors): reaching the data costs the
# reference and the object's header first, two dependent loads the second pass then finds in
# cache; the data lines are what gets prefetched
@inline prefetchable(::VectorDatabase{V}) where {V} = eltype(V) <: Union{Vector,String}
@inline function prefetch_item(db::VectorDatabase, i::Integer)
    @inbounds v = db.vecs[i]
    _prefetch_object(v)
end
@inline _prefetch_object(v::Union{Vector,String}) = (sizeof(v) <= PREFETCH_MAX_BYTES && _prefetch_bytes(pointer(v), sizeof(v)); nothing)
@inline _prefetch_object(::Any) = nothing
