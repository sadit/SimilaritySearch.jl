# This file is a part of SimilaritySearch.jl

export set_page_spread!, page_spread, spreadcopy

# Where the pages of a large structure end up. Linux and Windows place an anonymous page on the
# NUMA node of the thread that first writes it, and Linux backs a 2 MB-aligned stretch of a large
# allocation with one huge page. A structure written by one thread therefore sits on one node
# (every other node reaches it across the interconnect), and one made of many small allocations
# rarely gets huge pages. Large stores here are allocated uninitialized and written by all the
# threads, each one a contiguous stretch, so their pages spread over the nodes the threads run on.

const PAGE_SPREAD = Ref(true)

"""
    set_page_spread!(on::Bool)

Whether large stores are written by all threads (`true`, the default since 1.6.5) or by the
calling one (`false`, the behavior up to 1.6.4).

With `true`:
- bulk `append_items!` into a [`BlockMatrixDatabase`](@ref) or a quantized database fills the
  new items in parallel, in contiguous stretches, so the first write of each page comes from a
  different thread and the pages spread over the NUMA nodes;
- [`StaticAdjList`](@ref) fills its arrays the same way;
- a new `BlockMatrixDatabase` defaults to blocks of at least 32 MB, which the kernel backs with
  huge pages almost entirely.

With `false`, all of that runs on the calling thread and blocks default to 256 columns, as
before. It applies to structures built after the call. It never changes results, only where
the memory lies.
"""
set_page_spread!(on::Bool) = (PAGE_SPREAD[] = on)

"""
    page_spread() -> Bool

See [`set_page_spread!`](@ref).
"""
page_spread() = PAGE_SPREAD[]

# below this many items a store is filled by the caller: there is nothing worth spreading
const SPREAD_MIN = 1 << 14

"""
    _spread(f, m)

Calls `f(r)` on contiguous ranges `r` covering `1:m`, one per thread. Outside any threaded region
it is `Threads.@threads :static`, so each range runs on its own thread and the first writes split
evenly over the threads (and the NUMA nodes they sit on); nested inside one (where `:static` is
refused) the ranges become tasks.
"""
function _spread(f::F, m::Integer) where {F}
    nt = Threads.nthreads()
    if nt == 1 || m < 2
        f(1:m)
        return
    end
    chunk = cld(m, nt)
    ranges = [sp:min(m, sp + chunk - 1) for sp in 1:chunk:m]
    if Threads.threadid() == 1 && ccall(:jl_in_threaded_region, Cint, ()) == 0
        Threads.@threads :static for r in ranges
            f(r)
        end
    else
        foreach(wait, [Threads.@spawn f(r) for r in ranges])
    end
end

"Whether a bulk store of `m` items should be written by all threads."
_spreads(m::Integer) = PAGE_SPREAD[] && m >= SPREAD_MIN && Threads.nthreads() > 1

"""
    spreadcopy(x)

A copy of `x` whose memory is written by all threads, so its pages spread over the NUMA nodes
(and, for large arrays, over 2 MB huge pages). Same contents, same results; meant for structures
that were written by one thread, e.g. an index read back from disk, or one built with
[`set_page_spread!`](@ref)`(false)`. It runs whatever `page_spread()` says.

Methods: arrays of plain values, [`MatrixDatabase`](@ref), [`BlockMatrixDatabase`](@ref)
(rebuilt with blocks of at least 32 MB), quantized databases, [`AdjList`](@ref),
[`StaticAdjList`](@ref), [`SearchGraph`](@ref) and [`AsymmetricSearchGraph`](@ref) (database and
adjacency; the rest is copied as is).
"""
function spreadcopy(A::Array{T}) where {T}
    isbitstype(T) || throw(ArgumentError("spreadcopy: arrays of plain values only, got $(typeof(A))"))
    B = similar(A)
    _spread(length(A)) do r
        copyto!(B, first(r), A, first(r), length(r))
    end
    B
end
