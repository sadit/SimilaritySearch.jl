# This file is a part of SimilaritySearch.jl

export AdjList,
    neighbors, neighbors_length, add!

"""
    struct AdjList{T} <: AbstractAdjList{T}

Growable adjacency-list representation of a graph, backed by a `Vector{Vector{T}}`. Node `i`'s
neighbors are stored in `end_point[i]`; nodes are addressed by contiguous integer indices
(`1:length(adj)`). This is the usual mutable backend used while a graph-based index is being
built or updated.

# Fields
- `end_point`: vector of neighbor lists, one per node (`end_point[i]` holds the ids of node `i`'s
  neighbors).
- `glock`: a `ReentrantLock` guarding mutation (`resize!`, `add!`) for thread-safety.

# Examples
```julia
adj = AdjList(Int32, 10)   # preallocate for 10 nodes
add!(adj, 1, Int32[2, 3])  # node 1 is connected to nodes 2 and 3
neighbors(adj, 1)          # => Int32[2, 3]
```
"""
struct AdjList{T} <: AbstractAdjList{T}
    end_point::Vector{Vector{T}} # ending point of the i-th edge
    glock::Threads.ReentrantLock # global locks
end

"""
    ADJ_STRIPES

The locks that serialize changes to one node's list of an [`AdjList`](@ref); node `n` takes
stripe `n % length(ADJ_STRIPES) + 1`. They are shared by every `AdjList` instead of being a
field, so the struct keeps the two fields that stored graphs were written with. `glock` still
guards the growth of `end_point`, and [`add!`](@ref) still takes it. Only
[`add_resized!`](@ref SimilaritySearch.add_resized!), whose node must already exist, appends under the stripe alone.
"""
const ADJ_STRIPES = [Threads.SpinLock() for _ in 1:4096]

# the stripe that guards node `n`'s list; every in-place change to one list takes it
@inline adjstripe(n::Integer) = @inbounds ADJ_STRIPES[(n % length(ADJ_STRIPES)) + 1]

Base.eltype(::AdjList{T}) where {T} = Pair{T,Vector{T}}
Base.eachindex(adj::AdjList) = eachindex(adj.end_point)

function Base.iterate(adj::AdjList{T}, i=1) where {T}
    i = T(i)
    n = length(adj)
    (n == 0 || i > n) && return nothing
    i => neighbors(adj, i), i + 1
end

"""
    AdjList(A::Vector{Vector{T}}) where T

Wraps an existing vector of neighbor lists `A` as an `AdjList{T}` (no copy is made).
"""
function AdjList(A::Vector{Vector{T}}) where T
    AdjList{T}(A, Threads.ReentrantLock())
end

"""
    AdjList(::Type{T}, n::Integer=0) where T -> AdjList{T}

Creates an empty `AdjList{T}` preallocated for `n` nodes (entries are left undefined until
populated with `add!`).

# Examples
```julia
adj = AdjList(Int32, 100)  # preallocate room for 100 nodes
```
"""
function AdjList(::Type{T}, n::Integer=0) where T
    AdjList(Vector{Vector{T}}(undef, n))
end

"""
    Base.resize!(adj::AdjList, n::Integer) -> AdjList

Resizes `adj` in place to hold `n` nodes (growing or shrinking `end_point` accordingly), under
`adj.glock` for thread-safety. New slots created by growth are left undefined.
"""
function Base.resize!(adj::AdjList{T}, n::Integer) where T
    lock(adj.glock) do
        prev_n = length(adj.end_point)
        resize!(adj.end_point, n)
        for i in prev_n+1:n
            adj.end_point[i] = T[]
        end
    end

    adj
end

"""
    AdjList(adj::AdjList) -> AdjList

Creates a deep copy of `adj`.
"""
AdjList(adj::AdjList) = AdjList(deepcopy(adj.end_point))
@inline Base.length(adj::AdjList) = length(adj.end_point)

"""
    neighbors(adj::AdjList, i) -> Vector or Nothing

Returns the list of neighbors of node `i` in `adj`, i.e., `adj.end_point[i]`. Returns `nothing`
if node `i` has never been assigned a neighbor list (it is the caller's responsibility to ensure
`i` refers to an initialized node when a list is expected).

# Examples
```julia
adj = AdjList(Int32, 3)
add!(adj, 1, Int32[2, 3])
neighbors(adj, 1)  # => Int32[2, 3]
neighbors(adj, 2)  # => nothing (never assigned)
```
"""
Base.@propagate_inbounds @inline function neighbors(adj::AdjList, i)
    adj.end_point[i]
end

"""
    neighbors_length(adj::AdjList, i) -> Int

Returns the number of neighbors stored for node `i` in `adj`, or `0` if node `i` has never been
assigned a neighbor list.
"""
Base.@propagate_inbounds @inline function neighbors_length(adj::AdjList, i)
    length(adj.end_point[i])
end

"""
    add!(adj::AdjList{T}, n::Integer, N) where T -> AdjList

Adds the neighbors in `N` (an iterable of ids convertible to `T`) to node `n`'s neighbor list.
Grows `adj` first (via `resize!`) if `n` exceeds its current length. If node `n` already has a
neighbor list, `N` is appended to it; otherwise a new list is created from `N`. Thread-safe via
`adj.glock`, which also makes it safe to call while other threads grow `adj`.

# Examples
```julia
adj = AdjList(Int32, 2)
add!(adj, 1, Int32[2])
add!(adj, 1, Int32[3])   # appends to node 1's existing list
neighbors(adj, 1)        # => Int32[2, 3]
```
"""
Base.@propagate_inbounds @inline function add!(adj::AdjList{T}, n::Integer, N) where T
    lock(adj.glock) do
        n > length(adj) && resize!(adj, n)
        lock(adjstripe(n)) do   # excludes a concurrent `add_resized!` on the same node
            append!(adj.end_point[n], N)
        end
    end

    adj
end

"""
    add_resized!(adj::AbstractAdjList, n::Integer, N) -> adj

[`add!`](@ref) for a node that already exists, `n <= length(adj)`, while nothing grows `adj`:
the parallel insertion of a [`SearchGraph`](@ref), which resizes the list before each block, and
its reverse links. On an [`AdjList`](@ref) it appends under the node's stripe of
[`ADJ_STRIPES`](@ref SimilaritySearch.ADJ_STRIPES) instead of `glock`, so appends to different
nodes run concurrently; other lists fall back to `add!`. A node beyond `length(adj)` also goes
through `add!`, which grows `adj`; that is for a serial caller only, since this function must
not run while another thread grows `adj`.
"""
add_resized!(adj::AbstractAdjList, n::Integer, N) = add!(adj, n, N)

function add_resized!(adj::AdjList, n::Integer, N)
    n > length(adj) && return add!(adj, n, N)   # a serial caller that did not resize first
    # An append writes only node `n`'s own list, never `end_point`, so it needs a lock for that
    # node and not the global one. With `glock` here, the reverse links of a parallel insertion
    # block queued on a single lock: 9% of the compute threads' samples on ccnews, and the
    # stripes cut indexing time per distance evaluation by 30-45% on three SISAP 2025
    # collections (#138). A `SpinLock` is enough: the critical section is one `append!`, and
    # its spin loop reaches a GC safepoint, so the allocation inside it is safe.
    l = adjstripe(n)
    lock(l)
    try
        append!(adj.end_point[n], N)
    finally
        unlock(l)
    end

    adj
end

"""
    add!(adj::AdjList{T}, iter) where T -> AdjList

Bulk version of `add!`: `iter` yields `(i, N)` pairs, each adding neighbor set `N` to node `i`.
Grows `adj` if needed to accommodate `length(iter)` nodes. Thread-safe via `adj.glock`.

# Examples
```julia
adj = AdjList(Int32, 0)
add!(adj, [(1, Int32[2, 3]), (2, Int32[1])])
neighbors(adj, 2)  # => Int32[1]
```
"""
Base.@propagate_inbounds @inline function add!(adj::AdjList{T}, iter) where T
    n = max(length(iter), length(adj))
    lock(adj.glock) do
        n > length(adj) && resize!(adj, n)

        for (i, N) in iter
            add!(adj, i, N)
        end
    end

    adj
end

# each neighbor list is allocated and written by the thread that copies it
function spreadcopy(adj::AdjList{T}) where {T}
    n = length(adj)
    E = Vector{Vector{T}}(undef, n)
    _spread(n) do r
        for i in r
            isassigned(adj.end_point, i) && (@inbounds E[i] = copy(adj.end_point[i]))
        end
    end
    AdjList(E)
end
