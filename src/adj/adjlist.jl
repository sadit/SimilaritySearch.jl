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

Base.eltype(adj::AdjList{T}) where T = Pair{T,Vector{T}}
Base.eachindex(adj::AdjList) = eachindex(adj.end_point)

function Base.iterate(adj::AdjList{T}, i=1) where T
    i = T(i)
    n = length(adj)
    (n == 0 || i > n) && return nothing
    i => neighbors(adj, i), i+1
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
function Base.resize!(adj::AdjList, n::Integer)
    lock(adj.glock) do
        resize!(adj.end_point, n)
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
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? adj.end_point[i] : nothing
end

"""
    neighbors_length(adj::AdjList, i) -> Int

Returns the number of neighbors stored for node `i` in `adj`, or `0` if node `i` has never been
assigned a neighbor list.
"""
Base.@propagate_inbounds @inline function neighbors_length(adj::AdjList, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    isassigned(adj.end_point, i) ? length(adj.end_point[i]) : 0
end

"""
    add!(adj::AdjList{T}, n::Integer, N) where T -> AdjList

Adds the neighbors in `N` (an iterable of ids convertible to `T`) to node `n`'s neighbor list.
Grows `adj` first (via `resize!`) if `n` exceeds its current length. If node `n` already has a
neighbor list, `N` is appended to it; otherwise a new list is created from `N`. Thread-safe via
`adj.glock`.

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
        
        if isassigned(adj.end_point, n)
            append!(adj.end_point[n], N)
        else
            adj.end_point[n] = collect(T, N)
        end
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
