# This file is a part of SimilaritySearch.jl

export AdjDict

"""
    struct AdjDict{T} <: AbstractAdjList{T}

Dict-of-vectors adjacency-list representation of a graph, backed by a `Dict{T,Vector{T}}`. Node
`i`'s neighbors are stored in `end_point[i]`. Unlike [`AdjList`](@ref), node ids need not be
contiguous integers starting at 1, making this backend useful for sparse or non-contiguous node
id spaces.

# Fields
- `end_point`: dictionary mapping a node id to its vector of neighbor ids.
- `glock`: a `ReentrantLock` guarding mutation (`add!`) for thread-safety.

# Examples
```julia
adj = AdjDict(Int32, 0)
add!(adj, 1, Int32[2, 3])
neighbors(adj, 1)  # => Int32[2, 3]
```
"""
struct AdjDict{T} <: AbstractAdjList{T}
    end_point::Dict{T,Vector{T}} # ending point of the i-th edge
    glock::Threads.ReentrantLock # global locks
end

Base.eltype(::AdjDict{T}) where T = Pair{T,Vector{T}}
Base.eachindex(adj::AdjDict) = keys(adj.end_point)

function Base.iterate(adj::AdjDict{T}, state=nothing) where T
    S = state === nothing ? iterate(adj.end_point) : iterate(adj.end_point, state)
    S === nothing && return nothing
    S
end

"""
    AdjDict(L::Dict{T,Vector{T}}) where T

Wraps an existing dictionary `L` of neighbor lists as an `AdjDict{T}` (no copy is made).
"""
function AdjDict(L::Dict{T,Vector{T}}) where T
    AdjDict{T}(L, Threads.ReentrantLock())
end

"""
    AdjDict(L::Vector{Vector{T}}) where T

Creates an `AdjDict{T}` from a dense vector of neighbor lists `L`, keyed by their (1-based)
position in `L`.
"""
function AdjDict(L::Vector{Vector{T}}) where T
    AdjDict{T}(Dict(pairs(L)), Threads.ReentrantLock())
end

"""
    AdjDict(::Type{T}, n::Int) where T -> AdjDict{T}

Creates an empty `AdjDict{T}`, with its internal dictionary sized with `sizehint!(_, n)` as a
capacity hint for `n` nodes.

# Examples
```julia
adj = AdjDict(Int32, 100)  # hint capacity for ~100 nodes
add!(adj, 42, Int32[7])    # ids need not be contiguous
```
"""
function AdjDict(::Type{T}, n::Int) where T
    L = Dict{T,Vector{T}}()
    sizehint!(L, n)
    AdjDict(L)
end

"""
    AdjDict(::Type{T}; n::Int=0) where T -> AdjDict{T}

Keyword-argument variant of `AdjDict(::Type{T}, n::Int)`.
"""
AdjDict(::Type{T}; n::Int=0) where T = AdjDict(T, n::Int)

"""
    Base.resize!(adj::AdjDict, n) -> Nothing

No-op for `AdjDict`: since it is backed by a `Dict`, it does not need explicit preallocation of
node slots (unlike [`AdjList`](@ref)'s `resize!`). Kept only to satisfy the common `AbstractAdjList`
interface.
"""
function Base.resize!(adj::AdjDict, n)
    # do nothing
end

"""
    AdjDict(adj::AdjDict) -> AdjDict

Creates a deep copy of `adj`.
"""
AdjDict(adj::AdjDict) = AdjDict(deepcopy(adj.end_point))
@inline Base.length(adj::AdjDict) = length(adj.end_point)

"""
    neighbors(adj::AdjDict, i) -> Vector or Nothing

Returns the list of neighbors of node `i` in `adj`, or `nothing` if node `i` has no entry in
`adj.end_point` (it is the caller's responsibility to ensure `i` refers to an initialized node
when a list is expected).

# Examples
```julia
adj = AdjDict(Int32, 0)
add!(adj, 1, Int32[2, 3])
neighbors(adj, 1)  # => Int32[2, 3]
neighbors(adj, 2)  # => nothing (no entry)
```
"""
Base.@propagate_inbounds @inline function neighbors(adj::AdjDict, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    get(adj.end_point, i, nothing)
end

"""
    neighbors_length(adj::AdjDict, i) -> Int

Returns the number of neighbors stored for node `i` in `adj`, or `0` if node `i` has no entry.
"""
Base.@propagate_inbounds @inline function neighbors_length(adj::AdjDict, i)
    # we can access undefined posting lists, it is responsability of the algorithm to ensure this doesn't happens
    L = get(adj.end_point, i, nothing)
    L === nothing ? 0 : length(L)
end

"""
    add!(adj::AdjDict{T}, n, N) where T -> AdjDict

Adds the neighbors in `N` (an iterable of ids convertible to `T`) to node `n`'s neighbor list. If
node `n` already has an entry, `N` is appended to it; otherwise a new list is created from `N`.
Thread-safe via `adj.glock`.

# Examples
```julia
adj = AdjDict(Int32, 0)
add!(adj, 1, Int32[2])
add!(adj, 1, Int32[3])   # appends to node 1's existing list
neighbors(adj, 1)        # => Int32[2, 3]
```
"""
Base.@propagate_inbounds @inline function add!(adj::AdjDict{T}, n, N) where T
    lock(adj.glock) do
        L = get(adj.end_point, n, nothing)
        if L === nothing
            adj.end_point[n] = collect(T, N)
        else
            append!(L, N)
        end
    end

    adj
end

"""
    add!(adj::AdjDict{T}, iter) where T -> AdjDict

Bulk version of `add!`: `iter` yields `(n, N)` pairs, each adding neighbor set `N` to node `n`.
Thread-safe via `adj.glock`.

# Examples
```julia
adj = AdjDict(Int32, 0)
add!(adj, [(1, Int32[2, 3]), (5, Int32[1])])
neighbors(adj, 5)  # => Int32[1]
```
"""
Base.@propagate_inbounds @inline function add!(adj::AdjDict{T}, iter) where T
    lock(adj.glock) do
        for (n, N) in iter
            L = get(adj.end_point, n, nothing)
            if L === nothing
                adj.end_point[n] = collect(T, N)
            else
                append!(L, N)
            end
        end
    end

    adj
end

