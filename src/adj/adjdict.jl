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
struct AdjDict{K,T} <: AbstractAdjList{T}
    end_point::Dict{K,Vector{T}} # ending point of the i-th edge
    glock::Threads.ReentrantLock # global locks
end

Base.eltype(::AdjDict{K,T}) where {K,T} = Pair{K,Vector{T}}
Base.eachindex(adj::AdjDict) = keys(adj.end_point)

function Base.iterate(adj::AdjDict, state=nothing)
    S = state === nothing ? iterate(adj.end_point) : iterate(adj.end_point, state)
    S === nothing && return nothing
    S
end

"""
    AdjDict(L::Dict{K,Vector{T}}) where {K,T}

Wraps an existing dictionary `L` of neighbor lists as an `AdjDict{K,T}` (no copy is made).
"""
function AdjDict(L::Dict{K,Vector{T}}) where {K,T}
    AdjDict{K,T}(L, Threads.ReentrantLock())
end

"""
    AdjDict(L::Vector{Vector{T}}) where T

Creates an `AdjDict{Int,T}` from a dense vector of neighbor lists `L`, keyed by their (1-based)
position in `L`.
"""
function AdjDict(L::Vector{Vector{T}}) where T
    AdjDict{Int,T}(Dict(pairs(L)), Threads.ReentrantLock())
end

"""
    AdjDict(::Type{T}, n::Integer=0) where T -> AdjDict{T,T}

Creates an empty `AdjDict{T,T}`, with key type `T` and value type `T`.
"""
AdjDict(::Type{T}, n::Integer=0) where T = AdjDict(T, T, n)

"""
    AdjDict(::Type{K}, ::Type{T}, n::Integer) where {K,T} -> AdjDict{K,T}
    AdjDict(::Type{K}, ::Type{T}; n::Integer=0) where {K,T} -> AdjDict{K,T}

Creates an empty `AdjDict{K,T}`, with key type `K` and value type `T`, sizing its internal dictionary
with `sizehint!(_, n)` as a capacity hint for `n` keys.
"""
function AdjDict(::Type{K}, ::Type{T}, n::Integer) where {K,T}
    L = Dict{K,Vector{T}}()
    n > 0 && sizehint!(L, n)
    AdjDict(L)
end

AdjDict(::Type{K}, ::Type{T}; n::Integer=0) where {K,T} = AdjDict(K, T, n)

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
    lock(adj.glock) do
        get(adj.end_point, i, nothing)
    end
end

"""
    neighbors_length(adj::AdjDict, i) -> Int

Returns the number of neighbors stored for node `i` in `adj`, or `0` if node `i` has no entry.
"""
Base.@propagate_inbounds @inline function neighbors_length(adj::AdjDict, i)
    lock(adj.glock) do
        L = get(adj.end_point, i, nothing)
        L === nothing ? 0 : length(L)
    end
end

"""
    add!(adj::AdjDict{K,T}, n, N) where {K,T} -> AdjDict

Adds the neighbors in `N` (an iterable of ids convertible to `T`) to key `n`'s neighbor list. If
key `n` already has an entry, `N` is appended to it; otherwise a new list is created from `N`.
Thread-safe via `adj.glock`.
"""
Base.@propagate_inbounds @inline function add!(adj::AdjDict{K,T}, n, N) where {K,T}
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
    add!(adj::AdjDict{K,T}, iter) where {K,T} -> AdjDict

Bulk version of `add!`: `iter` yields `(n, N)` pairs, each adding neighbor set `N` to key `n`.
Thread-safe via `adj.glock`.
"""
Base.@propagate_inbounds @inline function add!(adj::AdjDict{K,T}, iter) where {K,T}
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

