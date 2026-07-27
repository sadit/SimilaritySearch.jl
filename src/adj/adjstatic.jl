# This file is a part of SimilaritySearch.jl

export StaticAdjList

"""
    struct StaticAdjList{T} <: AbstractAdjList{T}

Frozen, read-only adjacency-list representation of a graph, using a CSR-like (compressed sparse
row) encoding for compactness and fast access. It is typically built once from a growable
[`AdjList`](@ref) or [`AdjDict`](@ref) after the graph stops growing (see the conversion
constructor `StaticAdjList(adj::AbstractAdjList)`).

# Fields
- `offset`: cumulative neighbor counts; `offset[i]` is the index (in `end_point`) of the last
  neighbor of node `i`, so node `i`'s neighbors occupy `end_point[offset[i-1]+1:offset[i]]`
  (with `offset[0]` implicitly `0`).
- `end_point`: flat vector holding all neighbor ids, concatenated node by node.

# Examples
```julia
adj = AdjList(Int32, 0)
add!(adj, [(1, Int32[2, 3]), (2, Int32[1])])
sadj = StaticAdjList(adj)  # freeze into a compact, read-only representation
neighbors(sadj, 1)         # => view of Int32[2, 3]
```
"""
struct StaticAdjList{T} <: AbstractAdjList{T}
    offset::Vector{Int64}
    end_point::Vector{T}
end

Base.length(adj::StaticAdjList) = length(adj.offset)
Base.eltype(adj::StaticAdjList{T}) where T = Pair{T,typeof(view(adj.end_point, 1:1))}
Base.eachindex(adj::StaticAdjList) = eachindex(adj.offset)

function Base.iterate(adj::StaticAdjList{T}, i=1) where T
    i = T(i)
    n = length(adj)
    (n == 0 || i > n) && return nothing
    i => neighbors(adj, i), i+1
end

"""
    StaticAdjList(adj::StaticAdjList; offset=adj.offset, end_point=adj.end_point)

Creates a `StaticAdjList` reusing (by default) the `offset` and `end_point` vectors of `adj`, or
replacing either one via the corresponding keyword argument.
"""
function StaticAdjList(adj::StaticAdjList; offset=adj.offset, end_point=adj.end_point)
    StaticAdjList(offset, end_point)
end

"""
    StaticAdjList(adj::AbstractAdjList{T}) where T -> StaticAdjList{T}

Freezes a growable adjacency list `adj` (e.g., an [`AdjList`](@ref) or [`AdjDict`](@ref)) into a
compact, read-only `StaticAdjList`, by concatenating all neighbor lists into a single
`end_point` vector and recording per-node cumulative offsets. Intended to be called once a graph
has finished being built, to speed up subsequent read-only access.

# Examples
```julia
adj = AdjList(Int32, 0)
add!(adj, [(1, Int32[2, 3]), (2, Int32[1])])
sadj = StaticAdjList(adj)
neighbors(sadj, 2)  # => view of Int32[1]
```
"""
function StaticAdjList(adj::AbstractAdjList{T}) where T
    n = length(adj)
    @show n
    offset = Vector{Int64}(undef, n)
    end_point = let N = sum(length(N) for (_, N) in adj)
        Vector{T}(undef, N)
    end

    i = 1
    s = 0
    @inbounds @inbounds for (j, N) in adj
        s += length(N)
        offset[j] = s

        for l in N
            end_point[i] = l
            i += 1
        end
    end

    StaticAdjList{T}(offset, end_point)
end

"""
    neighbors(adj::StaticAdjList, i::Integer) -> AbstractVector

Returns a view onto the neighbors of node `i` in `adj`, computed from the CSR-like `offset`
encoding as `end_point[offset[i-1]+1:offset[i]]` (with an implicit `offset[0] == 0`).

# Examples
```julia
neighbors(sadj, 1)  # view of node 1's neighbor ids
```
"""
Base.@propagate_inbounds @inline function neighbors(adj::StaticAdjList, i::Integer)
    @inbounds sp::Int64 = i == 1 ? 1 : adj.offset[i-1] + 1
    @inbounds ep = adj.offset[i]
    view(adj.end_point, sp:ep)
end

"""
    neighbors_length(adj::StaticAdjList, i::Integer) -> Int

Returns the number of neighbors stored for node `i` in `adj`, computed from the `offset` encoding
(`offset[i] - offset[i-1]`, with an implicit `offset[0] == 0`).
"""
Base.@propagate_inbounds @inline function neighbors_length(adj::StaticAdjList, i::Integer)
    @inbounds if i == 1
        adj.offset[i]
    else
        adj.offset[i] - adj.offset[i-1]
    end
end

"""
    add!(adj::StaticAdjList, n, N)

Not supported: `StaticAdjList` is an immutable, frozen adjacency-list representation. Build the
graph with [`AdjList`](@ref) or [`AdjDict`](@ref) and convert it with `StaticAdjList(adj)` once
construction is complete.
"""
function add!(adj::StaticAdjList, n, N)
    error("ERROR: unsupported add! on a static adjacent list")
end

"""
    add!(adj::StaticAdjList, N)

Not supported: `StaticAdjList` is an immutable, frozen adjacency-list representation. Build the
graph with [`AdjList`](@ref) or [`AdjDict`](@ref) and convert it with `StaticAdjList(adj)` once
construction is complete.
"""
function add!(adj::StaticAdjList, N)
    error("ERROR: unsupported add! on a static adjacent list")
end
