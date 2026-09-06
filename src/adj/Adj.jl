# This file is a part of SimilaritySearch.jl
#module Adj

"""
    abstract type AbstractAdjList{T} end

Base type for adjacency-list backends used to store the neighbors of each node in a graph-based
index. The type parameter `T` is the element type stored per neighbor (e.g., an integer id, or an
`IdDist` pair combining an id with a distance).

Concrete subtypes provide different storage strategies with the same read/write API
(`neighbors`, `neighbors_length`, `add!`):

- [`AdjList`](@ref): growable `Vector{Vector{T}}`-backed adjacency list, indexed by contiguous
  integer node ids.
- [`AdjDict`](@ref): `Dict{T,Vector{T}}`-backed adjacency list, useful when node ids are sparse
  or non-contiguous.
- [`StaticAdjList`](@ref): frozen, CSR-like layout for fast read-only access once the graph stops
  growing.
"""
abstract type AbstractAdjList{T} end

export AbstractAdjList

include("adjlist.jl")
include("adjstatic.jl")
include("adjdict.jl")

import SparseArrays: sparse

"""
    sparse(idx::AbstractAdjList, val=1f0)

Creates an sparse matrix (from SparseArrays) from `idx` using `val` as value.

```
   I  
   ↓    1 2 3 4 5 … n  ← J
 L[1] = 0 1 0 0 1 … 0
 L[2] = 1 0 0 1 0 … 1
 L[3] = 1 0 1 0 0 … 1
 ⋮
 L[m] = 0 0 1 1 0 … 0
```
"""
function sparse(adj::AbstractAdjList{T}, val::AbstractFloat=1f0) where {T<:Integer}
    n = length(adj)
    I = T[]
    J = T[]
    F = eltype(val)[]
    sizehint!(I, n)
    sizehint!(J, n)
    sizehint!(F, n)

    for i in eachindex(adj)
        L = neighbors(adj, i)
        for j in L
            push!(I, i)
            push!(J, j)
            push!(F, val)
        end
    end

    sparse(I, J, F, length(adj), n)
end

"""
    sparse(idx::AbstractAdjList{IdDist})

Creates a sparse matrix (from SparseArrays) from `idx`, an adjacency list whose entries carry
both a neighbor id and a distance (`IdDist`). The distance stored in each entry
becomes the corresponding value in the sparse matrix (unlike the `val`-based `sparse` method
above, which fills a constant value).
"""
sparse(adj::AbstractAdjList{IdDist}) = sparse_from_adj(adj, Int32, Float32)

# Internal helper (not exported) used by the `sparse(::AbstractAdjList{IdDist})` method above
# to build the `I`, `J`, `F` triplet passed to `SparseArrays.sparse`.
function sparse_from_adj(adj::AbstractAdjList, IType, FType)
    n = length(adj)
    I = IType[]
    J = IType[]
    F = FType[]
    sizehint!(I, n)
    sizehint!(J, n)
    sizehint!(F, n)

    for i in eachindex(adj)
        L = neighbors(adj, i)

        for s in L
            push!(I, i)
            push!(J, s.id)
            push!(F, s.dist)
        end
    end

    sparse(I, J, F, length(adj), n)
end

#end