# This file is a part of SimilaritySearch.jl
#module Adj

abstract type AbstractAdjList{T} end

export AbstractAdjList


include("adjlist.jl")
include("adjstatic.jl")
include("adjdict.jl")

import SparseArrays: sparse

"""
    sparse(idx::AbstractAdjList, val=1f0; ignore_reverse_edges::Bool=true)

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
function sparse(adj::AbstractAdjList{T}, ::Type{TT}=T, val::AbstractFloat=1f0; ignore_reverse_edges::Bool=true) where {T<:Integer,TT<:Integer}
    n = length(adj)
    I = TT[]
    J = TT[]
    F = eltype(val)[]
    sizehint!(I, n)
    sizehint!(J, n)
    sizehint!(F, n)

    for i in eachindex(adj)
        L = packed_neighbors(adj, i)
        L === nothing && continue
        for j in L
            j, isdirect = unpack_edge(j)
            ignore_reverse_edges && !isdirect && continue
            push!(I, i)
            push!(J, j)
            push!(F, val)
        end
    end

    sparse(I, J, F, length(adj), n)
end

#end