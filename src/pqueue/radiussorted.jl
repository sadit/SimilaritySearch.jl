# This file is a part of SimilaritySearch.jl

"""
    RadiusSorted <: AbstractRadiusQueue

A radius-bounded result container that keeps its items always sorted by distance (ascending),
using the same bounded binary-search + block-shift insertion as [`KnnSorted`](@ref)
([`sort_last_item!`](@ref)). Unlike `KnnSorted`, it has no count-based capacity: it accepts every
`(id, dist)` pair with `dist <= radius`, growing its backing `Vector`s via `push!` as needed.

# Fields
- `ids::Vector{UInt32}`: backing storage for the identifiers.
- `dists::Vector{Float32}`: backing storage for the distances, parallel to `ids`.
- `radius::Float32`: the fixed acceptance threshold.

**Invariant**: `ids`/`dists` are always sorted in ascending order by distance.

# Examples

```julia
res = RadiusSorted(0.3f0)
push_item!(res, 1, 0.1f0)
push_item!(res, 2, 0.5f0)  # rejected, dist > radius
nearest(res)     # closest item
IdDistView(res)  # lazy view of the active items, sorted by distance
```
"""
mutable struct RadiusSorted <: AbstractRadiusQueue
    ids::Vector{UInt32}
    dists::Vector{Float32}
    radius::Float32
end

"""
    RadiusSorted(radius::Real)

Creates an empty `RadiusSorted` accepting items with `dist <= radius`.
"""
RadiusSorted(radius::Real) = RadiusSorted(UInt32[], Float32[], Float32(radius))

"""
    push_item!(res::RadiusSorted, p::IdDist)

Accepts `p` into `res` iff `p.dist <= res.radius`, keeping `res` sorted by distance. Returns
whether the item was accepted.
"""
@inline function push_item!(res::RadiusSorted, item::IdDist)
    item.dist > res.radius && return false
    push!(res.ids, item.id)
    push!(res.dists, item.dist)
    sort_last_item!(res.ids, res.dists, 1, length(res.ids))
    true
end

"Closest item ([`IdDist`](@ref)) currently stored in `res`."
@inline nearest(res::RadiusSorted) = @inbounds IdDist(res.ids[1], res.dists[1])

"Farthest item ([`IdDist`](@ref)) currently stored in `res`."
@inline frontier(res::RadiusSorted) = @inbounds IdDist(res.ids[end], res.dists[end])

"For `RadiusSorted` items are always sorted; returns the `IdDistView` view immediately."
@inline sortitems!(res::RadiusSorted) = IdDistView(res)

"""
    reuse!(res::RadiusSorted, radius::Real=res.radius)

Resets `res` to a fresh, empty state with acceptance threshold `radius`, truncating its backing
storage (unlike `KnnSorted.reuse!`, which keeps a fixed-size buffer, `RadiusSorted` must actually
free the grown storage to avoid leaking memory from previously-reused, larger result sets).
"""
function reuse!(res::RadiusSorted, radius::Real=res.radius)
    empty!(res.ids)
    empty!(res.dists)
    res.radius = Float32(radius)
    res
end
