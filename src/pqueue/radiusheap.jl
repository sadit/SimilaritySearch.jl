# This file is a part of SimilaritySearch.jl

"""
    RadiusHeap <: AbstractRadiusQueue

A radius-bounded result container that trades `RadiusSorted`'s "always sorted" invariant for a
cheap `O(1)` insertion: every accepted item is simply appended, with no ordering maintained on
each push (there is nothing to evict, so keeping a heap invariant up to date on every insert would
buy nothing). Items are only sorted lazily, once, the first time they're read after a push, via
[`heapify!`](@ref)/[`heapsort!`](@ref) (the same primitives [`KnnHeap`](@ref) uses).

# Fields
- `ids::Vector{UInt32}`: backing storage for the identifiers.
- `dists::Vector{Float32}`: backing storage for the distances, parallel to `ids`.
- `radius::Float32`: the fixed acceptance threshold.
- `sorted::Bool`: whether `ids`/`dists` are currently known to be sorted (invalidated by every
  `push_item!`, restored by [`sortitems!`](@ref)).

# Examples

```julia
res = RadiusHeap(0.3f0)
push_item!(res, 1, 0.1f0)
push_item!(res, 2, 0.5f0)  # rejected, dist > radius
nearest(res)     # forces a sort, then returns the closest item
```
"""
mutable struct RadiusHeap <: AbstractRadiusQueue
    ids::Vector{UInt32}
    dists::Vector{Float32}
    radius::Float32
    sorted::Bool
end

"""
    RadiusHeap(radius::Real)

Creates an empty `RadiusHeap` accepting items with `dist <= radius`.
"""
RadiusHeap(radius::Real) = RadiusHeap(UInt32[], Float32[], Float32(radius), true)

"""
    push_item!(res::RadiusHeap, p::IdDist)

Accepts `p` into `res` iff `p.dist <= res.radius`, appending it without maintaining any order
(marks `res` as unsorted). Returns whether the item was accepted.
"""
@inline function push_item!(res::RadiusHeap, item::IdDist)
    item.dist > res.radius && return false
    push!(res.ids, item.id)
    push!(res.dists, item.dist)
    res.sorted = false
    true
end

"""
    sortitems!(res::RadiusHeap)

Sorts `res`'s items by distance (ascending) if they aren't already known to be sorted, and
returns the resulting `IdDistView` view.
"""
function sortitems!(res::RadiusHeap)
    if !res.sorted
        n = length(res.ids)
        if n > 1
            X = (res.ids, res.dists)
            heapify!(_lt_dist, _swap_ids_dists, X, n)
            heapsort!(_lt_dist, _swap_ids_dists, X, n)
        end
        res.sorted = true
    end
    IdDistView(res.ids, res.dists, 1, length(res.ids))
end

"Closest item ([`IdDist`](@ref)) currently stored in `res`, sorting `res` first if needed."
@inline function nearest(res::RadiusHeap)
    sortitems!(res)
    @inbounds IdDist(res.ids[1], res.dists[1])
end

"Farthest item ([`IdDist`](@ref)) currently stored in `res`, sorting `res` first if needed."
@inline function frontier(res::RadiusHeap)
    sortitems!(res)
    @inbounds IdDist(res.ids[end], res.dists[end])
end

"""
    reuse!(res::RadiusHeap, radius::Real=res.radius)

Resets `res` to a fresh, empty state with acceptance threshold `radius`, truncating its backing
storage to avoid leaking memory from previously-reused, larger result sets.
"""
function reuse!(res::RadiusHeap, radius::Real=res.radius)
    empty!(res.ids)
    empty!(res.dists)
    res.radius = Float32(radius)
    res.sorted = true
    res
end
