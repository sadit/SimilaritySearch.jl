# This file is a part of SimilaritySearch.jl

export IdDist, IdIntDist, IdOrder, DistOrder, RevDistOrder

using Base.Order
import Base.Order: lt

"""
    IdDist(id, dist)

A lightweight pair `(id::UInt32, dist::Float32)` representing a single search result:
the identifier of an object together with its distance to the query. It is the basic
item type stored in [`KnnHeap`](@ref) and [`KnnSorted`](@ref) result containers.

# Examples

```julia
item = IdDist(3, 0.25f0)
item.id    # 3
item.dist  # 0.25f0
```
"""
struct IdDist
    id::UInt32
    dist::Float32
end


"""
    IdIntDist(id, dist)

Stores a pair of objects to be accessed. Similar to [`IdDist`](@ref) but it stores an integer dist

# Examples

```julia
item = IdIntDist(3, 5)
item.id    # 3
item.dist  # 5
```
"""
struct IdIntDist
    id::UInt32
    dist::Int32
end

Base.zero(::Type{IdDist}) = IdDist(zero(UInt32), zero(Float32))
Base.zero(::Type{IdIntDist}) = IdDist(zero(UInt32), zero(Int32))

struct IdOrderingType <: Ordering end
struct DistOrderingType <: Ordering end
struct RevDistOrderingType <: Ordering end

"""
    IdOrder

Singleton `Ordering` (from `Base.Order`) that compares items by `id` in ascending order.
Pass it to the heap/sort routines when the desired order is by object identifier instead
of by distance.
"""
const IdOrder = IdOrderingType()

"""
    DistOrder

Singleton `Ordering` (from `Base.Order`) that compares items by `dist` in ascending order
(nearest first). This is the ordering used internally by [`KnnHeap`](@ref) and
[`KnnSorted`](@ref) to keep the closest neighbors found so far.
"""
const DistOrder = DistOrderingType()

"""
    RevDistOrder

Singleton `Ordering` (from `Base.Order`) that compares items by `dist` in descending order
(farthest first).
"""
const RevDistOrder = RevDistOrderingType()

@inline lt(::IdOrderingType, a, b) = a.id < b.id
@inline lt(::DistOrderingType, a, b) = a.dist < b.dist
@inline lt(::RevDistOrderingType, a, b) = b.dist < a.dist
@inline lt(::IdOrderingType, a::Number, b::Number) = a < b
@inline lt(::DistOrderingType, a::Number, b::Number) = a < b
@inline lt(::RevDistOrderingType, a::Number, b::Number) = b < a
