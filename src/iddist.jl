# This file is a part of SimilaritySearch.jl

export IdDist, IdIntDist, IdOrder, DistOrder, RevDistOrder

using Base.Order
import Base.Order: lt

"""
    IdDist(id, dist)

Stores a pair of objects to be accessed. It is used in several places but mostly as an item in `KnnResult` algorithms where `dist` field is a distance instead of a dist
    
"""
struct IdDist
    id::UInt32
    dist::Float32
end


"""
    IdIntDist(id, dist)

Stores a pair of objects to be accessed. Similar to [`IdDist`](@ref) but it stores an integer dist 
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
const IdOrder = IdOrderingType()
const DistOrder = DistOrderingType()
const RevDistOrder = RevDistOrderingType()

@inline lt(::IdOrderingType, a, b) = a.id < b.id
@inline lt(::DistOrderingType, a, b) = a.dist < b.dist
@inline lt(::RevDistOrderingType, a, b) = b.dist < a.dist
@inline lt(::IdOrderingType, a::Number, b::Number) = a < b
@inline lt(::DistOrderingType, a::Number, b::Number) = a < b
@inline lt(::RevDistOrderingType, a::Number, b::Number) = b < a
