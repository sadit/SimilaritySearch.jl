# This file is a part of SimilaritySearch.jl

"""
    struct SubDatabase{DBType<:AbstractDatabase,RType} <: AbstractDatabase

A lightweight, read-only view over a subset (or a reordering, or a resampling) of a `parent` database,
without copying its objects. The `i`-th element of the view is `parent[map[i]]`. It is what `view(db, map)`,
`db[list]`, and `rand(db, n)` return for any `AbstractDatabase` `db`.

# Fields
- `parent`: the underlying database being viewed
- `map`: a collection of indices into `parent`; `map[i]` gives the parent index of the `i`-th element of the view

Please see [`AbstractDatabase`](@ref) for general usage.

# Examples

```julia
db = MatrixDatabase(rand(Float32, 8, 100))
sub = view(db, [1, 3, 5])   # a SubDatabase with 3 objects
sub[1] == db[1]             # true
sub2 = db[[2, 4]]           # getindex with a list of indices also returns a SubDatabase
```
"""
struct SubDatabase{DBType<:AbstractDatabase,RType} <: AbstractDatabase
    parent::DBType
    map::RType
end

function show(io::IO, db::SubDatabase; prefix="", indent="  ")
    println(io, prefix, "SubDatabase:")
    prefix = prefix * indent
    println(io, prefix, "eltype: ", eltype(db))
    println(io, prefix, "length: ", length(db))
    println(io, prefix, "parent-type: ", typeof(db.parent))
end

"""
    getindex(S::SubDatabase, i::Integer)

Retrieves the `i`-th object of the view, i.e., `S.parent[S.map[i]]`.
"""
@inline Base.getindex(S::SubDatabase, i::Integer) = @inbounds S.parent[S.map[i]]

"""
    length(S::SubDatabase)

Number of objects in the view (`length(S.map)`).
"""
@inline Base.length(S::SubDatabase) = length(S.map)

"""
    eachindex(S::SubDatabase)

An index iterator of the view (over its local indices, i.e., `eachindex(S.map)`).
"""
@inline Base.eachindex(S::SubDatabase) = eachindex(S.map)

"""
    push_item!(S::SubDatabase, v)

Not supported; `SubDatabase` is a read-only view over a `parent` database and cannot be mutated directly.
"""
@inline push_item!(S::SubDatabase, v) = error("push! unsupported operation on SubDatabase")

"""
    eltype(S::SubDatabase)

The element type of the view, forwarded from its `parent` database.
"""
@inline Base.eltype(S::SubDatabase) = eltype(S.parent)

"""
    rand(S::SubDatabase, n::Integer)

Retrieves `n` random elements from the view, returning a new `SubDatabase` over the same `parent`.
"""
@inline Random.rand(S::SubDatabase, n::Integer) = SubDatabase(S.parent, rand(S.map, n))
