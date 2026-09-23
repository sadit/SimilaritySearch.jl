# This file is a part of SimilaritySearch.jl

"""
    BallKnn <: AbstractKnnQueue

Internal navigation queue for radius-bounded searches over a graph index (issue #67). It keeps
**every item within `radius`**, growing without a count limit like an
[`AbstractRadiusQueue`](@ref), **plus a reserve of at least `kmin` nearest items even when those
fall outside the ball**.

That reserve is what makes a graph search work at all. A greedy/beam search needs two things from
its result container: something to start from ([`nearest`](@ref)) and a threshold that *shrinks*
as the search improves ([`maximum`](@ref), the k-th distance). A plain radius container gives
neither when the query's neighborhood starts outside the ball: it rejects every candidate, stays
empty -- so `nearest` reads out of bounds, which is the segfault of #67 -- and its `maximum` is
the constant `radius`, turning the admission test into an absolute one, so a beam that starts
outside the ball has no admissible child and dies on its first expansion. That second half is why
guarding the emptiness alone is not a fix: it converts the crash into silently empty balls.

Here `maximum` returns `max(radius, dists[capacity])`: the k-th distance while the reserve is what
bounds the queue, flattening at `radius` once the ball itself holds more than `kmin` items. So it
shrinks exactly as a k-NN queue's does, and stops shrinking where further tightening would start
excluding ball members.

`maxlength` reports that same `capacity` rather than `typemax(Int32)`, which is what lets
`create_error_function`'s coverage block (`length(r) == maxlength(r)`) accept these queues instead
of rejecting every configuration with `InvalidSetupError` -- see the plan for radius-aware
`optimize_index!` in #67.

The reserve is an implementation detail of navigation, never a result: callers copy out
[`ballview`](@ref) (the prefix within `radius`), so nothing outside the ball is ever returned.

# Fields
- `ids::Vector{UInt32}` / `dists::Vector{Float32}`: parallel storage, kept sorted by distance.
- `radius::Float32`: the ball being answered.
- `kmin::Int32`: minimum reserve kept for navigation.
"""
mutable struct BallKnn <: AbstractKnnQueue
    ids::Vector{UInt32}
    dists::Vector{Float32}
    radius::Float32
    kmin::Int32
end

"""
    BallKnn(radius::Real, kmin::Integer)

Creates an empty `BallKnn` answering the ball of `radius` while keeping a navigation reserve of
at least `kmin` items.
"""
function BallKnn(radius::Real, kmin::Integer)
    kmin > 0 || throw(ArgumentError("BallKnn: kmin=$kmin must be positive (a queue that can be empty is what #67 crashes on)"))
    BallKnn(UInt32[], Float32[], Float32(radius), Int32(kmin))
end

@inline Base.length(res::BallKnn) = length(res.ids)

"Number of items currently within `res.radius` (a prefix, since the queue is sorted by distance)."
@inline ninside(res::BallKnn) = searchsortedlast(res.dists, res.radius)

"How many items `res` keeps: the ball itself, or the navigation reserve while the ball is smaller."
@inline capacity(res::BallKnn) = max(Int(res.kmin), ninside(res))

@inline maxlength(res::BallKnn) = capacity(res)

"Items within `res.radius`, sorted by distance -- the actual result, without the navigation reserve."
@inline ballview(res::BallKnn) = IdDistView(res.ids, res.dists, 1, ninside(res))

@inline function Base.maximum(res::BallKnn)::Float32
    length(res) < res.kmin && return typemax(Float32)
    @inbounds max(res.radius, res.dists[capacity(res)])
end

"Closest item ([`IdDist`](@ref)) currently stored in `res`."
@inline nearest(res::BallKnn) = @inbounds IdDist(res.ids[1], res.dists[1])

"Farthest item ([`IdDist`](@ref)) currently stored in `res`, reserve included."
@inline frontier(res::BallKnn) = @inbounds IdDist(res.ids[end], res.dists[end])

"""
    push_item!(res::BallKnn, item::IdDist)

Accepts `item` if it falls within the ball, or if it improves the navigation reserve; returns
whether it was accepted.
"""
@inline function push_item!(res::BallKnn, item::IdDist)
    d = item.dist
    if d > res.radius && length(res) >= res.kmin && @inbounds(d >= res.dists[capacity(res)])
        return false     # neither a ball member nor an improvement of the reserve
    end

    push!(res.ids, item.id)
    push!(res.dists, d)
    sort_last_item!(res.ids, res.dists, 1, length(res.ids))

    # one push can exceed the capacity by at most one item, and `capacity` never shrinks on a
    # push, so this loop runs at most once; it is a `while` to keep that from being a silent
    # invariant of the arithmetic above.
    while length(res.ids) > capacity(res)
        pop!(res.ids)
        pop!(res.dists)
    end

    true
end

@inline push_item!(res::BallKnn, i::Integer, d::Real) = push_item!(res, IdDist(convert(UInt32, i), convert(Float32, d)))
@inline push_item!(res::BallKnn, p::Pair) = push_item!(res, IdDist(convert(UInt32, p.first), convert(Float32, p.second)))

"For `BallKnn` items are always sorted; returns the `IdDistView` view immediately."
@inline sortitems!(res::BallKnn) = IdDistView(res)

"""
    reuse!(res::BallKnn, radius::Real=res.radius)

Resets `res` to a fresh, empty state answering `radius`, truncating its backing storage (it grows
with the ball, like `RadiusSorted`).
"""
function reuse!(res::BallKnn, radius::Real=res.radius)
    empty!(res.ids)
    empty!(res.dists)
    res.radius = Float32(radius)
    res
end
