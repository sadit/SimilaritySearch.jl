# This file is a part of SimilaritySearch.jl

export CenterSelection

"""
    CenterSelection(centers, assign, assigndist, covering, separation, costdists, costblocks)

What every center-selection algorithm in `KCenters` returns: [`fft`](@ref), [`dnet`](@ref),
[`randsel`](@ref) and [`multirandsel`](@ref) all produce this same type, so they are
interchangeable at the call site.

# Fields
- `centers::Vector{UInt32}` -- the selected centers, as **identifiers into `X`**, without
  repetitions.
- `assign::Vector{UInt32}` -- one entry per object of `X`, in `X` order. `assign[i]` is the
  **position in `centers`** (a value in `1:length(centers)`) of the center object `i` was
  assigned to, *not* its identifier in `X`. The identifier is one indexing away:
  `centers[assign[i]]`. For [`fft`](@ref), [`randsel`](@ref) and [`multirandsel`](@ref) that
  center is the **nearest** one; [`dnet`](@ref) is the documented exception -- see below.
- `assigndist::Vector{Float32}` -- `assigndist[i]` is the distance from object `i` to the
  center it was assigned to, in `X` order. Always consistent with `assign`: it is the distance
  to the center `assign[i]` names, whether or not that center is the nearest.
- `covering::Float32` -- the largest `assigndist`: the radius the selected centers need in
  order to reach every object of `X` *under this assignment*. Smaller is a better cover. For
  the three nearest-center selectors this is the covering radius exactly; for `dnet` it is an
  upper bound on it, since a nearest-center assignment could only be tighter.
- `separation::Float32` -- the smallest distance between two selected centers. Larger is a
  more spread-out selection. `typemax(Float32)` when fewer than two centers exist, since
  there is then no pair to measure.
- `costdists::Int` / `costblocks::Int` -- distance and block evaluations performed by the
  call, including the ones spent computing `separation`.

# Why `assign` holds positions

Because the identifier is recoverable from the position in one indexing operation and the
position is not recoverable from the identifier without building a dictionary. Both consumers
of this result inside the library used to receive identifiers and immediately rebuild that
dictionary; producers computed the position and threw it away. The data now stops making the
round trip.

# `dnet` assigns to the center that took the object, not the nearest one

[`dnet`](@ref) carves balls out of a shrinking pool, so an object leaves with the ball that
absorbed it -- and a center chosen in a later round can turn out to be closer. It is not a
small effect: on 120 random points in 4 dimensions with `k=10`, 40% of the objects had a
nearer center than the one they are assigned to.

This is kept deliberately. The assignment *is* the ball structure `dnet` computed, and
recovering it costs nothing; turning it into a nearest-center assignment would mean another
`length(X) * k` distance evaluations for a quantity the caller can compute itself if that is
what they wanted. So `assign` always answers "which center is this object filed under", and
only the other three additionally promise "and it is the closest one".

# `covering` and `separation` are different numbers

They used to share the name `ε`, meaning the covering radius in `fft` and the separation in
`multirandsel`, while `fft`'s docstring described its own as the separation. They are not
interchangeable: on 300 random points in 4 dimensions with `k=8`, `fft` gives
`covering=0.715` and `separation=0.908`. Both are now always computed, by every algorithm.

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 4, 10^3))
R = fft(Dist.L2(), X, 16; verbose=false)

R.centers                       # 16 identifiers into X
R.centers[R.assign[7]]          # the identifier of the center object 7 belongs to
R.assigndist[7]                 # how far object 7 is from it
R.covering, R.separation        # how well the 16 cover X, and how spread out they are
count(==(3), R.assign)          # how many objects the third center took
```
"""
struct CenterSelection
    centers::Vector{UInt32}
    assign::Vector{UInt32}
    assigndist::Vector{Float32}
    covering::Float32
    separation::Float32
    costdists::Int
    costblocks::Int

    function CenterSelection(centers, assign, assigndist, covering, separation, costdists, costblocks)
        length(assign) == length(assigndist) ||
            throw(ArgumentError("assign and assigndist must be parallel, got $(length(assign)) and $(length(assigndist))"))
        allunique(centers) || throw(ArgumentError("centers must not repeat"))
        k = length(centers)
        for a in assign
            1 <= a <= k ||
                throw(ArgumentError("assign holds positions into centers (1:$k), got $a"))
        end

        new(convert(Vector{UInt32}, centers),
            convert(Vector{UInt32}, assign),
            convert(Vector{Float32}, assigndist),
            convert(Float32, covering),
            convert(Float32, separation),
            convert(Int, costdists),
            convert(Int, costblocks))
    end
end

function Base.show(io::IO, R::CenterSelection)
    print(io, "CenterSelection(|centers|=", length(R.centers),
          ", |X|=", length(R.assign),
          ", covering=", R.covering,
          ", separation=", R.separation,
          ", costdists=", R.costdists, ")")
end

"""
    empty_selection() -> CenterSelection

The result for an empty database: no centers, nothing assigned, and both radii at
`typemax(Float32)` since neither is defined without objects.
"""
empty_selection() = CenterSelection(UInt32[], UInt32[], Float32[],
                                    typemax(Float32), typemax(Float32), 0, 0)

"""
    center_separation(dist::SemiMetric, X::AbstractDatabase, centers; scheduler=get_batch_scheduler()) -> (separation, npairs)

The smallest distance between two of `centers` (identifiers into `X`), and the number of
distance evaluations it took (`k(k-1)/2`). `typemax(Float32)` when fewer than two centers
exist.

Every algorithm here already spends `length(X) * k` evaluations assigning objects to centers,
so this extra `k(k-1)/2` is smaller by a factor of `k/(2*length(X))` -- always well under one.
It is reported in `costdists` rather than hidden.
"""
function center_separation(dist::SemiMetric, X::AbstractDatabase, centers; scheduler::Symbol=get_batch_scheduler())
    k = length(centers)
    k < 2 && return typemax(Float32), 0

    best = fill(typemax(Float32), k)
    minbatch = getminbatch(k)
    # each i writes only its own slot, so the batches are disjoint by construction and no
    # reduction across them is needed
    @BATCHES minbatch scheduler=scheduler for i in 1:k-1
        u = X[centers[i]]
        m = typemax(Float32)
        for j in i+1:k
            d = convert(Float32, evaluate(dist, u, X[centers[j]]))
            d < m && (m = d)
        end

        best[i] = m
    end

    minimum(best), (k * (k - 1)) ÷ 2
end
