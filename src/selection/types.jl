# This file is a part of SimilaritySearch.jl

export AbstractSelection, CenterSelection, NearDupSelection

"""
    abstract type AbstractSelection end

What every algorithm in [`Selection`](@ref) returns. The two concrete types differ only in what
each one additionally reports; these five fields mean the same thing in both, so code that reads
one reads the other:

- `centers::Vector{UInt32}` -- the selected objects, as **identifiers into `X`**, without
  repetitions.
- `assign::Vector{UInt32}` -- one entry per object of `X`, in `X` order. `assign[i]` is the
  **position in `centers`** (a value in `1:length(centers)`) of the center object `i` was assigned
  to, *not* its identifier in `X`. The identifier is one indexing away: `centers[assign[i]]`.
- `assigndist::Vector{Float32}` -- the distance from object `i` to the center `assign[i]` names.
  Zero for a center itself.
- `costdists::Int` / `costblocks::Int` -- distance and block evaluations performed by the call.

# Why `assign` holds positions and not identifiers

Because the identifier is recoverable from the position in one indexing operation, and the position
is not recoverable from the identifier without building a dictionary. Every consumer of these
results inside the library used to receive identifiers and immediately rebuild that dictionary,
while the producers computed the position and threw it away.

# Which center an object is assigned to

`assigndist` always agrees with `assign` -- it is the distance to the center `assign` names. Whether
that center is the *nearest* one depends on how the algorithm works, and the split is clean:

- **Single-pass selectors know all their centers before assigning**, so they assign to the nearest:
  [`fft`](@ref), [`randsel`](@ref), [`multirandsel`](@ref).
- **Incremental selectors assign an object when they meet it**, to whichever center claimed it at
  that moment -- and a center created later can turn out to be closer: [`dnet`](@ref) (around 40%
  of objects, measured on 120 random points in 4 dimensions) and [`neardup`](@ref) (8.8% on 400
  points at `ϵ=0.25`). That assignment *is* the structure those algorithms compute; recomputing it
  as a nearest-center assignment would cost another full pass over the data for something the
  caller can do itself.

The practical consequence is that `covering`, being the largest `assigndist`, is the exact covering
radius for the single-pass selectors and an upper bound on it for the incremental ones.
"""
abstract type AbstractSelection end

function check_selection(centers, assign, assigndist)
    length(assign) == length(assigndist) ||
        throw(ArgumentError("assign and assigndist must be parallel, got $(length(assign)) and $(length(assigndist))"))
    allunique(centers) || throw(ArgumentError("centers must not repeat"))
    k = length(centers)
    for a in assign
        1 <= a <= k ||
            throw(ArgumentError("assign holds positions into centers (1:$k), got $a"))
    end

    nothing
end

"""
    CenterSelection(centers, assign, assigndist, covering, separation, costdists, costblocks)

What the fixed-count selectors return: [`fft`](@ref), [`dnet`](@ref), [`randsel`](@ref) and
[`multirandsel`](@ref) all produce this same type, so they are interchangeable at the call site.
See [`AbstractSelection`](@ref) for the fields they share with [`neardup`](@ref)'s result.

# Fields beyond the shared ones
- `covering::Float32` -- the largest `assigndist`: the radius the selected centers need in order to
  reach every object of `X`. Smaller is a better cover.
- `separation::Float32` -- the smallest distance between two selected centers. Larger is a more
  spread-out selection. `typemax(Float32)` when fewer than two centers exist, since there is then
  no pair to measure.

`costdists` includes the `k(k-1)/2` evaluations spent measuring `separation`.

# `covering` and `separation` are different numbers

They used to share the name `ε`, meaning the covering radius in `fft` and the separation in
`multirandsel`, while `fft`'s docstring described its own as the separation. They are not
interchangeable: on 300 random points in 4 dimensions with `k=8`, `fft` gives `covering=0.715` and
`separation=0.908`. Both are now always computed, by all four.

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
struct CenterSelection <: AbstractSelection
    centers::Vector{UInt32}
    assign::Vector{UInt32}
    assigndist::Vector{Float32}
    covering::Float32
    separation::Float32
    costdists::Int
    costblocks::Int

    function CenterSelection(centers, assign, assigndist, covering, separation, costdists, costblocks)
        check_selection(centers, assign, assigndist)
        new(convert(Vector{UInt32}, centers),
            convert(Vector{UInt32}, assign),
            convert(Vector{Float32}, assigndist),
            convert(Float32, covering),
            convert(Float32, separation),
            convert(Int, costdists),
            convert(Int, costblocks))
    end
end

"""
    NearDupSelection(idx, centers, assign, assigndist, covering, epsilon, costdists, costblocks)

What [`neardup`](@ref) returns: the ``ϵ``-net it found, i.e. the objects that survived as
non-duplicates and, for every object of `X`, which survivor covers it. See
[`AbstractSelection`](@ref) for the fields it shares with the fixed-count selectors.

# Fields beyond the shared ones
- `idx` -- the index holding the centers, ready to be reused: `search`ing it answers with positions
  into `centers`, and it is what makes `neardup` usable as a deduplicating index builder rather
  than just a report.
- `epsilon::Float32` -- the radius that defined "too close", so the result is self-describing:
  `all(assigndist .<= epsilon)` holds without the caller keeping `ϵ` around.
- `covering::Float32` -- the largest `assigndist`, the radius actually needed. Always `<= epsilon`,
  and how far below says how much of the budget the data used: on 400 random points at `ϵ=0.25` it
  came out at 0.2493.

# There is no `separation` here, on purpose

Being an ``ϵ``-net *is* a separation guarantee: an object only becomes a center when every existing
center is farther than `ϵ`, so no two centers are closer than that. Reporting it would echo the
input back (measured: 0.2501 for `ϵ=0.25`). It would also be the one expensive field in this type
-- unlike the fixed-count selectors, `neardup` does not know how many centers it will end up with,
and that count can approach `length(X)`.

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 4, 10^3))
R = neardup(Dist.L2(), X, 0.1)

R.centers                    # the ϵ-net: identifiers into X that survived
R.centers[R.assign[7]]       # which survivor covers object 7
R.assigndist[7] <= R.epsilon # always true
R.covering                   # the radius the net actually needed, <= epsilon
length(R.centers)            # how many survived -- the output here, not the input
```
"""
struct NearDupSelection{IndexType} <: AbstractSelection
    idx::IndexType
    centers::Vector{UInt32}
    assign::Vector{UInt32}
    assigndist::Vector{Float32}
    covering::Float32
    epsilon::Float32
    costdists::Int
    costblocks::Int

    function NearDupSelection(idx::IndexType, centers, assign, assigndist, covering, epsilon, costdists, costblocks) where IndexType
        check_selection(centers, assign, assigndist)
        new{IndexType}(idx,
            convert(Vector{UInt32}, centers),
            convert(Vector{UInt32}, assign),
            convert(Vector{Float32}, assigndist),
            convert(Float32, covering),
            convert(Float32, epsilon),
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

function Base.show(io::IO, R::NearDupSelection)
    print(io, "NearDupSelection(|centers|=", length(R.centers),
          ", |X|=", length(R.assign),
          ", epsilon=", R.epsilon,
          ", covering=", R.covering,
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

Every fixed-count algorithm here already spends `length(X) * k` evaluations assigning objects to
centers, so this extra `k(k-1)/2` is smaller by a factor of `k/(2*length(X))` -- always well under
one. It is reported in `costdists` rather than hidden. [`neardup`](@ref) does not call it; see
[`NearDupSelection`](@ref) for why.
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
