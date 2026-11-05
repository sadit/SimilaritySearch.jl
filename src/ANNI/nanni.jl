# This file is a part of SimilaritySearch.jl

"""
    NANNI(dist::Dist.Metric, db::AbstractDatabase; checkmetric::Bool=true)
    NANNI(db::AbstractDatabase; dist::Dist.Metric, checkmetric::Bool=true)

The node auto-tuned nearest neighbor index (NANNI) of Tellez, Ruiz and Chávez, *Singleton
indexes for nearest neighbor search*, Information Systems 60 (2016) 50--68: the Dirichlet
partition of an [`ANNI`](@ref) stored by cluster, an **exact** index for metrics. Every center
keeps its cell (the objects nearer to it than to any other center), the covering radius of the
cell, and each member's distance to the center.

A search evaluates every center, then visits the cells by increasing distance from the query
to their centers. A cell is skipped when the query ball misses its covering ball,
`d(q, c) > cov(c) + r`, and the search stops at the first cell beyond the hyperplane rule,
`d(q, c) > d(q, c*) + 2r` with `c*` the center nearest to `q` (every member `x` of the cell of
`c` has `d(x, c) <= d(x, c*)`; the rule holds for every later cell too, since the cells come in
increasing `d(q, c)` and `r` only shrinks). Inside a cell the members are sorted by their
distance to the center, so the ones the pivot filter `|d(q, c) - d(x, c)| <= r` keeps form a
contiguous run found by binary search. With no sequential scan of the database, NANNI is the
member of the family that scales best with the size of the database.

[`index!`](@ref index!(::NANNI, ::AbstractContext)) builds it as [`ANNI`](@ref) is built,
promoting random centers in steps, but measures NANNI's own search to decide when to stop.

# Arguments
- `dist`: the distance function, a metric
- `db`: the database to index

# Keyword Arguments
- `checkmetric`: whether a non-`Dist.Metric` distance is rejected (default `true`)

# Examples

```julia
using SimilaritySearch

db = MatrixDatabase(rand(Float32, 4, 10_000))
nanni = NANNI(Dist.L2(), db)
ctx = GenericContext()
index!(nanni, ctx)
search(nanni, ctx, rand(Float32, 4), knnqueue(KnnSorted, 5))
```
"""
struct NANNI{DistanceType<:Dist.PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
    centers::Vector{UInt32}    # database ids of the centers
    cov::Vector{Float32}       # per cell: its covering radius (0 for an empty cell)
    cellstart::Vector{Int32}   # per cell: 1-based offset into members/mdist; length = #centers + 1
    members::Vector{UInt32}    # the non-center objects, grouped by cell, ascending distance to its center
    mdist::Vector{Float32}     # each member's distance to its center, aligned with `members`
end

function NANNI(dist::Dist.PreMetric, db::AbstractDatabase; checkmetric::Bool=true)
    _checkmetric(NANNI, dist, checkmetric)
    NANNI(dist, db, UInt32[], Float32[], Int32[], UInt32[], Float32[])
end

NANNI(db::AbstractDatabase; dist::Dist.PreMetric, checkmetric::Bool=true) = NANNI(dist, db; checkmetric)

@inline database(nanni::NANNI) = nanni.db
@inline database(nanni::NANNI, i) = nanni.db[i]
@inline distance(nanni::NANNI) = nanni.dist
@inline Base.length(nanni::NANNI) = length(nanni.db)
getcontext(::NANNI; kwargs...) = GenericContext(; kwargs...)

"""
    _fromanni!(nanni::NANNI, anni::ANNI) -> nanni

Rewrites `nanni`'s cells from `anni`'s partition: a counting sort of the objects by center, then
each cell sorted by distance to its center.
"""
function _fromanni!(nanni::NANNI, anni::ANNI)
    m = length(anni.centers)
    n = length(anni.center)
    resize!(nanni.centers, m)
    nanni.centers .= anni.centers
    resize!(nanni.cov, m)
    fill!(nanni.cov, 0f0)
    resize!(nanni.cellstart, m + 1)
    fill!(nanni.cellstart, 0)
    @inbounds for u in 1:n
        a = anni.center[u]
        a == 0 && continue
        nanni.cellstart[a+1] += 1
        nanni.cov[a] = max(nanni.cov[a], anni.dcenter[u])
    end

    nanni.cellstart[1] = 1
    for a in 1:m
        nanni.cellstart[a+1] += nanni.cellstart[a]
    end

    nmembers = n - m
    resize!(nanni.members, nmembers)
    resize!(nanni.mdist, nmembers)
    pos = nanni.cellstart[1:m]
    @inbounds for u in 1:n
        a = anni.center[u]
        a == 0 && continue
        p = pos[a]
        nanni.members[p] = u
        nanni.mdist[p] = anni.dcenter[u]
        pos[a] = p + 1
    end

    Threads.@threads for a in 1:m
        lo, hi = nanni.cellstart[a], nanni.cellstart[a+1] - 1
        if hi > lo
            o = sortperm(view(nanni.mdist, lo:hi))
            nanni.members[lo:hi] = nanni.members[lo:hi][o]
            nanni.mdist[lo:hi] = nanni.mdist[lo:hi][o]
        end
    end

    nanni
end

"""
    _searchcount(nanni::NANNI, q, res, factor=1f0) -> evaluations

The search of [`NANNI`](@ref), returning the number of distances it evaluated. Every filter
tests against `factor` times the covering radius: the exact search for 1, the stretched
triangle inequality of [`PruningSearch`](@ref) below it.
"""
function _searchcount(nanni::NANNI, q, res::AbstractMetricQueue, factor::Float32=1f0)
    dist = distance(nanni)
    m = length(nanni.centers)
    m == 0 && return 0
    dcq = Vector{Float32}(undef, m)
    dmin = typemax(Float32)
    @inbounds for i in 1:m
        id = nanni.centers[i]
        d = Dist.evaluate(dist, q, database(nanni, id))
        dcq[i] = d
        dmin = min(dmin, d)
        push_item!(res, id, d)
    end

    cost = m
    @inbounds for i in sortperm(dcq)
        dc = dcq[i]
        r = factor * covradius(res)
        dc > dmin + 2r && break      # every later cell is farther: the hyperplane rule holds for all of them
        dc > nanni.cov[i] + r && continue
        lo, hi = nanni.cellstart[i], nanni.cellstart[i+1] - 1
        lo > hi && continue
        # the members with |dc - d(x, c)| <= r: a run of the cell, sorted by d(x, c)
        j = searchsortedfirst(view(nanni.mdist, lo:hi), dc - r) + lo - 1
        while j <= hi
            dx = nanni.mdist[j]
            dx > dc + r && break
            if dc - dx <= r   # r shrinks as the run is visited, so the lower end is checked again
                u = nanni.members[j]
                push_item!(res, u, Dist.evaluate(dist, q, database(nanni, u)))
                r = factor * covradius(res)
                cost += 1
            end
            j += 1
        end
    end

    cost
end

"""
    search(nanni::NANNI, ctx::AbstractContext, q, res::AbstractMetricQueue)

Solves `q` exactly (see [`NANNI`](@ref)): every center is evaluated, then the cells the
covering-radius and hyperplane rules cannot discard, and in each one the members the pivot
filter keeps.
"""
function search(nanni::NANNI, ctx::AbstractContext, q, res::AbstractMetricQueue)
    add_distance_evaluations!(ctx, _searchcount(nanni, q, res))
    res
end

"""
    index!(nanni::NANNI, ctx::AbstractContext; queries=nothing, k=1, nqueries=32,
           step=isqrt(length(nanni)) + 1, alphastop=0.001, maxcenters=length(nanni),
           rng=Random.default_rng())

Builds `nanni` (see [`NANNI`](@ref)): the construction of [`ANNI`](@ref) (see
[`index!`](@ref index!(::ANNI, ::AbstractContext)) for the keyword arguments), measuring NANNI's
own search after every step. Returns `nanni`. The index must be empty.
"""
function index!(nanni::NANNI, ctx::AbstractContext; queries=nothing, k::Integer=1, nqueries::Integer=32,
                step::Integer=isqrt(length(nanni)) + 1, alphastop::Real=0.001, maxcenters::Integer=length(nanni),
                rng::AbstractRNG=default_rng())
    isempty(nanni.centers) || throw(ArgumentError("the index is already built; NANNI is built once"))
    anni = ANNI(distance(nanni), database(nanni); checkmetric=false)
    Q, kq = _trainingqueries(database(nanni), queries, k, nqueries, rng)
    measure() = (_fromanni!(nanni, anni); _querycost(nanni, ctx, Q, kq))
    _autotune!(anni, ctx, measure; step, alphastop, maxcenters, rng)
    isempty(anni.centers) || _fromanni!(nanni, anni)
    nanni
end

"""
    nannipermutation(nanni::NANNI) -> Vector{UInt32}

The order that stores `nanni`'s database as its search visits it: the centers first, then the
members cell by cell, each cell by increasing distance to its center. See [`permutenanni`](@ref).
"""
nannipermutation(nanni::NANNI) = vcat(nanni.centers, nanni.members)

"""
    permutenanni(nanni::NANNI, π=nannipermutation(nanni), π′=invperm(π); permutedb=_permutedb)

A [`PermutedSearchIndex`](@ref) over a copy of `nanni` whose database is stored in the order `π`
(by default the centers and then the cells, see [`nannipermutation`](@ref)), answering with the
original identifiers. The centers are then evaluated from contiguous memory, and so is every run
of members a cell evaluates. `permutedb(db, π)` makes the copy of the database.
"""
function permutenanni(nanni::NANNI, π=nannipermutation(nanni), π′=invperm(π); permutedb=_permutedb)
    N = NANNI(distance(nanni), permutedb(database(nanni), π), convert(Vector{UInt32}, π′[nanni.centers]),
              copy(nanni.cov), copy(nanni.cellstart), convert(Vector{UInt32}, π′[nanni.members]), copy(nanni.mdist))
    PermutedSearchIndex(; index=N, π, π′)
end
