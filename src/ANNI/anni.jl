# This file is a part of SimilaritySearch.jl

"""
    ANNI(dist::Dist.Metric, db::AbstractDatabase; checkmetric::Bool=true)
    ANNI(db::AbstractDatabase; dist::Dist.Metric, checkmetric::Bool=true)

The auto-tuned nearest neighbor index (ANNI) of Tellez, Ruiz and Chávez, *Singleton indexes
for nearest neighbor search*, Information Systems 60 (2016) 50--68: an **exact** index for
metrics. A set of objects of the database are its centers, and every other object keeps two
numbers, the center nearest to it and its distance to that center. The centers induce a
Dirichlet (Voronoi) partition of the database, and the index is a pivot table where each
object is filtered by a single pivot, its own center.

A search evaluates every center, then visits the objects in database order and evaluates an
object `u` only when the triangle inequality cannot discard it,
`|d(q, c(u)) - d(u, c(u))| <= r`, with `r` the covering radius of the result. The search is
therefore a scan of two arrays plus the evaluations that survive the filter: cheap internal
work, which is what makes the index fast in practice even where it evaluates more distances
than heavier indexes.

[`index!`](@ref index!(::ANNI, ::AbstractContext)) builds it incrementally. Random objects are
promoted to centers in steps, each new center stealing the objects that are now nearer to it
(the only distances computed are those the triangle inequality cannot rule out), and after
every step the mean cost of a sample of queries is measured. The construction stops when a
step no longer lowers that cost, so the number of centers is tuned to the data and to the
queries, with no parameter to set by hand. [`NANNI`](@ref) stores the same partition by
cluster, and adds the covering-radius and hyperplane rules to discard whole cells.

# When it pays off

Like every metric index, it prunes when the query's covering radius is small next to the
spread of the distances: low intrinsic dimension, or costly distances (edit distances over
strings, for example), where a scan with few evaluations wins. In high intrinsic dimension the
filter discards little and the search approaches an exhaustive scan with the cost of the
centers on top; there an approximate index such as [`SearchGraph`](@ref) is the right tool.

# Why a `Metric` and not a `SemiMetric`

The filter, the stealing rule of the construction and the Dirichlet partition all rely on the
triangle inequality. The constructor rejects a non-`Metric` distance; `checkmetric=false`
overrides it when the distance is a metric typed loosely, or when an approximate index is
acceptable.

# Arguments
- `dist`: the distance function, a metric
- `db`: the database to index

# Keyword Arguments
- `checkmetric`: whether a non-`Dist.Metric` distance is rejected (default `true`)

# Examples

```julia
using SimilaritySearch

db = MatrixDatabase(rand(Float32, 4, 10_000))
anni = ANNI(Dist.L2(), db)
ctx = GenericContext()
index!(anni, ctx)
search(anni, ctx, rand(Float32, 4), knnqueue(KnnSorted, 5))
```
"""
struct ANNI{DistanceType<:Dist.PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
    centers::Vector{UInt32}   # database ids of the centers, in the order they were promoted
    center::Vector{UInt32}    # per object: index into `centers` of its nearest center; 0 for a center
    dcenter::Vector{Float32}  # per object: its distance to that center; 0 for a center
end

function _checkmetric(T, dist, checkmetric)
    if checkmetric && !(dist isa Dist.Metric)
        throw(ArgumentError("""
            $T filters with the triangle inequality, which $(typeof(dist)) does not promise \
            (it is not a `Dist.Metric`); an index built on it can miss true neighbors. \
            Pass `checkmetric=false` if this distance really is a metric but is not typed as one."""))
    end
end

function ANNI(dist::Dist.PreMetric, db::AbstractDatabase; checkmetric::Bool=true)
    _checkmetric(ANNI, dist, checkmetric)
    ANNI(dist, db, UInt32[], UInt32[], Float32[])
end

ANNI(db::AbstractDatabase; dist::Dist.PreMetric, checkmetric::Bool=true) = ANNI(dist, db; checkmetric)

@inline database(anni::ANNI) = anni.db
@inline database(anni::ANNI, i) = anni.db[i]
@inline distance(anni::ANNI) = anni.dist
@inline Base.length(anni::ANNI) = length(anni.db)

"""
    getcontext(anni::ANNI; kwargs...)
    getcontext(nanni::NANNI; kwargs...)

A [`GenericContext`](@ref): these indexes need no dedicated context type. Not part of the
formal `AbstractSearchIndex` interface, a convenience.
"""
getcontext(::ANNI; kwargs...) = GenericContext(; kwargs...)

"""
    _promote!(anni::ANNI, ctx, newcenters) -> evaluations

Promotes the objects `newcenters` (none of them a center yet) to centers, in order, and moves
to each new center the objects nearer to it, keeping the partition a Dirichlet domain. An object `u` whose current center is `a` can only move to a new center
`c` when `d(a, c) <= 2 d(u, a)`: otherwise `d(u, c) >= d(a, c) - d(u, a) > d(u, a)`, and that
distance is never computed (the stealing condition of the paper, its Fig. 5). Every object is
visited once per call, against all the new centers in order, so a step of `s` centers costs one
parallel pass over the database instead of `s`.
"""
function _promote!(anni::ANNI, ctx::AbstractContext, newcenters::AbstractVector{<:Integer})
    dist, db = distance(anni), database(anni)
    m0 = length(anni.centers)
    s = length(newcenters)
    for c in newcenters
        push!(anni.centers, c)
        anni.center[c] = 0
        anni.dcenter[c] = 0f0
    end

    m = length(anni.centers)
    # D[a, j]: distance from center a (old or new) to the j-th new center
    D = Matrix{Float32}(undef, m, s)
    centers = anni.centers
    @BATCHES getminbatch(ctx, m) scheduler=ctx.scheduler begin
    @BEGINBATCH
        bdist = beginbatch(dist)
    @LOOP for a in 1:m
        ca = database(anni, centers[a])
        for j in 1:s
            @inbounds D[a, j] = Dist.evaluate(bdist, ca, db[newcenters[j]])
        end
    end
    end

    nb = max(1, Int(ctx.maxbatches))
    counts = zeros(Int, nb)
    center, dcenter = anni.center, anni.dcenter
    @BATCHES getminbatch(ctx, length(db)) scheduler=ctx.scheduler begin
    @BEGINBATCH
        bdist = beginbatch(dist)
        cnt = 0
    @LOOP for u in eachindex(center)
        @inbounds begin
            center[u] == 0 && continue
            obj = db[u]
            for j in 1:s
                a = center[u]
                if D[a, j] <= 2dcenter[u]
                    d = Dist.evaluate(bdist, obj, db[newcenters[j]])
                    cnt += 1
                    if d < dcenter[u]
                        center[u] = m0 + j
                        dcenter[u] = d
                    end
                end
            end
        end
    end
    @ENDBATCH
        counts[@batchid()] += cnt
    end

    m * s + sum(counts)
end

"""
    _searchcount(anni::ANNI, q, res, factor=1f0) -> evaluations

The search of [`ANNI`](@ref), returning the number of distances it evaluated. Every filter
tests against `factor` times the covering radius: the exact search for 1, the stretched
triangle inequality of [`PruningSearch`](@ref) below it.
"""
function _searchcount(anni::ANNI, q, res::AbstractMetricQueue, factor::Float32=1f0)
    dist = distance(anni)
    m = length(anni.centers)
    m == 0 && return 0
    dcq = Vector{Float32}(undef, m)
    @inbounds for i in 1:m
        id = anni.centers[i]
        d = Dist.evaluate(dist, q, database(anni, id))
        dcq[i] = d
        push_item!(res, id, d)
    end

    cost = m
    center, dcenter = anni.center, anni.dcenter
    r = factor * covradius(res)   # it only changes when an object is pushed
    @inbounds for u in eachindex(center)
        a = center[u]
        a == 0 && continue
        if abs(dcq[a] - dcenter[u]) <= r
            push_item!(res, u, Dist.evaluate(dist, q, database(anni, u)))
            r = factor * covradius(res)
            cost += 1
        end
    end

    cost
end

"""
    search(anni::ANNI, ctx::AbstractContext, q, res::AbstractMetricQueue)

Solves `q` exactly (see [`ANNI`](@ref)): every center is evaluated, then every object the filter
`|d(q, c(u)) - d(u, c(u))| <= r` cannot discard.
"""
function search(anni::ANNI, ctx::AbstractContext, q, res::AbstractMetricQueue)
    add_distance_evaluations!(ctx, _searchcount(anni, q, res))
    res
end

"""
    _querycost(index, ctx, Q, k) -> Float64

The mean number of distance evaluations of solving the `k` nearest neighbors of every object of
`Q` with `index`. The construction minimizes evaluations, not wall time: they carry no timing
noise, and the distances of this package are optimized at a low level, so the evaluations are
what the search cost tracks. The paper's TANNI, which minimizes wall time, is left out on purpose.
"""
function _querycost(index, ctx::AbstractContext, Q::AbstractDatabase, k::Integer)
    nq = length(Q)
    evals = zeros(Int, nq)
    @BATCHES getminbatch(ctx, nq) scheduler=ctx.scheduler begin
    @LOOP for i in 1:nq
        evals[i] = _searchcount(index, Q[i], SimilaritySearch.knnqueue(KnnSorted, k))
    end
    end

    sum(evals) / nq
end

"""
    _autotune!(anni::ANNI, ctx, measure; step, alphastop, maxcenters, rng) -> trace

The construction of the paper (its Algorithm 4): a random first center, then steps of `step`
random promotions; after each step `measure()` gives the current cost, and the construction
stops at the first step whose cost (evaluations per query) is not below the previous one by a factor `1 + alphastop`.
As in the paper, the index keeps that last step: undoing it would cost a reassignment of its
cells, and the cost curve is flat around its minimum. Returns the cost after each step.
"""
function _autotune!(anni::ANNI, ctx::AbstractContext, measure::Function; step::Integer, alphastop::Real,
                    maxcenters::Integer, rng::AbstractRNG)
    isempty(anni.centers) || throw(ArgumentError("the index is already built; ANNI is built once"))
    dist, db = distance(anni), database(anni)
    n = length(db)
    resize!(anni.center, n)
    resize!(anni.dcenter, n)
    trace = Float64[]
    n == 0 && return trace
    order = randperm(rng, n)
    c1 = order[1]
    push!(anni.centers, c1)
    center, dcenter = anni.center, anni.dcenter
    q1 = database(anni, c1)
    @BATCHES getminbatch(ctx, n) scheduler=ctx.scheduler begin
    @BEGINBATCH
        bdist = beginbatch(dist)
    @LOOP for u in 1:n
        @inbounds center[u] = 1
        @inbounds dcenter[u] = Dist.evaluate(bdist, db[u], q1)
    end
    end

    center[c1] = 0
    dcenter[c1] = 0f0
    next = 2
    prev = Inf
    curr = measure()
    push!(trace, curr)
    maxcenters = min(maxcenters, n)
    while next <= n && length(anni.centers) < maxcenters
        s = min(step, maxcenters - length(anni.centers), n - next + 1)
        _promote!(anni, ctx, view(order, next:next+s-1))
        next += s
        prev = curr
        curr = measure()
        push!(trace, curr)
        verbose(ctx) && @info "ANNI construction: $(length(anni.centers)) centers, cost $prev -> $curr"
        prev > curr * (1 + alphastop) || break
    end

    trace
end

"""
    _trainingqueries(db, queries, k, nqueries, rng) -> (Q, k)

The queries the construction measures: `queries` with `k` neighbors when given, otherwise
`nqueries` random objects of the database with `k + 1` neighbors (each finds itself first), as
the paper does when no query sample is available.
"""
function _trainingqueries(db::AbstractDatabase, queries, k::Integer, nqueries::Integer, rng::AbstractRNG)
    queries === nothing || return queries, k
    n = length(db)
    db[randperm(rng, n)[1:min(nqueries, n)]], k + 1
end

"""
    index!(anni::ANNI, ctx::AbstractContext; queries=nothing, k=1, nqueries=32,
           step=isqrt(length(anni)) + 1, alphastop=0.001, maxcenters=length(anni),
           rng=Random.default_rng())

Builds `anni` (see [`ANNI`](@ref)) and tunes its number of centers: random objects are promoted
in steps of `step` centers while the mean number of distance evaluations of the training queries
keeps falling. Returns
`anni`. The index must be empty (`ANNI` is built once; it has no incremental insertion).

# Keyword Arguments
- `queries`: a sample of the expected queries (an `AbstractDatabase`); by default `nqueries`
  random objects of the database
- `k`: the number of neighbors the queries will ask for
- `nqueries`: the size of the default sample (the paper's default is 32)
- `step`: centers promoted between two measurements (the paper's default is `sqrt(n) + 1`)
- `alphastop`: the relative gain below which the construction stops
- `maxcenters`: an upper bound on the number of centers (a memory bound)
- `rng`: the random source of the promotions and of the default sample
"""
function index!(anni::ANNI, ctx::AbstractContext; queries=nothing, k::Integer=1, nqueries::Integer=32,
                step::Integer=isqrt(length(anni)) + 1, alphastop::Real=0.001, maxcenters::Integer=length(anni),
                rng::AbstractRNG=default_rng())
    Q, kq = _trainingqueries(database(anni), queries, k, nqueries, rng)
    measure() = _querycost(anni, ctx, Q, kq)
    _autotune!(anni, ctx, measure; step, alphastop, maxcenters, rng)
    anni
end

"""
    _permutedb(db, π) -> AbstractDatabase

A copy of `db` stored in the order `π`: a `MatrixDatabase` keeps its matrix layout; any other
database becomes a `VectorDatabase` of copies, allocated in that order.
"""
_permutedb(db::MatrixDatabase, π) = MatrixDatabase(db.matrix[:, π])
_permutedb(db::AbstractDatabase, π) = VectorDatabase([copy(db[i]) for i in π])

"""
    annipermutation(anni::ANNI) -> Vector{UInt32}

The order that stores `anni`'s database cell by cell: the centers first, in the order of
`anni.centers`, then the members of every cell, cells in center order and members by increasing
distance to their center. See [`permuteanni`](@ref).
"""
function annipermutation(anni::ANNI)
    members = UInt32[u for u in eachindex(anni.center) if anni.center[u] != 0]
    sort!(members, by=u -> (anni.center[u], anni.dcenter[u]))
    vcat(anni.centers, members)
end

"""
    permuteanni(anni::ANNI, π=annipermutation(anni), π′=invperm(π); permutedb=_permutedb)

A [`PermutedSearchIndex`](@ref) over a copy of `anni` whose database is stored in the order `π`
(by default cell by cell, see [`annipermutation`](@ref)), answering with the original
identifiers. The scan then meets the members of a cell together, so the objects it evaluates in a
cell sit next to each other in memory, and the center lookup of consecutive objects repeats.
`permutedb(db, π)` makes the copy of the database.
"""
function permuteanni(anni::ANNI, π=annipermutation(anni), π′=invperm(π); permutedb=_permutedb)
    A = ANNI(distance(anni), permutedb(database(anni), π), convert(Vector{UInt32}, π′[anni.centers]),
             anni.center[π], anni.dcenter[π])
    PermutedSearchIndex(; index=A, π, π′)
end
