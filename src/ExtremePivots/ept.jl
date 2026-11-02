# This file is a part of SimilaritySearch.jl

"""
    EPTable(dist::Dist.Metric, db::AbstractDatabase; groups::Integer=4, checkmetric::Bool=true)
    EPTable(db::AbstractDatabase; dist::Dist.Metric, groups::Integer=4, checkmetric::Bool=true)

The extreme pivot table (EPT) of Ruiz, Chávez, Ruiz and Tellez, *Extreme pivots: a pivot
selection strategy for faster metric search*, Knowledge and Information Systems (2020),
doi:10.1007/s10115-019-01423-5: an **exact** index for metrics.

A classic pivot table stores the distance from every object to every pivot, yet for a given
object a single pivot ends up discarding it: the one that sees it far from the bulk of its
distances, since a query lands near that bulk. The EPT keeps only that pivot. Each of the
`groups` pivot groups is a set of pivots, and every object is associated, in each group, with
the pivot `p` that maximizes `|d(u, p) - μ_p|`, where `μ_p` is the mean distance from `p` to the
database; the object stores that pivot and `d(u, p)`. The memory is `groups` (pivot, distance)
pairs per object, whatever the number of pivots, so the groups can hold many pivots: their
number is tuned during the construction.

A search evaluates every pivot (they are pushed into the result, being objects of the
database), then evaluates an object only when none of its `groups` pivots discards it,
`|d(q, p) - d(u, p)| > r` with `r` the covering radius of the result. The table is stored in
blocks, one per pivot of the first group holding the objects associated with it, each block
sorted by the distance to that pivot. The search visits the blocks by increasing distance from
the query to their pivots, so the radius shrinks early, and in each block the objects the first
group lets through form a contiguous run found by binary search; the other groups are checked
only on that run. The paper describes this kind of sublinear scan as possible but implements the
flat scan.

[`index!`](@ref index!(::EPTable, ::AbstractContext)) builds the groups. Each one grows by
windows of random pivots while an estimate of the search cost keeps falling: the cost model of
the paper (its Algorithm 3), from the variance of the distances the pivots keep and from
statistics of the queries, or the measured cost of a sample of queries.

# When it pays off

Like every metric index, it prunes when the query radius is small next to the spread of the
distances: low intrinsic dimension, or costly distances. In high intrinsic dimension a pivot
discards little and the scan approaches an exhaustive search; an approximate index such as
[`SearchGraph`](@ref) is the right tool there.

# Arguments
- `dist`: the distance function, a metric
- `db`: the database to index

# Keyword Arguments
- `groups`: the number of pivot groups, i.e., of (pivot, distance) pairs stored per object, the
  memory of the index (the paper's `ℓ`; it suggests 2 to 4 for cheap distances and more for
  costly ones)
- `checkmetric`: whether a non-`Dist.Metric` distance is rejected (default `true`)

# Examples

```julia
using SimilaritySearch

db = MatrixDatabase(rand(Float32, 4, 10_000))
ept = EPTable(Dist.L2(), db; groups=4)
ctx = GenericContext()
index!(ept, ctx)
search(ept, ctx, rand(Float32, 4), knnqueue(KnnSorted, 5))
```
"""
struct EPTable{DistanceType<:Dist.PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
    pivots::Vector{UInt32}      # database ids of the pivots of every group, distinct
    pgroup::Vector{Int32}       # per pivot: its group
    pmean::Vector{Float32}      # per pivot: its mean distance to the database, μ_p
    # n × groups, a column per group, rows in the order of `order`: most objects are discarded by
    # their first group, so a search reads the first column and touches the others only for the
    # rows that survive it
    piv::Matrix{UInt32}         # per row and group: the row's pivot (an index into `pivots`)
    pdist::Matrix{Float32}      # per row and group: its distance to that pivot
    ispivot::BitVector          # per object (database id): whether it is a pivot of some group
    order::Vector{UInt32}       # per row: its object; the identity while the table is being built
    bpiv::Vector{UInt32}        # per block: its pivot of the first group (an index into `pivots`)
    blockstart::Vector{Int32}   # per block: its first row; length = #blocks + 1, empty while building
end

function EPTable(dist::Dist.PreMetric, db::AbstractDatabase; groups::Integer=4, checkmetric::Bool=true)
    if checkmetric && !(dist isa Dist.Metric)
        throw(ArgumentError("""
            EPTable filters with the triangle inequality, which $(typeof(dist)) does not promise \
            (it is not a `Dist.Metric`); an index built on it can miss true neighbors. \
            Pass `checkmetric=false` if this distance really is a metric but is not typed as one."""))
    end
    groups >= 1 || throw(ArgumentError("groups must be positive, not $groups"))
    n = length(db)
    EPTable(dist, db, UInt32[], Int32[], Float32[], zeros(UInt32, n, groups), zeros(Float32, n, groups), falses(n),
            collect(UInt32(1):UInt32(n)), UInt32[], Int32[])
end

EPTable(db::AbstractDatabase; dist::Dist.PreMetric, groups::Integer=4, checkmetric::Bool=true) =
    EPTable(dist, db; groups, checkmetric)

@inline database(ept::EPTable) = ept.db
@inline database(ept::EPTable, i) = ept.db[i]
@inline distance(ept::EPTable) = ept.dist
@inline Base.length(ept::EPTable) = length(ept.db)
ngroups(ept::EPTable) = size(ept.piv, 2)

"""
    getcontext(ept::EPTable; kwargs...)

A [`GenericContext`](@ref): an `EPTable` needs no dedicated context type. Not part of the formal
`AbstractSearchIndex` interface, a convenience.
"""
getcontext(::EPTable; kwargs...) = GenericContext(; kwargs...)

"""
    _addpivot!(ept, ctx, g, p, row) -> evaluations

Adds the object `p` as a pivot of group `g`: its distance to every object (into `row`), its mean
`μ_p`, and every object of the group moves to `p` when `|d(u, p) - μ_p|` beats the same measure
for its current pivot (the first pivot of a group takes every object).
"""
function _addpivot!(ept::EPTable, ctx::AbstractContext, g::Integer, p::Integer, row::Vector{Float32})
    dist, db = distance(ept), database(ept)
    n = length(db)
    q = db[p]
    @BATCHES getminbatch(ctx, n) scheduler=ctx.scheduler begin
    @BEGINBATCH
        bdist = beginbatch(dist)
    @LOOP for u in 1:n
        @inbounds row[u] = Dist.evaluate(bdist, db[u], q)
    end
    end

    μ = Float32(sum(Float64, row) / n)
    isfirst = !any(==(g), ept.pgroup)
    push!(ept.pivots, p)
    push!(ept.pgroup, g)
    push!(ept.pmean, μ)
    ept.ispivot[p] = true
    P = length(ept.pivots)
    piv, pdist, pmean = ept.piv, ept.pdist, ept.pmean
    @BATCHES getminbatch(ctx, n) scheduler=ctx.scheduler begin
    @LOOP for u in 1:n
        @inbounds if isfirst || abs(row[u] - μ) > abs(pdist[u, g] - pmean[piv[u, g]])
            piv[u, g] = P
            pdist[u, g] = row[u]
        end
    end
    end

    n
end

"the variance of the distances group `g` keeps around their pivots' means, σ_X²"
function _groupvariance(ept::EPTable, g::Integer)
    s = 0.0
    @inbounds for u in axes(ept.piv, 1)
        s += abs2(ept.pdist[u, g] - ept.pmean[ept.piv[u, g]])
    end
    s / size(ept.piv, 1)
end

"""
    _querystats(db, nqueries, nsample, rng) -> (σ_Y², r)

The query statistics of the cost model, estimated as the paper's implementation does: `nqueries`
random objects, each against `nsample` random objects; σ_Y² is the variance of those distances,
and `r`, the query radius, the largest over the queries of the smallest nonzero distance seen.
"""
function _querystats(dist, db::AbstractDatabase, nqueries::Integer, nsample::Integer, rng::AbstractRNG)
    n = length(db)
    s1 = s2 = 0.0
    r = 0.0
    N = nqueries * nsample
    for _ in 1:nqueries
        q = db[rand(rng, 1:n)]
        m = Inf
        for _ in 1:nsample
            d = Float64(Dist.evaluate(dist, q, db[rand(rng, 1:n)]))
            s1 += d / N
            s2 += d * d / N
            d > 0 && (m = min(m, d))
        end
        isfinite(m) && (r = max(r, m))
    end
    s2 - s1 * s1, r
end

"""
    _searchcount(ept::EPTable, q, res) -> evaluations

The search of [`EPTable`](@ref), returning the number of distances it evaluated.
"""
function _searchcount(ept::EPTable, q, res::AbstractMetricQueue)
    dist = distance(ept)
    np = length(ept.pivots)
    np == 0 && return 0
    dqp = Vector{Float32}(undef, np)
    @inbounds for i in 1:np
        id = ept.pivots[i]
        d = Dist.evaluate(dist, q, database(ept, id))
        dqp[i] = d
        push_item!(res, id, d)
    end

    cost = np
    piv, pdist = ept.piv, ept.pdist
    G = ngroups(ept)
    r = covradius(res)   # it only changes when an object is pushed
    if !isempty(ept.blockstart)   # the built table: blocks by increasing d(q, p), a run per block
        @inbounds for b in sortperm(view(dqp, ept.bpiv))
            dp = dqp[ept.bpiv[b]]
            lo, hi = ept.blockstart[b], ept.blockstart[b+1] - 1
            j = searchsortedfirst(view(pdist, lo:hi, 1), dp - r) + lo - 1
            while j <= hi
                d1 = pdist[j, 1]
                d1 > dp + r && break
                if dp - d1 <= r   # r shrinks along the run, so the lower end is checked again
                    u = ept.order[j]
                    review = !ept.ispivot[u]
                    for g in 2:G
                        review || break
                        review = abs(dqp[piv[j, g]] - pdist[j, g]) <= r
                    end

                    if review
                        push_item!(res, u, Dist.evaluate(dist, q, database(ept, u)))
                        r = covradius(res)
                        cost += 1
                    end
                end
                j += 1
            end
        end

        return cost
    end

    @inbounds for u in axes(piv, 1)   # while building: rows are objects, a flat scan
        abs(dqp[piv[u, 1]] - pdist[u, 1]) > r && continue
        ept.ispivot[u] && continue
        review = true
        for g in 2:G
            if abs(dqp[piv[u, g]] - pdist[u, g]) > r
                review = false
                break
            end
        end

        if review
            push_item!(res, u, Dist.evaluate(dist, q, database(ept, u)))
            r = covradius(res)
            cost += 1
        end
    end

    cost
end

"""
    search(ept::EPTable, ctx::AbstractContext, q, res::AbstractMetricQueue)

Solves `q` exactly (see [`EPTable`](@ref)): every pivot is evaluated, then every object that none
of its pivots discards.
"""
function search(ept::EPTable, ctx::AbstractContext, q, res::AbstractMetricQueue)
    add_distance_evaluations!(ctx, _searchcount(ept, q, res))
    res
end

"the mean number of distance evaluations of solving the `k` nearest neighbors of every object of `Q`"
function _querycost(ept::EPTable, ctx::AbstractContext, Q::AbstractDatabase, k::Integer)
    nq = length(Q)
    evals = zeros(Int, nq)
    @BATCHES getminbatch(ctx, nq) scheduler=ctx.scheduler begin
    @LOOP for i in 1:nq
        evals[i] = _searchcount(ept, Q[i], SimilaritySearch.knnqueue(KnnSorted, k))
    end
    end

    sum(evals) / nq
end

"""
    index!(ept::EPTable, ctx::AbstractContext; window=16, cost=:model, maxpivots=length(ept),
           nstats=128, queries=nothing, k=1, nqueries=32, rng=Random.default_rng())

Builds the pivot groups of `ept` (see [`EPTable`](@ref)) from random pivots, distinct across
groups, and returns `ept`. The index must be empty (`EPTable` is built once).

With `cost=:model` (the default) each group is built as the paper's Algorithm 3 does: pivots
are added by windows of `window`; after every pivot the expected cost
`m ℓ + n (1 - (σ_X² + σ_Y²) / r²)^ℓ` is computed (`m` pivots in the group, `ℓ` groups, σ_X² the
variance the group keeps around its pivots' means, σ_Y² and `r` the query statistics estimated
from `nstats` random objects against as many others), and the group stops at the first window
whose mean expected cost is not below the previous one. With `cost=:distances` all the groups
grow together by windows, and the construction stops at the first window that does not lower the
mean number of distance evaluations of the training queries. Both minimize evaluations, not wall
time: they carry no timing noise, and the distances of this package are optimized at a low level.

# Keyword Arguments
- `window`: pivots added between two cost estimates (the paper fixes 16)
- `cost`: `:model` or `:distances`
- `maxpivots`: an upper bound on the pivots of each group
- `nstats`: the sample of the query statistics of the model
- `queries`, `k`, `nqueries`: the training queries of the measured costs, as in
  [`index!`](@ref index!(::ANNI, ::AbstractContext)) for [`ANNI`](@ref): `queries` with `k`
  neighbors, or `nqueries` random objects of the database with `k + 1`
- `rng`: the random source of the pivots and of the samples
"""
function index!(ept::EPTable, ctx::AbstractContext; window::Integer=16, cost::Symbol=:model,
                maxpivots::Integer=length(ept), nstats::Integer=128, queries=nothing, k::Integer=1,
                nqueries::Integer=32, rng::AbstractRNG=default_rng())
    cost in (:model, :distances) || throw(ArgumentError("cost must be :model or :distances, not $cost"))
    isempty(ept.pivots) || throw(ArgumentError("the index is already built; EPTable is built once"))
    window >= 1 || throw(ArgumentError("window must be positive, not $window"))
    db = database(ept)
    n = length(db)
    n == 0 && return ept
    G = ngroups(ept)
    maxpivots = min(maxpivots, n ÷ G)
    maxpivots >= 1 || throw(ArgumentError("$n objects cannot feed $G groups of pivots"))
    order = randperm(rng, n)
    next = 1
    row = Vector{Float32}(undef, n)

    if cost === :model
        varY, r = _querystats(distance(ept), db, nstats, nstats, rng)
        r > 0 || (r = 1.0)
        for g in 1:G
            prev = Float64(n)
            m = 0
            while m < maxpivots
                c = 0.0
                w = 0
                while w < window && m < maxpivots
                    _addpivot!(ept, ctx, g, order[next], row)
                    next += 1; m += 1; w += 1
                    s = min(1.0, (_groupvariance(ept, g) + varY) / r^2)
                    c += m * G + n * (1 - s)^G
                end

                verbose(ctx) && @info "EPTable group $g: $m pivots, expected cost $(c / w)"
                c / w - prev >= 0 && break
                prev = c / w
            end
        end
    else
        Q, kq = queries === nothing ? (db[randperm(rng, n)[1:min(nqueries, n)]], k + 1) : (queries, k)
        prev = Inf
        m = 0
        while m < maxpivots
            w = 0
            while w < window && m < maxpivots
                for g in 1:G
                    _addpivot!(ept, ctx, g, order[next], row)
                    next += 1
                end
                m += 1; w += 1
            end

            curr = _querycost(ept, ctx, Q, kq)
            verbose(ctx) && @info "EPTable: $m pivots per group, cost $prev -> $curr"
            curr < prev || break
            prev = curr
        end
    end

    _blocks!(ept)
end

"""
    _blocks!(ept::EPTable) -> ept

Stores the built table in blocks: rows sorted by their pivot of the first group and, within a
block, by the distance to it.
"""
function _blocks!(ept::EPTable)
    n = size(ept.piv, 1)
    o = sortperm(collect(1:n), by=u -> (ept.piv[u, 1], ept.pdist[u, 1]))
    ept.piv .= ept.piv[o, :]
    ept.pdist .= ept.pdist[o, :]
    ept.order .= ept.order[o]
    empty!(ept.bpiv); empty!(ept.blockstart)
    for j in 1:n
        if j == 1 || ept.piv[j, 1] != ept.piv[j-1, 1]
            push!(ept.bpiv, ept.piv[j, 1])
            push!(ept.blockstart, j)
        end
    end
    push!(ept.blockstart, n + 1)
    ept
end

"""
    eptpermutation(ept::EPTable) -> Vector{UInt32}

The order that stores `ept`'s database as its table: by block (pivot of the first group) and,
within a block, by increasing distance to that pivot. See [`permuteept`](@ref).
"""
eptpermutation(ept::EPTable) = copy(ept.order)

"""
    permuteept(ept::EPTable, π=eptpermutation(ept), π′=invperm(π); permutedb)

A [`PermutedSearchIndex`](@ref) over a copy of `ept` whose database is stored in the order `π`
(by default by the first group, see [`eptpermutation`](@ref)), answering with the original
identifiers. The runs of objects a block lets through are then stored next to each other. `permutedb(db, π)` makes the copy of the
database (by default a `MatrixDatabase` keeps its layout and anything else becomes a
`VectorDatabase` of copies).
"""
function permuteept(ept::EPTable, π=eptpermutation(ept), π′=invperm(π); permutedb=_permutedb)
    # rows keep their place; each row's object gets its new identifier
    E = EPTable(distance(ept), permutedb(database(ept), π), convert(Vector{UInt32}, π′[ept.pivots]), copy(ept.pgroup),
                copy(ept.pmean), copy(ept.piv), copy(ept.pdist), ept.ispivot[π], convert(Vector{UInt32}, π′[ept.order]),
                copy(ept.bpiv), copy(ept.blockstart))
    PermutedSearchIndex(; index=E, π, π′)
end

_permutedb(db::MatrixDatabase, π) = MatrixDatabase(db.matrix[:, π])
_permutedb(db::AbstractDatabase, π) = VectorDatabase([copy(db[i]) for i in π])
