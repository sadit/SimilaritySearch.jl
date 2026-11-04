# This file is a part of SimilaritySearch.jl

"""
    NEP(dist::Dist.Metric, db::AbstractDatabase; nfar::Integer=1, checkmetric::Bool=true)
    NEP(db::AbstractDatabase; dist::Dist.Metric, nfar::Integer=1, checkmetric::Bool=true)

The node-based extreme pivots (NEP), which are to [`EPTable`](@ref) what [`NANNI`](@ref) is to
[`ANNI`](@ref): an **exact** index for metrics that gives one set of pivots two roles: the nearest pivot of
every object defines its node, a cell of a Dirichlet partition as in [`NANNI`](@ref), and the
pivots that see it farthest above their mean distance act as extreme pivots, as in
[`EPTable`](@ref), filtering it inside its cell. Every object keeps its distance to its center
and, for each of its `nfar` far pivots, the pivot and the distance to it; these data are stored
by cell, aligned with the cell's members, so a cell is read as contiguous memory.

A far pivot is chosen on the far side, by the largest `d(u, p) - μ_p`, `μ_p` being the mean
distance from `p` to the database. On the near side the most extreme pivot of an object is
usually its own center, which already filters it.

A search evaluates every pivot (they are objects of the database and go into the result), then
visits the cells by increasing distance from the query to their centers, with the rules of
[`NANNI`](@ref): it stops at the first cell beyond the hyperplane rule, skips a cell whose
covering ball misses the query ball, and inside a cell considers only the run of members whose
distance to the center is within `r` of the query's. Each member of that run is evaluated only
when none of its far pivots discards it, `|d(q, p) - d(u, p)| > r`. The far filters cost no
evaluation: the query's distance to every pivot is already known.

[`index!`](@ref index!(::NEP, ::AbstractContext)) adds random pivots in steps and
stops when the mean number of distance evaluations of a sample of queries no longer falls.
`nfar=0` is a [`NANNI`](@ref) over the same pivots, a reference to measure the far filters by.

# Arguments
- `dist`: the distance function, a metric
- `db`: the database to index

# Keyword Arguments
- `nfar`: far pivots kept per object (its internal filters)
- `checkmetric`: whether a non-`Dist.Metric` distance is rejected (default `true`)
"""
struct NEP{DistanceType<:Dist.PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
    nfar::Int
    pivots::Vector{UInt32}     # database ids of the pivots; pivot i is the center of cell i
    pmean::Vector{Float32}     # per pivot: its mean distance to the database
    cov::Vector{Float32}       # per cell: its covering radius
    cellstart::Vector{Int32}   # per cell: 1-based offset into the member arrays; length = #pivots + 1
    members::Vector{UInt32}    # the non-pivot objects, by cell, ascending distance to their center
    mdist::Vector{Float32}     # each member's distance to its center
    far::Vector{UInt32}        # nfar per member, member-major: each member's far pivots (indices into `pivots`)
    fdist::Vector{Float32}     # nfar per member, aligned with `far`: its distances to them
end

function NEP(dist::Dist.PreMetric, db::AbstractDatabase; nfar::Integer=1, checkmetric::Bool=true)
    if checkmetric && !(dist isa Dist.Metric)
        throw(ArgumentError("""
            NEP filters with the triangle inequality, which $(typeof(dist)) does not promise \
            (it is not a `Dist.Metric`); an index built on it can miss true neighbors. \
            Pass `checkmetric=false` if this distance really is a metric but is not typed as one."""))
    end
    nfar >= 0 || throw(ArgumentError("nfar must be non-negative, not $nfar"))
    NEP(dist, db, Int(nfar), UInt32[], Float32[], Float32[], Int32[], UInt32[], Float32[], UInt32[], Float32[])
end

NEP(db::AbstractDatabase; dist::Dist.PreMetric, nfar::Integer=1, checkmetric::Bool=true) =
    NEP(dist, db; nfar, checkmetric)

@inline database(nf::NEP) = nf.db
@inline database(nf::NEP, i) = nf.db[i]
@inline distance(nf::NEP) = nf.dist
@inline Base.length(nf::NEP) = length(nf.db)
getcontext(::NEP; kwargs...) = GenericContext(; kwargs...)

# the construction's per-object state: center, distance to it, and the nfar best far pivots with
# their scores d(u, p) - μ_p (descending) and distances
struct _NEPState
    near::Vector{UInt32}
    dnear::Vector{Float32}
    far::Matrix{UInt32}
    fscore::Matrix{Float32}
    fdist::Matrix{Float32}
end

_NEPState(n, nfar) = _NEPState(zeros(UInt32, n), fill(typemax(Float32), n), zeros(UInt32, nfar, n),
                                       fill(-typemax(Float32), nfar, n), zeros(Float32, nfar, n))

"""
    _addpivot!(nf::NEP, st, ctx, p, row) -> evaluations

Adds the object `p` as a pivot: its distance to every object (into `row`) and its mean; every
object moves to it if it is nearer than its center, and takes it among its far pivots if its
score `d(u, p) - μ_p` is among the `nfar` largest.
"""
function _addpivot!(nf::NEP, st::_NEPState, ctx::AbstractContext, p::Integer, row::Vector{Float32})
    dist, db = distance(nf), database(nf)
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
    push!(nf.pivots, p)
    push!(nf.pmean, μ)
    P = UInt32(length(nf.pivots))
    F = nf.nfar
    near, dnear, far, fscore, fdist = st.near, st.dnear, st.far, st.fscore, st.fdist
    @BATCHES getminbatch(ctx, n) scheduler=ctx.scheduler begin
    @LOOP for u in 1:n
        @inbounds begin
            d = row[u]
            if d < dnear[u]
                near[u] = P
                dnear[u] = d
            end
            s = d - μ
            if F > 0 && s > fscore[F, u]   # insertion into the descending top-nfar
                t = F
                while t > 1 && fscore[t-1, u] < s
                    far[t, u] = far[t-1, u]; fscore[t, u] = fscore[t-1, u]; fdist[t, u] = fdist[t-1, u]
                    t -= 1
                end
                far[t, u] = P; fscore[t, u] = s; fdist[t, u] = d
            end
        end
    end
    end

    n
end

"""
    _fromstate!(nf::NEP, st) -> nf

Rewrites the cells of `nf` from the construction state: a counting sort of the non-pivot objects
by center, each cell by distance to its center, and the far pivots aligned with the members.
"""
function _fromstate!(nf::NEP, st::_NEPState)
    m = length(nf.pivots)
    n = length(st.near)
    ispivot = falses(n)
    ispivot[nf.pivots] .= true
    resize!(nf.cov, m); fill!(nf.cov, 0f0)
    resize!(nf.cellstart, m + 1); fill!(nf.cellstart, 0)
    @inbounds for u in 1:n
        ispivot[u] && continue
        a = st.near[u]
        nf.cellstart[a+1] += 1
        nf.cov[a] = max(nf.cov[a], st.dnear[u])
    end

    nf.cellstart[1] = 1
    for a in 1:m
        nf.cellstart[a+1] += nf.cellstart[a]
    end

    nm = n - m
    resize!(nf.members, nm)
    resize!(nf.mdist, nm)
    pos = nf.cellstart[1:m]
    @inbounds for u in 1:n
        ispivot[u] && continue
        a = st.near[u]
        nf.members[pos[a]] = u
        nf.mdist[pos[a]] = st.dnear[u]
        pos[a] += 1
    end

    Threads.@threads for a in 1:m
        lo, hi = nf.cellstart[a], nf.cellstart[a+1] - 1
        if hi > lo
            o = sortperm(view(nf.mdist, lo:hi))
            nf.members[lo:hi] = nf.members[lo:hi][o]
            nf.mdist[lo:hi] = nf.mdist[lo:hi][o]
        end
    end

    F = nf.nfar
    resize!(nf.far, F * nm)
    resize!(nf.fdist, F * nm)
    @inbounds for j in 1:nm
        u = nf.members[j]
        for t in 1:F
            nf.far[(j-1)*F+t] = st.far[t, u]
            nf.fdist[(j-1)*F+t] = st.fdist[t, u]
        end
    end

    nf
end

"""
    _searchcount(nf::NEP, q, res, factor=1f0) -> evaluations

The search of [`NEP`](@ref), returning the number of distances it evaluated. Every filter
tests against `factor` times the covering radius: the exact search for 1, the stretched
triangle inequality of [`PruningSearch`](@ref) below it.
"""
function _searchcount(nf::NEP, q, res::AbstractMetricQueue, factor::Float32=1f0)
    dist = distance(nf)
    m = length(nf.pivots)
    m == 0 && return 0
    dq = Vector{Float32}(undef, m)
    dmin = typemax(Float32)
    @inbounds for i in 1:m
        id = nf.pivots[i]
        d = Dist.evaluate(dist, q, database(nf, id))
        dq[i] = d
        dmin = min(dmin, d)
        push_item!(res, id, d)
    end

    cost = m
    F = nf.nfar
    r = factor * covradius(res)
    @inbounds for i in sortperm(dq)
        dc = dq[i]
        dc > dmin + 2r && break      # every later cell is farther: the hyperplane rule holds for all of them
        dc > nf.cov[i] + r && continue
        lo, hi = nf.cellstart[i], nf.cellstart[i+1] - 1
        lo > hi && continue
        j = searchsortedfirst(view(nf.mdist, lo:hi), dc - r) + lo - 1
        while j <= hi
            dx = nf.mdist[j]
            dx > dc + r && break
            if dc - dx <= r
                keep = true
                for t in (j-1)*F+1:j*F
                    if abs(dq[nf.far[t]] - nf.fdist[t]) > r
                        keep = false
                        break
                    end
                end

                if keep
                    u = nf.members[j]
                    push_item!(res, u, Dist.evaluate(dist, q, database(nf, u)))
                    r = factor * covradius(res)
                    cost += 1
                end
            end
            j += 1
        end
    end

    cost
end

"""
    search(nf::NEP, ctx::AbstractContext, q, res::AbstractMetricQueue)

Solves `q` exactly (see [`NEP`](@ref)).
"""
function search(nf::NEP, ctx::AbstractContext, q, res::AbstractMetricQueue)
    add_distance_evaluations!(ctx, _searchcount(nf, q, res))
    res
end

"""
    index!(nf::NEP, ctx::AbstractContext; queries=nothing, k=1, nqueries=32,
           step=isqrt(length(nf)) + 1, alphastop=0.001, maxpivots=length(nf), rng=Random.default_rng())

Builds `nf` (see [`NEP`](@ref)) and returns it. Random pivots are added in steps of `step` while
the mean number of distance evaluations of the training queries falls by more than a factor
`1 + alphastop`, as [`index!`](@ref index!(::ANNI, ::AbstractContext)) does for [`ANNI`](@ref),
whose keyword arguments these are. Every pivot costs one distance per object of the database.
"""
function index!(nf::NEP, ctx::AbstractContext; queries=nothing, k::Integer=1, nqueries::Integer=32,
                step::Integer=isqrt(length(nf)) + 1, alphastop::Real=0.001, maxpivots::Integer=length(nf),
                rng::AbstractRNG=default_rng())
    isempty(nf.pivots) || throw(ArgumentError("the index is already built; NEP is built once"))
    db = database(nf)
    n = length(db)
    n == 0 && return nf
    Q, kq = queries === nothing ? (db[randperm(rng, n)[1:min(nqueries, n)]], k + 1) : (queries, k)
    st = _NEPState(n, nf.nfar)
    order = randperm(rng, n)
    row = Vector{Float32}(undef, n)
    maxpivots = min(maxpivots, n)
    next = 1
    prev = Inf
    while next <= maxpivots
        for _ in 1:min(step, maxpivots - next + 1)
            _addpivot!(nf, st, ctx, order[next], row)
            next += 1
        end

        _fromstate!(nf, st)
        curr = _querycost(nf, ctx, Q, kq)
        verbose(ctx) && @info "NEP: $(length(nf.pivots)) pivots, cost $prev -> $curr"
        prev > curr * (1 + alphastop) || break
        prev = curr
    end

    nf
end

function _querycost(nf::NEP, ctx::AbstractContext, Q::AbstractDatabase, k::Integer)
    nq = length(Q)
    evals = zeros(Int, nq)
    @BATCHES getminbatch(ctx, nq) scheduler=ctx.scheduler begin
    @LOOP for i in 1:nq
        evals[i] = _searchcount(nf, Q[i], SimilaritySearch.knnqueue(KnnSorted, k))
    end
    end

    sum(evals) / nq
end

"""
    neppermutation(nf::NEP) -> Vector{UInt32}

The order that stores `nf`'s database as its search visits it: the pivots first, then the
members cell by cell. See [`permutenep`](@ref).
"""
neppermutation(nf::NEP) = vcat(nf.pivots, nf.members)

"""
    permutenep(nf::NEP, π=neppermutation(nf), π′=invperm(π); permutedb=_permutedb)

A [`PermutedSearchIndex`](@ref) over a copy of `nf` whose database is stored in the order `π`,
answering with the original identifiers: the pivots and every run of members a cell evaluates
then sit in contiguous memory.
"""
function permutenep(nf::NEP, π=neppermutation(nf), π′=invperm(π); permutedb=_permutedb)
    N = NEP(distance(nf), permutedb(database(nf), π), nf.nfar, convert(Vector{UInt32}, π′[nf.pivots]),
                      copy(nf.pmean), copy(nf.cov), copy(nf.cellstart), convert(Vector{UInt32}, π′[nf.members]),
                      copy(nf.mdist), copy(nf.far), copy(nf.fdist))
    PermutedSearchIndex(; index=N, π, π′)
end
