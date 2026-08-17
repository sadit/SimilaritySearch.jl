# This file is part of SimilaritySearch.jl

abstract type AbstractSortSat end
struct RandomSortSat <: AbstractSortSat end
struct ProximalSortSat <: AbstractSortSat end
struct DistalSortSat <: AbstractSortSat end

abstract type AbstractInitialPartition end

struct SatInitialPartition <: AbstractInitialPartition end

struct RandomInitialPartition <: AbstractInitialPartition
    nparts::Int
    shuffle::Bool
    RandomInitialPartition(; nparts=max(4, Threads.nthreads()), shuffle=false) = new(nparts, shuffle)
end

"""
    struct Sat{DT<:Dist.SemiMetric,DBT<:AbstractDatabase} <: AbstractSearchIndex
        dist::DT
        db::DBT
        root::UInt32
        children::Vector{Union{Nothing,Vector{UInt32}}}
        cov::Vector{Float32}
    end

Spatial Access Tree data structure. Please see [`Sat`](@ref) constructor for the
high level entry point, and [`index!`](@ref) to build the tree once constructed.

`cov[i]` is always non-negative: for an internal node (`children[i] !== nothing`) it is the
covering radius (the max distance from `i` to any of its children); for a leaf node
(`children[i] === nothing`) it is simply the distance from `i` to its actual tree parent.
The sign of `cov` does not distinguish leaf/inner status -- use `children[i] === nothing`
for that.
"""
struct Sat{DT<:Dist.SemiMetric,DBT<:AbstractDatabase} <: AbstractSearchIndex
    dist::DT
    db::DBT
    root::UInt32
    children::Vector{Union{Nothing,Vector{UInt32}}}
    cov::Vector{Float32}
end

function Sat(
    sat::Sat;
    dist=sat.dist,
    db=sat.db,
    root=sat.root,
    children=sat.children,
    cov=sat.cov
)
    Sat(dist, db, convert(UInt32, root), children, cov)
end

"""
    Sat(db::AbstractDatabase; dist=Dist.L2(), root=1)

Prepares the metric data structure. After calling this constructor, please call `index!`.

# Arguments

- `db`: database to index

# Keyword arguments
- `dist`: distance function, defaults to `Dist.L2()`
- `root`: The dataset's element to be used as root
"""
function Sat(db::AbstractDatabase; dist::Dist.SemiMetric=Dist.L2(), root=1)
    n = length(db)
    C = Union{Nothing,Vector{UInt32}}[nothing for _ in 1:n]
    cov = Vector{Float32}(undef, n)
    Sat(dist, db, convert(UInt32, root), C, cov)
end

@inline database(sat::Sat) = sat.db
@inline database(sat::Sat, i) = sat.db[i]
@inline distance(sat::Sat) = sat.dist
@inline Base.length(sat::Sat) = length(sat.cov)

"""
    getcontext(sat::Sat; kwargs...) -> GenericContext

Convenience constructor for a plain [`GenericContext`](@ref) suitable for `Sat`'s exact
construction/search. Not part of the formal `AbstractSearchIndex` interface (see
[`SatContext`](@ref) for the approximate variants, which need richer per-batch caches).
"""
getcontext(sat::Sat; kwargs...) = GenericContext(; kwargs...)

"""
    index!(sat::Sat, ctx::AbstractContext, ipart::SatInitialPartition=SatInitialPartition(); <kwargs...>)
    index!(sat::Sat, ctx::AbstractContext, ipart::RandomInitialPartition; <kwargs...>)

Performs the indexing of the referenced dataset in the tree. It supports limited forms of
multithreading, induced by initial partitioning schemes.

# Arguments
- `sat`: The metric data structure.
- `ctx`: context object (caches and hyperparameters).
- `ipart`: initial partitioning scheme for the tree. It supports the following kinds of objects:
    - `SatInitialPartition()`: Traditional construction, default value. Each part is a SAT partition and will be processed in parallel via `@BATCHES`.
    - `RandomInitialPartition(nparts=Threads.nthreads(), shuffle=false)`:
        construction that divides the dataset (randomly if `shuffle=true`) in `nparts` disjoint parts. The resulting structure violates the SAT partitioning in a whole
        and creates a kind of SAT forest that are fine SAT partitions. Useful to limit the height of the tree and for multiprocessing purposes, i.e., each part will be processed in parallel.

# Keyword arguments
- `sortsat`: The strategy to create the spatial access tree, it heavily depends on the order of elements while it is build. It accepts:
   - `RandomSortSat()`: children are randomized (default value)
   - `ProximalSortSat()`: classical approach, near elements are put first.
   - `DistalSortSat()`: recent approach, distant elements are put first.
- `minleaf`: Minimum number of children to perform a spatial access separation (half space partitioning)
"""
function index!(
    sat::Sat,
    ctx::AbstractContext,
    ipart::SatInitialPartition=SatInitialPartition();
    sortsat::AbstractSortSat=RandomSortSat(),
    minleaf::Int=(length(sat) <= 1 ? 0 : ceil(Int, log2(length(sat))))
)
    n = length(sat)
    p::UInt32 = sat.root
    sat.children[p] = collect(UInt32, Iterators.flatten((1:p-1, p+1:n)))

    # Root separation always uses minleaf=0 (never the caller's minleaf), guaranteeing a
    # genuine split into (potentially many) subtrees regardless of thread count -- this is
    # what makes the per-child loop below embarrassingly parallel. Deeper recursion (inside
    # index_loop!) always uses the caller's real `minleaf`. @BATCHES's own fast path (single
    # serial batch whenever Threads.nthreads()==1, ctx.scheduler===:sequential, or the child
    # count is small) makes this produce an identical tree shape regardless of thread count.
    D = Vector{Tuple{Float32,UInt32}}(undef, n)
    index_sat_neighbors!(sat, ctx, sortsat, sat.children[p], p, D, 0)

    C = sat.children[p]
    minbatch = getminbatch(ctx, length(C))
    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
    @LOOP for i in eachindex(C)
        c = C[i]
        if sat.children[c] !== nothing
            Dc = Vector{Tuple{Float32,UInt32}}(undef, length(sat.children[c]))
            index_loop!(sat, bctx, sortsat, Dc, minleaf, UInt32[c])
        end
    end
    end

    sat
end

function index!(
    sat::Sat,
    ctx::AbstractContext,
    ipart::RandomInitialPartition;
    sortsat::AbstractSortSat=RandomSortSat(),
    minleaf::Int=(length(sat) <= 1 ? 0 : ceil(Int, log2(length(sat))))
)
    n = length(sat)
    nparts = 8ipart.nparts > n ? ceil(Int, ipart.nparts / 8) : ipart.nparts
    nparts == 1 && return index!(sat, ctx, SatInitialPartition(); sortsat, minleaf)

    p::UInt32 = sat.root
    P = collect(UInt32, Iterators.flatten((1:p-1, p+1:n)))
    n = length(P)
    ipart.shuffle && shuffle!(P)
    sat.children[p] = P[1:nparts]
    m = ceil(Int, (n - nparts) / nparts)

    minbatch = getminbatch(ctx, nparts)
    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
    @LOOP for i in 1:nparts
        sp = nparts + (i - 1) * m
        c = P[i]
        C = sat.children[c] = P[sp+1:min(n, sp + m)]
        D = Vector{Tuple{Float32,UInt32}}(undef, length(C))
        index_loop!(sat, bctx, sortsat, D, minleaf, UInt32[c])
    end
    end

    cov = sat.cov
    cov[p] = 0.0f0
    for c in (sat.children[p])::Vector{UInt32}
        cov[p] = max(cov[p], abs(cov[c]))
    end

    sat
end

function index_loop!(sat::Sat, ctx::AbstractContext, sortsat::AbstractSortSat, D::AbstractVector, minleaf::Int, queue::Vector{UInt32})
    while length(queue) > 0
        p = pop!(queue)
        index_sat_neighbors!(sat, ctx, sortsat, sat.children[p], p, D, minleaf)
        @inbounds for c in (sat.children[p])::Vector{UInt32}
            C = sat.children[c]
            C !== nothing && push!(queue, c)
        end
    end

    sat
end

function index_sat_neighbors!(sat::Sat, ctx::AbstractContext, sortsat::AbstractSortSat, C::Nothing, p::UInt32, D::Vector, minleaf::Integer)
    # do nothing
end

function index_sat_neighbors!(sat::Sat, ctx::AbstractContext, sortsat::AbstractSortSat, C::AbstractVector, p::UInt32, D::Vector, minleaf::Integer)
    # note: D is a cache of distances and objects, it is used in two ways in this function
    n = length(C)

    resize!(D, n)
    parent = database(sat, p)
    dist = distance(sat)

    # computing distance to its parent (stored in D)
    sat.cov[p] = 0.0f0

    for i in eachindex(C)
        c = C[i]
        d = Dist.evaluate(dist, parent, database(sat, c))
        D[i] = (convert(Float32, d), c)
        # not thread-safe:
        sat.cov[p] = max(sat.cov[p], d) # covering radius
    end

    T = typeof(sortsat)
    T === RandomSortSat && minleaf == 0 || sort!(D, by=first)
    minleaf = min(n, minleaf)

    if minleaf > 0
        @inbounds for i in 1:minleaf  # mandatory leafs
            (d_, i_) = D[i]
            D[i] = (-d_, i_)
        end
    end

    if T === RandomSortSat
        shuffle!(D)
    elseif T === DistalSortSat
        reverse!(D)
    end

    # computing nearest neighbors of $child \in D$ (using previous D and storing the new set on D)
    empty!(C)
    res = knnqueue(KnnSorted, 1)  # reused scratch, avoids one allocation per non-mandatory child

    for (d_, i_) in D
        d_ <= 0 && continue # negative distances encode mandatory leafs, see next outside-for-loop

        reuse!(res)
        push_item!(res, p, d_)  # insert parent
        child = database(sat, i_)

        for j in C
            d = Dist.evaluate(dist, child, database(sat, j))
            push_item!(res, j, d)
        end

        nn = argmin(res)
        sat.cov[i_] = minimum(res) # distance to parent (or a sibling), marked as such
        if sat.children[nn] === nothing
            sat.children[nn] = UInt32[i_]
        else
            push!(sat.children[nn], i_)
        end
    end

    for (d_, i_) in D
        if d_ <= 0  # negative distances encode mandatory leafs
            sat.cov[i_] = abs(d_)
            push!(C, i_)
            continue
        end
    end
end

function searchtree(sat::Sat, ctx::AbstractContext, q, p::Integer, res::AbstractKnnQueue)
    cost = 1
    dist = distance(sat)
    dqp = Dist.evaluate(dist, q, database(sat, p))
    push_item!(res, p, dqp)

    if sat.children[p] !== nothing # inner node
        if length(res) < maxlength(res) || dqp < maximum(res) + sat.cov[p]
            C = sat.children[p]::Vector{UInt32}
            for c in C
                cost += searchtree(sat, ctx, q, c, res)
            end
        end
    end

    cost
end

"""
    search(sat::Sat, ctx::AbstractContext, q, res::AbstractKnnQueue) -> res

Solves query `q` with the spatial access tree, descending from `sat.root` and pruning
subtrees using each internal node's stored covering radius.
"""
function search(sat::Sat, ctx::AbstractContext, q, res::AbstractKnnQueue)
    cost = searchtree(sat, ctx, q, sat.root, res)
    add_distance_evaluations!(ctx, cost)
    res
end
