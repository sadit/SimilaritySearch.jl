# This file is a part of SimilaritySearch.jl

export ParallelExhaustiveSearch

"""
    struct ParallelExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex

    ParallelExhaustiveSearch(dist::PreMetric, db::AbstractDatabase)
    ParallelExhaustiveSearch(dist::PreMetric, db::AbstractVecOrMat)
    ParallelExhaustiveSearch(; dist=Dist.SqL2(), db=VectorDatabase{Float32}())

A brute-force exact index, like [`ExhaustiveSearch`](@ref), but that solves each query by evaluating `dist`
against every element of `db` in parallel (across `Threads.nthreads()` tasks), using an internal lock to
guard concurrent pushes into the result set. Useful as a gold-standard baseline for small-to-medium datasets
where parallelizing a single query is beneficial.

Note that this should not be used in conjunction with `searchbatch(...; parallel=true)` since they will
compete for the same thread pool.

# Arguments
- `dist`: the distance function
- `db`: the database being indexed, given either as an `AbstractDatabase` or as a raw vector/matrix
"""
struct ParallelExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
    lock::Threads.SpinLock
end


"""
    ParallelExhaustiveSearch(dist, db)

Keyword constructor for [`ParallelExhaustiveSearch`](@ref).

# Keyword Arguments
- `dist`: the distance function
- `db`: the database being indexed

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 8, 10^3))
Q = MatrixDatabase(rand(Float32, 8, 10))
P = ParallelExhaustiveSearch(; dist=Dist.SqL2(), db=X)
ctx = getcontext(P)

knns = searchbatch(P, ctx, Q, 8)  # (8, 10) matrix of `IdDist`, exact nearest neighbors
```
"""
function ParallelExhaustiveSearch(dist::PreMetric, db::AbstractDatabase)
    ParallelExhaustiveSearch(dist, db, Threads.SpinLock())
end


function getcontext(::ParallelExhaustiveSearch)
    GenericContext()
end

"""
    
    search(pex::ParallelExhaustiveSearch, ctx::GenericContext, q, res::AbstractKnn) -> res

Solves queries evaluating `dist` in parallel for the query and all elements in the dataset.


Solves query `q` by evaluating the distance between `q` and every item of the indexed database in
parallel, pushing each candidate into `res` under a lock.

# Arguments
- `pex`: the search structure
- `ctx`: the running context (unused by this method, kept for interface consistency)
- `q`: the query to solve
- `res`: the result set that receives the candidates
"""
function search(pex::ParallelExhaustiveSearch, ctx::GenericContext, q, res::AbstractKnn)
    dist = distance(pex)
    elock = pex.lock
    n = length(pex)
    minbatch = getminbatch(n)

    # NOTE: forced to scheduler=:default (not the global :static default) because this
    # per-query search is itself commonly invoked from *within* an outer @BATCHES-
    # parallelized per-query loop (e.g. searchbatch!/allknn/closestpair when `pex` is the
    # given index) -- native `:static` errors ("cannot be used concurrently or nested") in
    # that situation. This loop body only uses a shared lock (no Threads.threadid()-
    # indexed state), so :default's migratable tasks are safe here regardless of the
    # global scheduler.
    @BATCHES minbatch scheduler=:default for i in 1:n
        d = Dist.evaluate(dist, database(pex, i), q)
        try
            lock(elock)
            push_item!(res, i, d)
        finally
            unlock(elock)
        end
    end

    add_distance_evaluations!(res, length(pex))
    res
end

function push_item!(pex::ParallelExhaustiveSearch, ctx::GenericContext, u)
    push_item!(pex.db, u)
    LOG(ctx.logger, :push_item!, pex, ctx, length(pex), length(pex))
    pex
end

function append_items!(pex::ParallelExhaustiveSearch, ctx::GenericContext, u::AbstractDatabase)
    sp = length(pex)
    append_items!(pex.db, u)
    ep = length(pex)
    LOG(ctx.logger, :append_items!, pex, ctx, sp, ep)
    pex
end

function index!(pex::ParallelExhaustiveSearch, ::GenericContext)
    # do nothing
    LOG(ctx.logger, :index!, pex, ctx, length(pex), length(pex))
    pex
end
