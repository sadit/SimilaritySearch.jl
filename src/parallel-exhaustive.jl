# This file is a part of SimilaritySearch.jl

import Base: push!
export ParallelExhaustiveSearch, search


struct ParallelExhaustiveSearch{DistanceType<:SemiMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
    lock::Threads.SpinLock
end

ParallelExhaustiveSearch(dist::SemiMetric, db::AbstractVecOrMat) = ParallelExhaustiveSearch(dist, convert(AbstractDatabase, db))
ParallelExhaustiveSearch(dist::SemiMetric, db::AbstractDatabase) = ParallelExhaustiveSearch(dist, db, Threads.SpinLock())

"""
    ParallelExhaustiveSearch(; dist=SqL2Distance(), db=VectorDatabase{Float32}())

Solves queries evaluating `dist` in parallel for the query and all elements in the dataset.
Note that this should not be used in conjunction with `searchbatch(...; parallel=true)` since they will compete for resources.
"""
function ParallelExhaustiveSearch(; dist=SqL2Distance(), db=VectorDatabase{Float32}())
    ParallelExhaustiveSearch(dist, db, Threads.SpinLock())
end

function getcontext(::ParallelExhaustiveSearch)
    GenericContext()
end

Base.copy(ex::ParallelExhaustiveSearch; dist=ex.dist, db=ex.db) = ParallelExhaustiveSearch(dist, db, Threads.SpinLock())

"""
    search(ex::ParallelExhaustiveSearch, ctx::GenericContext, q, res)

Solves the query evaluating all items in the given query.

# Arguments
- `ex`: the search structure
- `q`: the query to solve
- `res`: the result set
- `ctx`: running ctx

"""
function search(ex::ParallelExhaustiveSearch, ctx::GenericContext, q, res)
    dist = distance(ex)
    elock = ex.lock
    minbatch = getminbatch(ctx.minbatch, length(ex))
    @batch minbatch=minbatch per=thread for i in eachindex(ex)
        d = evaluate(dist, database(ex, i), q)
        try
            lock(elock)
            push_item!(res, i, d)
        finally
            unlock(elock)
        end
    end

    res.cost = length(ex)
    res
end

function push_item!(ex::ParallelExhaustiveSearch, ctx::GenericContext, u)
    push_item!(ex.db, u)
    ctx.logger !== nothing && LOG(ctx.logger, push_item!, ex, length(ex))
    ex
end

function append_items!(ex::ParallelExhaustiveSearch, ctx::GenericContext, u::AbstractDatabase)
    sp = length(ex)
    push_item!(ex.db, u)
    ep = length(ex)
    ctx.logger !== nothing && LOG(ctx.logger, append_items!, ex, sp, ep, ep)
    ex
end

function index!(ex::ParallelExhaustiveSearch, ctx::GenericContext)
    # do nothing
    ex
end
