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

getcontext(index::ParallelExhaustiveSearch) = DEFAULT_CONTEXT[]
Base.copy(ex::ParallelExhaustiveSearch; dist=ex.dist, db=ex.db) = ParallelExhaustiveSearch(dist, db, Threads.SpinLock())

"""
    search(ex::ParallelExhaustiveSearch, context::GenericContext, q, res::KnnResult)

Solves the query evaluating all items in the given query.

# Arguments
- `ex`: the search structure
- `q`: the query to solve
- `res`: the result set
- `context`: running context

"""
function search(ex::ParallelExhaustiveSearch, context::GenericContext, q, res::KnnResult)
    dist = distance(ex)
    elock = ex.lock
    minbatch = getminbatch(context.minbatch, length(ex))
    @batch minbatch=minbatch per=thread for i in eachindex(ex)
        d = evaluate(dist, database(ex, i), q)
        try
            lock(elock)
            push_item!(res, i, d)
        finally
            unlock(elock)
        end
    end

    SearchResult(res, length(ex))
end

function push_item!(ex::ParallelExhaustiveSearch, context::GenericContext, u)
    push_item!(ex.db, u)
    context.logger !== nothing && LOG(context.logger, push_item!, ex, length(ex))
    ex
end

function append_items!(ex::ParallelExhaustiveSearch, context::GenericContext, u::AbstractDatabase)
    sp = length(ex)
    push_item!(ex.db, u)
    ep = length(ex)
    context.logger !== nothing && LOG(context.logger, append_items!, ex, sp, ep, ep)
    ex
end

function index!(ex::ParallelExhaustiveSearch, ctx::AbstractContext)
    # do nothing
    ex
end
