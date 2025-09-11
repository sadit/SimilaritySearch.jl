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

Base.copy(pex::ParallelExhaustiveSearch; dist=pex.dist, db=pex.db) = ParallelExhaustiveSearch(dist, db, Threads.SpinLock())

"""
    search(pex::ParallelExhaustiveSearch, ctx::GenericContext, q, res)

Solves the query evaluating all items in the given query.

# Arguments
- `pex`: the search structure
- `q`: the query to solve
- `res`: the result set
- `ctx`: running ctx

"""
function search(pex::ParallelExhaustiveSearch, ctx::GenericContext, q, res::AbstractKnn)
    dist = distance(pex)
    elock = pex.lock
    minbatch = getminbatch(ctx, length(pex))
    
    @batch minbatch=minbatch per=thread for i in eachindex(pex)
        d = evaluate(dist, database(pex, i), q)
        try
            lock(elock)
            push_item!(res, i, d)
        finally
            unlock(elock)
        end
    end
    
    res.costevals = length(pex)
    res.costblocks = 0
    res
end

function push_item!(pex::ParallelExhaustiveSearch, ctx::GenericContext, u)
    push_item!(pex.db, u)
    LOG(ctx.logger, :push_item!, pex, ctx, length(pex),  length(pex))
    pex
end

function append_items!(pex::ParallelExhaustiveSearch, ctx::GenericContext, u::AbstractDatabase)
    sp = length(pex)
    append_items!(pex.db, u)
    ep = length(pex)
    LOG(ctx.logger, :append_items!, pex, ctx, sp, e)
    pex
end
    
function index!(pex::ParallelExhaustiveSearch, ::GenericContext)
    # do nothing
    LOG(ctx.logger, :index!, pex, ctx, length(pex), length(pex))
    pex
end
