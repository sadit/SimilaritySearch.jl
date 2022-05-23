# This file is a part of SimilaritySearch.jl

import Base: push!
export ParallelExhaustiveSearch, search


struct ParallelExhaustiveSearch{DistanceType<:SemiMetric,DataType<:AbstractDatabase} <: AbstractSearchContext
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


getpools(index::ParallelExhaustiveSearch) = nothing
Base.copy(ex::ParallelExhaustiveSearch; dist=ex.dist, db=ex.db) = ParallelExhaustiveSearch(dist, db, Threads.SpinLock())

"""
    search(ex::ParallelExhaustiveSearch, q, res::KnnResult; pools=nothing)

Solves the query evaluating all items in the given query.
"""
function search(ex::ParallelExhaustiveSearch, q, res::KnnResult; pools=nothing)
    dist = ex.dist
    elock = ex.lock
    Threads.@threads for i in eachindex(ex)
        d = evaluate(dist, ex[i], q)
        try
            lock(elock)
            push!(res, i, d)
        finally
            unlock(elock)
        end
    end

    (res=res, cost=length(ex))
end

function Base.push!(ex::ParallelExhaustiveSearch, u)
    push!(ex.db, u)
end