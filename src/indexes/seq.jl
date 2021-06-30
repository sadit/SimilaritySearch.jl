# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base: push!
import StatsBase: fit

export ExhaustiveSearch, search, push!, fit, optimize!

"""
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector, knn::KnnResult)
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector, k::Integer)
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector; ksearch::Integer=10)    

Solves queries evaluating `dist` for the query and all elements in the dataset.
"""
@with_kw struct ExhaustiveSearch{DistanceType<:PreMetric, DataType<:AbstractVector} <: AbstractSearchContext
    dist::DistanceType = SqL2Distance()
    db::DataType = Vector{Int32}[]
    res::KnnResult = KnnResult(10)
end

ExhaustiveSearch(dist::PreMetric, db::AbstractVector; ksearch::Integer=10) =
    ExhaustiveSearch(dist, db, KnnResult(ksearch))

Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db, res=KnnResult(maxlength(seq.res))) = 
    ExhaustiveSearch(dist, db, res)

"""
    search(seq::ExhaustiveSearch, q, res::KnnResult)

Solves the query evaluating all items in the given query.

By default, it uses an internal result buffer;
multithreading applications must duplicate specify another `res` object.
"""
function search(seq::ExhaustiveSearch, q, res::KnnResult)
    db = seq.db

    @inbounds for i in eachindex(db)
        push!(res, i, evaluate(seq.dist, db[i], q))
    end

    res
end

