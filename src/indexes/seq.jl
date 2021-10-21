# This file is a part of SimilaritySearch.jl

import Base: push!

export ExhaustiveSearch, search, push!

"""
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector)

Solves queries evaluating `dist` for the query and all elements in the dataset
"""
@with_kw struct ExhaustiveSearch{DistanceType<:PreMetric, DataType<:AbstractVector} <: AbstractSearchContext
    dist::DistanceType = SqL2Distance()
    db::DataType = Vector{Int32}[]
end

Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db) = ExhaustiveSearch(dist, db)

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

