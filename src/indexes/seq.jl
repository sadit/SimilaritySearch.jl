# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base: push!
import StatsBase: fit

export ExhaustiveSearch, search, push!, fit, optimize!

"""
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector, knn::KnnResult)
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector, k::Integer)

Defines an exhaustive search
"""
struct ExhaustiveSearch{DistanceType<:PreMetric, DataType<:AbstractVector} <: AbstractSearchContext
    dist::DistanceType
    db::DataType
    res::KnnResult
end

StructTypes.StructType(::Type{<:ExhaustiveSearch}) = StructTypes.Struct()

ExhaustiveSearch(dist::PreMetric, db::AbstractVector; ksearch::Integer=10) =
    ExhaustiveSearch(dist, db, KnnResult(ksearch))

Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db, res=KnnResult(maxlength(seq.res))) = 
    ExhaustiveSearch(dist, db, res)

Base.string(seq::ExhaustiveSearch) =
    "{ExhaustiveSearch: dist=$(seq.dist), n=$(length(seq.db))}"

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

