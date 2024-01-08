# This file is a part of SimilaritySearch.jl

import Base: push!

export ExhaustiveSearch, search

"""
    ExhaustiveSearch(dist::SemiMetric, db::AbstractVector)

Solves queries evaluating `dist` for the query and all elements in the dataset
"""
struct ExhaustiveSearch{DistanceType<:SemiMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
end

ExhaustiveSearch(dist::SemiMetric, db::AbstractVector) = ExhaustiveSearch(dist, convert(AbstractDatabase, db))
ExhaustiveSearch(dist::SemiMetric, db::Matrix) = ExhaustiveSearch(dist, convert(AbstractDatabase, db))
function ExhaustiveSearch(; dist=SqL2Distance(), db=VectorDatabase{Float32}())
    ExhaustiveSearch(dist, db)
end

getcontext(index::ExhaustiveSearch) = GenericContext()
Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db) = ExhaustiveSearch(dist, db)

function push_item!(seq::ExhaustiveSearch, context::GenericContext, u)
    push_item!(seq.db, u)
end

function append_items!(seq::ExhaustiveSearch, context::GenericContext, u::AbstractDatabase)
    append_items!(seq.db, u)
end

"""
    search(seq::ExhaustiveSearch, context::AbstractContext, q, res::KnnResult)

Solves the query evaluating all items in the given query.
"""
function search(seq::ExhaustiveSearch, ctx::AbstractContext, q, res::KnnResult)
    dist = distance(seq)
    @inbounds for i in eachindex(seq)
        d = evaluate(dist, database(seq, i), q)
        push_item!(res, i, d)
    end

    SearchResult(res, length(seq))
end

