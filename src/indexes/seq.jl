# This file is a part of SimilaritySearch.jl

import Base: push!

export ExhaustiveSearch, search

"""
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector)

Solves queries evaluating `dist` for the query and all elements in the dataset
"""
struct ExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchContext
    dist::DistanceType
    db::DataType
end

ExhaustiveSearch(dist::PreMetric, db::AbstractVector) = ExhaustiveSearch(dist, convert(AbstractDatabase, db))
ExhaustiveSearch(dist::PreMetric, db::Matrix) = ExhaustiveSearch(dist, convert(AbstractDatabase, db))
function ExhaustiveSearch(; dist=SqL2Distance(), db=VectorDatabase{Float32}())
    ExhaustiveSearch(dist, db)
end


Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db) = ExhaustiveSearch(dist, db)

"""
    search(seq::ExhaustiveSearch, q, res::AbstractKnnResult)

Solves the query evaluating all items in the given query.
"""
function search(seq::ExhaustiveSearch, q, res::AbstractKnnResult)
    st = initialstate(res)
    @inbounds for i in eachindex(seq)
        st = push!(res, st, i, evaluate(seq.dist, seq[i], q))
    end

    st
end

function Base.push!(seq::ExhaustiveSearch, u)
    push!(seq.db, u)
end