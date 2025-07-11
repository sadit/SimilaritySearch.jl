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

distance(seq::ExhaustiveSearch) = seq.dist
database(seq::ExhaustiveSearch) = seq.db
database(seq::ExhaustiveSearch, i::Integer) = seq.db[i]
Base.length(seq::ExhaustiveSearch) = length(seq.db)

#ExhaustiveSearch(dist::SemiMetric, db::AbstractVector) = ExhaustiveSearch(dist, convert(AbstractDatabase, db))
#ExhaustiveSearch(dist::SemiMetric, db::Matrix) = ExhaustiveSearch(dist, convert(AbstractDatabase, db))
function ExhaustiveSearch(; dist=SqL2Distance(), db=VectorDatabase{Float32}())
    ExhaustiveSearch(dist, db)
end

getcontext(::ExhaustiveSearch) = GenericContext()

Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db) = ExhaustiveSearch(dist, db)

function push_item!(seq::ExhaustiveSearch, ctx::GenericContext, u)
    push_item!(seq.db, u)
    ctx.logger !== nothing && LOG(ctx.logger, push_item!, seq, length(seq))
    seq
end

function append_items!(seq::ExhaustiveSearch, ctx::GenericContext, u::AbstractDatabase)
    sp = length(seq)
    append_items!(seq.db, u)
    ep = length(seq)
    ctx.logger !== nothing && LOG(ctx.logger, append_items!, seq, sp, ep, ep)
    seq
end

index!(seq::ExhaustiveSearch, ::GenericContext) = seq # do nothing

"""
    search(seq::ExhaustiveSearch, ctx::GenericContext, q, res)

Solves the query evaluating all items in the given query.
"""
@inline function search(seq::ExhaustiveSearch, ::GenericContext, q, res::AbstractKnn)
    dist = distance(seq)
    for i in eachindex(seq)
        d = evaluate(dist, database(seq, i), q)
        push_item!(res, i, d)
    end

    res.cost = length(seq)
    res
end

