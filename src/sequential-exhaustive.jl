# This file is a part of SimilaritySearch.jl

import Base: push!

export ExhaustiveSearch, search

"""
    ExhaustiveSearch(dist::PreMetric, db::AbstractVector)

Solves queries evaluating `dist` for the query and all elements in the dataset
"""
struct ExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
end

@inline distance(seq::ExhaustiveSearch) = seq.dist
@inline database(seq::ExhaustiveSearch) = seq.db
@inline database(seq::ExhaustiveSearch, i::Integer) = seq.db[i]
@inline Base.length(seq::ExhaustiveSearch) = length(seq.db)

function ExhaustiveSearch(; dist=SqL2Distance(), db=VectorDatabase{Float32}())
    ExhaustiveSearch(dist, db)
end

getcontext(::ExhaustiveSearch) = GenericContext()

Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db) = ExhaustiveSearch(dist, db)

function push_item!(seq::ExhaustiveSearch, ctx::GenericContext, u)
    push_item!(seq.db, u)
    n = length(seq)
    LOG(ctx.logger, :push_item!, seq, ctx, n, n)
    seq
end

function append_items!(seq::ExhaustiveSearch, ctx::GenericContext, u::AbstractDatabase)
    sp = length(seq)
    append_items!(seq.db, u)
    ep = length(seq)
    LOG(ctx.logger, :append_items!, seq, ctx, sp, ep)
    seq
end

function index!(seq::ExhaustiveSearch, ::AbstractContext)
    # do nothing
    n = length(seq)
    LOG(ctx.logger, :index!, seq, ctx, n, n)
    seq
end

"""
    search(seq::ExhaustiveSearch, ctx::AbstractContext, q, res)

Solves the query evaluating all items in the given query.
"""
@inline function search(seq::ExhaustiveSearch, ::AbstractContext, q, res::AbstractKnn)
    dist = distance(seq)
    db = database(seq)
    n = length(db)
    i = 0
    while (i += 1) <= n
        d = evaluate(dist, db[i], q)
        push_item!(res, i, d)
    end

    add_distance_evaluations!(res, n)
    res
end

