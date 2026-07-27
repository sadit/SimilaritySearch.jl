# This file is a part of SimilaritySearch.jl

import Base: push!

export ExhaustiveSearch, search

"""
    struct ExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex

    ExhaustiveSearch(dist::PreMetric, db::AbstractDatabase)
    ExhaustiveSearch(; dist=Dist.SqL2(), db=VectorDatabase{Float32}())

A brute-force (sequential) exact index that solves queries by evaluating `dist` between the query and every
element in `db`. Useful as a gold-standard baseline or for small datasets where an approximate index is not
worth its construction cost.

# Arguments
- `dist`: the distance function
- `db`: the database being indexed
"""
struct ExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex
    dist::DistanceType
    db::DataType
end

@inline distance(seq::ExhaustiveSearch) = seq.dist
@inline database(seq::ExhaustiveSearch) = seq.db
@inline database(seq::ExhaustiveSearch, i::Integer) = seq.db[i]
@inline Base.length(seq::ExhaustiveSearch) = length(seq.db)

"""
    ExhaustiveSearch(; dist=Dist.SqL2(), db=VectorDatabase{Float32}())

Keyword constructor for [`ExhaustiveSearch`](@ref).

# Keyword Arguments
- `dist`: the distance function
- `db`: the database being indexed

# Examples

```julia
using SimilaritySearch

X = MatrixDatabase(rand(Float32, 8, 10^3))
Q = MatrixDatabase(rand(Float32, 8, 10))
E = ExhaustiveSearch(; dist=Dist.SqL2(), db=X)
ctx = getcontext(E)

knns = searchbatch(E, ctx, Q, 8)  # (8, 10) matrix of `IdDist`, exact nearest neighbors
```
"""
function ExhaustiveSearch(; dist=Dist.SqL2(), db=VectorDatabase{Float32}())
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
    search(seq::ExhaustiveSearch, ctx::AbstractContext, q, res::AbstractKnn) -> res

Solves query `q` by sequentially evaluating the distance between `q` and every item of the indexed
database, pushing each candidate into `res`.

# Arguments
- `seq`: the exhaustive search index
- `ctx`: the running context (unused by this method, kept for interface consistency)
- `q`: the query to solve
- `res`: the result set that receives the candidates
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

