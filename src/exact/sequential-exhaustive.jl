# This file is a part of SimilaritySearch.jl

export ExhaustiveSearch, search

"""
    struct ExhaustiveSearch{DistanceType<:PreMetric,DataType<:AbstractDatabase} <: AbstractSearchIndex

    ExhaustiveSearch(dist::PreMetric, db::AbstractDatabase)

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


getcontext(::ExhaustiveSearch) = GenericContext()

Base.copy(seq::ExhaustiveSearch; dist=seq.dist, db=seq.db) = ExhaustiveSearch(dist, db)

function push_item!(seq::ExhaustiveSearch, ctx::GenericContext, u)
    push_item!(seq.db, u)
    n = length(seq)
    OBSERVE(ctx, :add!, seq, n, n)
    @inform ctx "add! sp=$n ep=$n" index=seq
    seq
end

function append_items!(seq::ExhaustiveSearch, ctx::GenericContext, u::AbstractDatabase)
    sp = length(seq) + 1
    append_items!(seq.db, u)
    ep = length(seq)
    if ep >= sp
        OBSERVE(ctx, :add!, seq, sp, ep)
        @inform ctx "add! sp=$sp ep=$ep" index=seq
    end
    seq
end

function index!(seq::ExhaustiveSearch, ctx::AbstractContext)
    # a no-op: `db` already *is* the index, there is no separate structure to build. Nothing
    # structural happened, so this is a message and not an event -- see `OBSERVE`'s contract.
    @inform ctx "index! is a no-op on $(typeof(seq)): db already is the index" index=seq
    seq
end

"""
    search(seq::ExhaustiveSearch, ctx::AbstractContext, q, res::AbstractMetricQueue) -> res

Solves query `q` by sequentially evaluating the distance between `q` and every item of the indexed
database, pushing each candidate into `res`.

# Arguments
- `seq`: the exhaustive search index
- `ctx`: the running context, charged with the distance-evaluation count
- `q`: the query to solve
- `res`: the result set that receives the candidates
"""
@inline function search(seq::ExhaustiveSearch, ctx::AbstractContext, q, res::AbstractMetricQueue)
    dist = distance(seq)
    db = database(seq)
    n = length(db)
    i = 0
    while (i += 1) <= n
        d = Dist.evaluate(dist, db[i], q)
        push_item!(res, i, d)
    end

    add_distance_evaluations!(ctx, n)
    res
end
