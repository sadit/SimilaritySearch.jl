# This file is a part of SimilaritySearch.jl
#
# AsymmetricSearchGraph: the other AbstractSearchGraph, kept apart from SearchGraph's files
# so each graph is maintained on its own. It wraps a SearchGraph and owns everything about
# working with raw objects over a transformed storage: what the insertion loops query with
# (InsertionSource), the callbacks' raw queries, and the interface that refuses the symmetric
# operations. The estimator interface it navigates with is estimators.jl, included early.

export AsymmetricSearchGraph

"""
    InsertionSource(dist, db, items, offset)

What the insertion loops query the graph with, when that is not the database itself: object
`i` is `encodequery(dist, items[i - offset])` when it was just appended (its raw form,
prepared once for the query side of `dist`), and `db[i]` otherwise (what the database
stores). It is how an [`AsymmetricSearchGraph`](@ref) inserts raw objects into a graph over
their transformed storage; a `SearchGraph` always queries with the database. Indexing only;
the loops never iterate it, and each object is read once, so the preparation runs once per
inserted item.
"""
struct InsertionSource{D<:PreMetric,DB<:AbstractDatabase,ITEMS<:AbstractDatabase}
    dist::D
    db::DB
    items::ITEMS
    offset::Int
end

Base.@propagate_inbounds Base.getindex(s::InsertionSource, i::Integer) =
    i > s.offset ? encodequery(s.dist, s.items[i - s.offset]) : s.db[i]

"""
    AsymmetricSearchGraph(dist::PreMetric, db::AbstractDatabase; kwargs...)

A [`SearchGraph`](@ref) whose database stores a *transformed* form of the objects it is given
-- quantized codes, today, from a [`ScalarQuant.QuantDatabase`](@ref) that quantizes on
`push_item!`; the codes of an [`AbstractEstimator`](@ref) tomorrow -- and which inserts and
searches with the objects in their **raw** form, evaluated against the stored form by
`dist`. It is the asymmetric mode of the graph: each new item picks its neighbors by its
exact distance to the stored codes, so the quantization error is not baked into the edges,
and every query is evaluated the same way.

A [`SearchGraph`](@ref) over the same database is the symmetric mode, codes against codes on
both sides: cheaper at every step, and the only one available when the codes are all there
is (a database read back from storage). The mode is a property of the instance, not of a
call: this type never inserts or searches with the stored form, and a `SearchGraph` never
sees the raw objects. Which one pays was measured on the SISAP 2025 `ccnews` benchmark
(issue #86): the asymmetric edges are better below 8 bits (searched in exact precision,
0.901 against 0.882 of recall@10 at 4 bits and 0.940 against 0.884 at 2), but a query
evaluated against codes cannot cash that (0.81 at 4 bits and 0.65 at 2 with either graph),
so the asymmetric graph is the one to build when its distance re-evaluates what the codes
alone cannot resolve, and the symmetric one otherwise.

# Arguments
- `dist`: evaluates a raw object against a stored one, and says through
  [`encode`](@ref)`(dist, obj)` what is stored and through [`encodequery`](@ref)`(dist, q)`
  what a raw query becomes before being evaluated (once per query, once per inserted item). It is what the graph navigates with, at
  insertion and at query time, and what the hyperparameters callback tunes with -- on a
  sample of the raw items being inserted, not on stored codes. `ScalarQuant.SqL2`, `L1`,
  `NormCosine` and `Cosine` all take a plain vector against an `SQVec` and encode nothing
  (the `QuantDatabase` quantizes); an [`AbstractEstimator`](@ref) encodes its own codes and
  re-evaluates inside its `evaluate` when its error model says it must.
- `db`: the storage, any growable `AbstractDatabase` whose `push_item!` accepts what
  `encode(dist, obj)` produces
- remaining keywords go to the [`SearchGraph`](@ref) constructor (`adj`, `hints`, `algo`)

# Interface
[`append_items!`](@ref)`(g, ctx, items)` and [`push_item!`](@ref)`(g, ctx, item)` take raw
objects; [`search`](@ref)/[`searchbatch`](@ref) take raw queries;
[`optimize_index!`](@ref)`(g, ctx, kind; queries)` requires raw `queries`, since the index
holds no raw object to sample from. `index!(g, ctx)` and `rebuild` only have the stored form
to work with and are not available: they are the symmetric operations, use a `SearchGraph`
over the same database for them.

# Examples

```julia
using SimilaritySearch, SimilaritySearch.ScalarQuant

X = randn(Float32, 64, 10_000)
db = GlobalQuantDatabase(4, BlockMatrixDatabase(32, UInt8), extrema(X); dim=64)  # stores 4-bit codes
G = AsymmetricSearchGraph(ScalarQuant.SqL2(), db)         # SqL2 evaluates Float32 against codes
ctx = SearchGraphContext(; reporters=[])
append_items!(G, ctx, MatrixDatabase(X))                  # edges on Float32 vs codes; storage at 4 bits
search(G, ctx, randn(Float32, 64), knnqueue(KnnSorted, 10))
```

An estimator that stores a coarse code next to a finer one and re-evaluates from the fine
one only when the coarse estimate is close enough to matter:

```julia
struct TwoLevel <: AbstractEstimator
    coarse::GlobalQuantDatabase{2}    # the parameters; empty databases, used to quantize
    fine::GlobalQuantDatabase{8}
    τ::Float32                        # coarse estimates above it are far enough to trust
end
SimilaritySearch.encode(e::TwoLevel, v) = (ScalarQuant.quantize(e.coarse, v), ScalarQuant.quantize(e.fine, v))
function SimilaritySearch.evaluate(e::TwoLevel, q, stored)
    d = evaluate(ScalarQuant.SqL2(), q, stored[1])
    d > e.τ ? d : evaluate(ScalarQuant.SqL2(), q, stored[2])
end
SimilaritySearch.evaluate(::TwoLevel, a::Tuple, b::Tuple) = evaluate(ScalarQuant.SqL2(), a[2], b[2])   # candidates among themselves
G = AsymmetricSearchGraph(TwoLevel(c2, c8, 0.5f0), VectorDatabase(type=Tuple{SQVec{2,Vector{UInt8}},SQVec{8,Vector{UInt8}}}))
```

# Sketches and estimators
`index!(idx, ctx, :bitsketch)` builds a `SearchGraph`'s topology from sketches and keeps the
raw vectors: it is symmetric on the sketch side, code against code, which is the weaker
estimate. This type is the path for the asymmetric estimators, a raw query against a code:
the graph only evaluates the distance, and everything the model needs travels in the code it
encodes. Two ship with the package: [`ScalarQuant.SQEncoder`](@ref), the scalar quantizers
as a plain codification with no error model (an optional rotation in front), and the [`RaBitQ`](@ref) estimators, sign bits with
a per-object error bound and, in `RaBitQ.RaBitQRefined`, a fallback re-evaluated inside the
estimate when that bound cannot rule an object out.
"""
struct AsymmetricSearchGraph{G<:SearchGraph} <: AbstractSearchGraph
    graph::G
end

function AsymmetricSearchGraph(dist::PreMetric, db::AbstractDatabase; kwargs...)
    AsymmetricSearchGraph(SearchGraph(dist, db; kwargs...))
end

@inline database(g::AsymmetricSearchGraph) = database(g.graph)
@inline distance(g::AsymmetricSearchGraph) = distance(g.graph)
@inline Base.length(g::AsymmetricSearchGraph) = length(g.graph)
ismember(g::AsymmetricSearchGraph, id::Integer) = ismember(g.graph, id)
representative(g::AsymmetricSearchGraph, id::Integer) = representative(g.graph, id)
members(g::AsymmetricSearchGraph, id::Integer) = members(g.graph, id)
# the expansion evaluates raw against stored, so the raw query is prepared once, as in `search`
expand(g::AsymmetricSearchGraph, q, res::AbstractMetricQueue) = expand(g.graph, encodequery(distance(g), q), res)
expand(g::AsymmetricSearchGraph, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32}) = expand(g.graph, encodequery(distance(g), q), ids, dists)
expand!(g::AsymmetricSearchGraph, q, res::AbstractMetricQueue) = expand!(g.graph, encodequery(distance(g), q), res)
expand!(g::AsymmetricSearchGraph, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32}) = expand!(g.graph, encodequery(distance(g), q), ids, dists)
expand!(g::AsymmetricSearchGraph, Q::AbstractDatabase, knns::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}) =
    expand!(g.graph, VectorDatabase([encodequery(distance(g), q) for q in Q]), knns, dists)
_expand!(g::AsymmetricSearchGraph, q, res::AbstractMetricQueue) = _expand!(g.graph, encodequery(distance(g), q), res)

function Base.show(io::IO, g::AsymmetricSearchGraph; prefix="", indent="  ")
    println(io, prefix, "AsymmetricSearchGraph (raw objects in, stored as `encode(dist, obj)`):")
    Base.show(io, g.graph; prefix=prefix * indent, indent)
end

"""
    search(g::AsymmetricSearchGraph, ctx::SearchGraphContext, q, res::AbstractMetricQueue)

Searches with the raw query `q`, prepared once by `encodequery(distance(g), q)` and evaluated
against the stored form by the graph's distance.
"""
search(g::AsymmetricSearchGraph, ctx::SearchGraphContext, q, res::AbstractMetricQueue) =
    search(g.graph, ctx, encodequery(distance(g), q), res)

"""
    append_items!(g::AsymmetricSearchGraph, ctx::SearchGraphContext, items::AbstractDatabase)

Appends the raw `items`: each is stored as `encode(distance(g), item)`, and then indexed with
its raw form as the query that picks its neighbors (see [`InsertionSource`](@ref)). Parallel
or sequential as `ctx` says, like a `SearchGraph`'s.
"""
function append_items!(g::AsymmetricSearchGraph, ctx::SearchGraphContext, items::AbstractDatabase)
    db = database(g)
    dist = distance(g)
    offset = length(db)
    for item in items
        push_item!(db, encode(dist, item))
    end

    # the pool must carry prepared raw objects: `db` holds codes, which is not what the
    # distance takes on the query side (see `tuningpool`)
    ctx = tuningpool(ctx, offset + 1, offset + length(items), k -> encodequery(dist, items[k - offset]))
    _index!(g.graph, ctx, InsertionSource(dist, db, items, offset))
    g
end

"""
    push_item!(g::AsymmetricSearchGraph, ctx::SearchGraphContext, item)

Appends one raw `item`: stored as `encode(distance(g), item)`, indexed with its raw form.
"""
function push_item!(g::AsymmetricSearchGraph, ctx::SearchGraphContext, item)
    db = database(g)
    offset = length(db)
    dist = distance(g)
    push_item!(db, encode(dist, item))
    _index!(g.graph, ctx, InsertionSource(dist, db, VectorDatabase([item]), offset))
    g
end

function index!(::AsymmetricSearchGraph, ::SearchGraphContext)
    throw(ArgumentError("index!: an AsymmetricSearchGraph indexes the raw objects it is given through append_items!/push_item!; the stored form alone can only be indexed symmetrically, with a SearchGraph over the same database"))
end

function rebuild(::AsymmetricSearchGraph, ::SearchGraphContext; kwargs...)
    throw(ArgumentError("rebuild: an AsymmetricSearchGraph holds only the stored form of its objects, and rebuilding from it is the symmetric operation; use a SearchGraph over the same database"))
end

"""
    optimize_index!(g::AsymmetricSearchGraph, ctx::SearchGraphContext, kind=MinRecall(0.9); queries, kwargs...)

Tunes the graph's search parameters on raw `queries`, which are required (the index holds no
raw object to sample from) and prepared once by `encodequery`.
"""
function optimize_index!(g::AsymmetricSearchGraph, ctx::SearchGraphContext, kind::ErrorFunction=MinRecall(0.9); queries=nothing, kwargs...)
    queries === nothing &&
        throw(ArgumentError("optimize_index!: an AsymmetricSearchGraph must be tuned with raw `queries`; it stores only the transformed objects, which are not what its distance takes on the query side"))
    dist = distance(g)
    optimize_index!(g.graph, ctx, kind; queries=VectorDatabase([encodequery(dist, q) for q in queries]), kwargs...)
end
