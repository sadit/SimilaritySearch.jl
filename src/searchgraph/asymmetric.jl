# This file is a part of SimilaritySearch.jl

export AsymmetricSearchGraph, AbstractEstimator

"""
    encode(dist::PreMetric, obj)

What an [`AsymmetricSearchGraph`](@ref) stores for the raw `obj` under `dist`: the form
`dist` evaluates a raw query against. The default returns `obj` itself, for a storage that
transforms what it stores on its own, as a [`ScalarQuant.QuantDatabase`](@ref) does under
the scalar quantizers' distances. An [`AbstractEstimator`](@ref) whose codes the storage
does not produce overrides it.
"""
encode(::PreMetric, obj) = obj

"""
    encodequery(dist::PreMetric, q)

What an [`AsymmetricSearchGraph`](@ref) evaluates `dist` with on the query side for the raw
`q`: the form `evaluate(dist, encodequery(dist, q), stored)` takes. The default returns `q`
itself. An [`AbstractEstimator`](@ref) whose evaluation needs the query prepared once --
rotated, projected, quantized on the query side -- overrides it, and the graph applies it
once per query and once per inserted item (which is the query of its own neighborhood
search), never per evaluation.
"""
encodequery(::PreMetric, q) = q

"""
    abstract type AbstractEstimator <: PreMetric end

A distance that is an estimator: it evaluates a raw query against an *encoded* object, and
may carry an error of its own. It is one plain type whose parameters are fields, so a graph
and everything that gives its codes meaning serialize together; nothing in it is a closure.

An estimator implements:

- `encode(est, obj)`: what is stored for the raw `obj`, the code plus whatever the
  estimator keeps with it (a norm, a correction term, a finer code);
- `encodequery(est, q)`, when the query side needs preparing: what `evaluate` takes as its
  query for the raw `q`. A rotation, for one -- applying it inside `evaluate` would cost
  `D^2` per pair against the `D` of the estimate, so the graph applies it once per query
  and once per inserted item. The default is the identity;
- `evaluate(est, q, stored)`: the distance between the raw query `q` and a stored object,
  in that order -- every index here evaluates its query first. It receives everything the model has -- the raw query, the encoded object with what was
  kept beside the code, and the estimator's own parameters -- so a model that can bound its
  error re-evaluates *inside* the evaluation when it must, and returns the distance it
  stands behind. The graph only ever evaluates the distance; whether that was an estimate,
  a corrected estimate or a re-evaluation is the estimator's business.
- `evaluate(est, a, b)` between two **stored** objects as well: the neighborhood filters
  (`SatNeighborhood` and its relatives) compare a new item's candidates among themselves to
  decide which edges to keep, and those candidates are stored objects. That is the
  symmetric estimate, code against code, and it only shapes the edges; the scalar
  quantizers' distances already have it.

The no-op estimator is a plain distance: `ScalarQuant.SqL2()` against an `SQVec` evaluates
the query against the codes and nothing needs correcting. Any `PreMetric` works as the
distance of an asymmetric graph; this type is the documented home for the ones that encode.
"""
abstract type AbstractEstimator <: PreMetric end

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
estimate. This type is the path for the asymmetric estimators -- a raw query against a
sketch, RaBitQ-style codes -- once written as an [`AbstractEstimator`](@ref): the graph only
evaluates the distance, and everything the model needs travels in the code it encodes.
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
    rawqueries(dist::PreMetric, ctx::SearchGraphContext, items::AbstractDatabase) -> SearchGraphContext

The context an asymmetric insertion runs its callbacks with: the same one, except that an
[`OptimizeParameters`](@ref) callback left to sample its own queries samples them from the raw
`items` being inserted, prepared by `encodequery`, rather than from the stored codes, which
is not what the distance takes on the query side. A callback given explicit `queries` keeps
them.
"""
function rawqueries(dist::PreMetric, ctx::SearchGraphContext, items::AbstractDatabase)
    cb = ctx.hyperparameters_callback
    (cb isa OptimizeParameters && cb.queries === nothing && length(items) > 0) || return ctx
    sample = VectorDatabase([encodequery(dist, x) for x in rand(items, min(Int(cb.numqueries), length(items)))])
    cb2 = OptimizeParameters(cb.kind, cb.initialpopulation, cb.maxiters, cb.bsize, cb.mutbsize, cb.crossbsize,
                             cb.maxpopulation, cb.ksearch, sample, cb.numqueries, cb.space)
    @set ctx.hyperparameters_callback = cb2
end

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

    _index!(g.graph, rawqueries(dist, ctx, items), InsertionSource(dist, db, items, offset))
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
