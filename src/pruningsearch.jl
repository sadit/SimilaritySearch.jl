# This file is a part of SimilaritySearch.jl

export PruningSearch, PruningSearchSpace

"""
    PruningConfig(factor)

The configuration of a [`PruningSearch`](@ref): the factor that scales the covering radius in
every filter of the wrapped index.
"""
struct PruningConfig
    factor::Float32
end

const PrunableIndex = Union{ANNI,NANNI,EPTable,NEP}

"""
    PruningSearch(index; factor=1)

An exact metric index searched with the stretched triangle inequality of Chávez and Navarro,
*Probabilistic proximity search: fighting the curse of dimensionality in metric spaces*,
Information Processing Letters 85 (2003) 39--46: every filter of `index` discards an object or
a cell when its lower bound exceeds `factor` times the covering radius instead of the radius
itself. `factor = 1` is the exact search; below it the search evaluates fewer distances and can
miss neighbors. `index` is an [`ANNI`](@ref), [`NANNI`](@ref), [`EPTable`](@ref) or [`NEP`](@ref),
already built.

Why it works: in high dimension the lower bounds that pivots give are a small fraction of the
true distances, so an exact index evaluates almost everything, and the quotient
`LB(q, u) / d(q, u)` stays below a small value for most pairs. A neighbor `u` is lost only when
`LB(q, u) > factor · r ≥ factor · d(q, u)`, so the fraction lost is at most the fraction of pairs
whose quotient exceeds `factor`; the factor of a target recall is read off that distribution, and
it is lower the higher the intrinsic dimension.

The factor is a hyperparameter like the `Δ` of [`BeamSearch`](@ref): [`optimize_index!`](@ref)
tunes it for a goal such as [`MinRecall`](@ref), minimizing distance evaluations, over
[`PruningSearchSpace`](@ref). The structure of `index` (its number of centers or pivots) stays
the one its own construction chose for the exact search.

For a permuted index (`permuteanni` and the like), tune a `PruningSearch` over the inner index
and wrap it back: `PermutedSearchIndex(; index=PruningSearch(P.index), P.π, P.π′)`.

# Examples

```julia
db = MatrixDatabase(rand(Float32, 16, 10_000))
nanni = NANNI(Dist.L2(), db)
index!(nanni, GenericContext())
P = PruningSearch(nanni)
optimize_index!(P, GenericContext(), MinRecall(0.9))
search(P, GenericContext(), rand(Float32, 16), knnqueue(KnnSorted, 10))
```
"""
struct PruningSearch{IndexType<:PrunableIndex} <: AbstractSearchIndex
    config::Base.RefValue{PruningConfig}
    index::IndexType
end

function PruningSearch(index::PrunableIndex; factor::Real=1f0)
    0 < factor <= 1 || throw(ArgumentError("PruningSearch: factor=$factor must lie in (0, 1]"))
    PruningSearch(Ref(PruningConfig(factor)), index)
end

@inline database(p::PruningSearch) = database(p.index)
@inline database(p::PruningSearch, i) = database(p.index, i)
@inline distance(p::PruningSearch) = distance(p.index)
@inline Base.length(p::PruningSearch) = length(p.index)

function Base.show(io::IO, p::PruningSearch)
    print(io, "PruningSearch(factor=", p.config[].factor, ") over ")
    show(io, p.index)
end

_prunedcount(index::Union{ANNI,NANNI}, q, res, factor::Float32) = Anni._searchcount(index, q, res, factor)
_prunedcount(index::Union{EPTable,NEP}, q, res, factor::Float32) = ExtremePivots._searchcount(index, q, res, factor)

"""
    search(p::PruningSearch, ctx::AbstractContext, q, res::AbstractMetricQueue)

Solves `q` with the wrapped index and the stretched filters of `p`'s current factor.
"""
function search(p::PruningSearch, ctx::AbstractContext, q, res::AbstractMetricQueue)
    add_distance_evaluations!(ctx, _prunedcount(p.index, q, res, p.config[].factor))
    res
end

## optimize_index! integration

"""
    PruningSearchSpace(; factor=0.02:0.02:0.9, factor_scale=(...)) <: AbstractSolutionSpace

Autotuning space of [`PruningSearch`](@ref)'s `factor`. The factor a target needs falls with the
intrinsic dimension, so the initial range is wide and low: for the nearest of 256--4096 objects
of five SISAP 2025 embedding collections the 0.9 quantile of the quotient of the lower bound over
the true distance was 0.11--0.30, while on 5000 uniform vectors of dimension 16 a recall of 0.9
at k = 10 took factors of 0.5--0.56. Mutations scale the factor by `s`, up to `upper = 1`, the
exact search.
"""
@kwdef struct PruningSearchSpace <: AbstractSolutionSpace
    factor = 0.02:0.02:0.9
    factor_scale = (s=1.15, p1=0.75, p2=0.5, lower=0.005, upper=1.0)
end

Base.hash(c::PruningConfig, h::UInt) = hash(round(c.factor, digits=4), h)
Base.isequal(a::PruningConfig, b::PruningConfig) = a.factor == b.factor
Base.eltype(::PruningSearchSpace) = PruningConfig
Base.rand(rng::AbstractRNG, space::PruningSearchSpace) = PruningConfig(rand(rng, space.factor))

combine(a::PruningConfig, b::PruningConfig) = PruningConfig((a.factor + b.factor) / 2)
mutate(space::PruningSearchSpace, c::PruningConfig, iter) =
    PruningConfig(SearchModels.scale(c.factor; space.factor_scale...))

optimization_space(::PruningSearch) = PruningSearchSpace()
setconfig!(config::PruningConfig, p::PruningSearch, perf) = (p.config[] = config)

"""
    _MaskedQueue(res, mask)

The result queue of a tuning query that lives in the index: the objects in `mask` (the query
itself, see [`tuningmask`](@ref)) are evaluated as usual but never enter `res`, so they neither
answer the query nor shrink the radius the filters prune with. An internal query would
otherwise find itself at distance 0 and prune with a radius no external query has.
"""
struct _MaskedQueue{Q<:AbstractMetricQueue} <: AbstractMetricQueue
    res::Q
    mask::Vector{UInt32}
end

@inline push_item!(m::_MaskedQueue, id::Integer, d::Real) = (UInt32(id) in m.mask || push_item!(m.res, id, d); m)
@inline PQueue.covradius(m::_MaskedQueue) = PQueue.covradius(m.res)

function runconfig(config::PruningConfig, p::PruningSearch, ctx::AbstractContext, q, mask::AbstractVector{UInt32}, res::AbstractKnnQueue)
    out = isempty(mask) ? res : _MaskedQueue(res, Vector{UInt32}(mask))
    add_distance_evaluations!(ctx, _prunedcount(p.index, q, out, config.factor))
    res
end

runconfig(config::PruningConfig, p::PruningSearch, ctx::AbstractContext, q, res::AbstractKnnQueue) =
    runconfig(config, p, ctx, q, EMPTY_MASK, res)

_pivots(index::Union{ANNI,NANNI}) = index.centers
_pivots(index::Union{EPTable,NEP}) = index.pivots

# Left to sample its own queries, the tuning draws them among the objects that are not pivots or
# centers of the wrapped index. A pivot as a query sits at distance 0 from it, so its own filters
# bound every other object by its true distance to that pivot, and in the cell indexes the
# hyperplane rule closes on d_min = 0: a configuration no external query is in. Measured on 5000
# uniform vectors of dimension 16 (NANNI, 640 centers, k = 10, factor 0.6): recall 0.39 for the
# queries that were centers, 0.975 for the other internal ones, 0.982 for external queries; the
# 12 centers among 64 tuning queries pulled the tuned factor up until the recall on external
# queries was 0.99 for a target of 0.9.
function optimize_index!(p::PruningSearch, ctx::AbstractContext, kind::ErrorFunction=MinRecall(0.9);
                         queries=nothing, queries_identifiers=nothing, numqueries=64,
                         rng=Random.default_rng(), kwargs...)
    if queries === nothing && queries_identifiers === nothing
        ispivot = falses(length(p))
        ispivot[_pivots(p.index)] .= true
        queries_identifiers = unique(rand(rng, findall(!, ispivot), numqueries))
    end
    invoke(optimize_index!, Tuple{AbstractSearchIndex,AbstractContext,ErrorFunction}, p, ctx, kind;
           queries, queries_identifiers, numqueries, rng, kwargs...)
end
