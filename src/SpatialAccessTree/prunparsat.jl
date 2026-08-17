# This file is part of SimilaritySearch.jl

struct PrunParSatConfig
    depth::Int32
    factor::Float32
end
PrunParSatConfig(depth::Integer, factor::Real) = PrunParSatConfig(convert(Int32, depth), convert(Float32, factor))

struct PrunParSat{ST<:Sat} <: AbstractSearchIndex
    sat::ST
    parents::Vector{UInt32}
    config::Ref{PrunParSatConfig}
end

"""
    PrunParSat(sat::Sat; depth=3, factor=0.95)

`allknn`-oriented pruning search: each query that is one of `sat`'s own indexed points
starts its search from its `depth`-th ancestor instead of the tree root, drastically
reducing cost for the all-vs-all setting. Queries that are *not* one of `sat`'s own points
(plain [`search`](@ref)) fall back to exact SAT search, since no ancestor is known for them.
"""
PrunParSat(sat::Sat; depth::Integer=3, factor::Real=0.95) =
    PrunParSat(sat, compute_parents(sat), Ref(PrunParSatConfig(depth, factor)))

@inline database(p::PrunParSat) = database(p.sat)
@inline database(p::PrunParSat, i) = database(p.sat, i)
@inline distance(p::PrunParSat) = distance(p.sat)
@inline Base.length(p::PrunParSat) = length(p.sat)

function Base.show(io::IO, p::PrunParSat)
    println(io, typeof(p), " config=", p.config[])
    show(io, p.sat)
end

"""
    ith_par(parents, c, ith)

Retrieves the `ith`-th ancestor of `c` by following `parents` upward `ith` times.
"""
@inline function ith_par(parents::Vector{UInt32}, c::Integer, ith::Integer)
    j = 0
    @inbounds while j < ith
        c = parents[c]
        j += 1
    end
    c
end

"""
    compute_parents(sat::Sat) -> Vector{UInt32}

Computes the parent of every element in the `sat` tree (the root is its own parent).
"""
function compute_parents(sat::Sat)
    n = length(sat)
    P = Vector{UInt32}(undef, n)
    P[sat.root] = sat.root
    _compute_parents!(sat, P, sat.root)
    P
end

function _compute_parents!(sat::Sat, P::Vector{UInt32}, r::UInt32)
    C = sat.children[r]
    C === nothing && return
    for c in C
        P[c] = r
        sat.children[c] !== nothing && _compute_parents!(sat, P, c)
    end
end

function travelsat!(psat::PrunParSat, ctx::SatContext, i::Integer, res::AbstractKnnQueue,
                     c::PrunParSatConfig=psat.config[])
    q = database(psat, i)
    p = ith_par(psat.parents, i, c.depth)
    pruningsearchtree!(psat.sat, ctx, q, p, res, c.factor)
end

# Generic (out-of-sample) query: no ancestor is known, fall back to exact SAT search.
search(psat::PrunParSat, ctx::SatContext, q, res::AbstractKnnQueue) = search(psat.sat, ctx, q, res)

# allknn hook: query *is* an indexed point, so ancestor-based pruning applies.
allknn_single_search!(psat::PrunParSat, ctx::SatContext, i::Integer, res) = travelsat!(psat, ctx, i, res)

## optimize_index! integration

"""
    PrunParSatSpace(; depth=1:2:7, factor=0.3:0.1:0.99, ...) <: AbstractSolutionSpace

Autotuning search space for [`PrunParSat`](@ref)'s `depth`/`factor` hyperparameters.
"""
@kwdef struct PrunParSatSpace <: AbstractSolutionSpace
    depth = 1:2:7
    factor = 0.3:0.1:0.99
    depth_scale = (s=1.5, p1=0.5, p2=0.5, lower=1, upper=15)
    factor_scale = (s=1.07, p1=0.5, p2=0.5, lower=0.1, upper=0.9999)
end

Base.hash(c::PrunParSatConfig, h::UInt) = hash((c.depth, round(c.factor, digits=2)), h)
Base.isequal(a::PrunParSatConfig, b::PrunParSatConfig) = a.depth == b.depth && a.factor == b.factor
Base.eltype(::PrunParSatSpace) = PrunParSatConfig
Base.rand(rng::AbstractRNG, space::PrunParSatSpace) = PrunParSatConfig(rand(rng, space.depth), rand(rng, space.factor))

combine(a::PrunParSatConfig, b::PrunParSatConfig) =
    PrunParSatConfig(ceil(Int, (a.depth + b.depth) / 2), (a.factor + b.factor) / 2)

mutate(space::PrunParSatSpace, c::PrunParSatConfig, iter) =
    PrunParSatConfig(scale(c.depth; space.depth_scale...), scale(c.factor; space.factor_scale...))

optimization_space(::PrunParSat) = PrunParSatSpace()
setconfig!(config::PrunParSatConfig, index::PrunParSat, perf) = (index.config[] = config)

# Generic single-query fallback (used when `queries` isn't a SubDatabase over this index's
# own points): degrades gracefully to plain (config-ignoring) exact search.
runconfig(config::PrunParSatConfig, index::PrunParSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    search(index, ctx, q, res)

# Specialized batch-level form: recovers each query's original id via `queries.map[i]`, so
# the ancestor-based pruning this algorithm is designed for actually gets exercised during
# autotuning. Selected automatically by dispatch whenever `queries::SubDatabase`
# (`optimize_index!`'s default when `queries=nothing` builds exactly this).
function runconfig(config::PrunParSatConfig, index::PrunParSat, ctx::SatContext,
                    queries::SubDatabase, knns::AbstractVector{<:AbstractKnnQueue})
    m = length(queries)
    minbatch = getminbatch(ctx, m)
    @BATCHES minbatch scheduler=ctx.scheduler begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
    @LOOP for i in 1:m
        travelsat!(index, bctx, queries.map[i], reuse!(knns[i]), config)
    end
    end
    knns
end
