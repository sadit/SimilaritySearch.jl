# This file is part of SimilaritySearch.jl

struct BeamSearchParSatConfig
    α::Float32
    bs::BeamSearch
end

struct BeamSearchParSat{ST<:Sat} <: AbstractSearchIndex
    sat::ST
    parents::Vector{UInt32}
    config::Ref{BeamSearchParSatConfig}
end

"""
    BeamSearchParSat(sat::Sat; α=1f0, bsize=4, Δ=1.0, maxvisits=typemax(Int))

`allknn`-oriented beam search: each query that is one of `sat`'s own indexed points climbs
its ancestor chain, running a beam search rooted at each ancestor in turn until the result
stops improving. Queries that are *not* one of `sat`'s own points fall back to a plain
root-rooted beam search.
"""
function BeamSearchParSat(sat::Sat; α::Real=1f0, bsize::Integer=4, Δ::Real=1.0, maxvisits::Integer=typemax(Int))
    c = BeamSearchParSatConfig(convert(Float32, α), BeamSearch(; bsize, Δ, maxvisits))
    BeamSearchParSat(sat, compute_parents(sat), Ref(c))
end

@inline database(p::BeamSearchParSat) = database(p.sat)
@inline database(p::BeamSearchParSat, i) = database(p.sat, i)
@inline distance(p::BeamSearchParSat) = distance(p.sat)
@inline Base.length(p::BeamSearchParSat) = length(p.sat)

function Base.show(io::IO, p::BeamSearchParSat)
    println(io, typeof(p), " config=", p.config[])
    show(io, p.sat)
end

function travelsat!(psat::BeamSearchParSat, ctx::SatContext, i::Integer, res::AbstractKnnQueue,
                     c::BeamSearchParSatConfig=psat.config[])
    q = database(psat, i)
    p = psat.parents[i]
    beamsearchsat!(psat.sat, ctx, p, c.bs, q, res; tabu=zero(UInt32))

    prev = typemax(Float32)
    while length(res) < maxlength(res) || maximum(res) < prev
        prev = maximum(res)
        newp = psat.parents[p]
        newp == p && break     # reached the root and converged: stop instead of re-running
        beamsearchsat!(psat.sat, ctx, newp, c.bs, q, res; tabu=p)
        p = newp
    end

    res
end

# Generic (out-of-sample) query: root beam search, ignoring the ancestor hint.
search(bs::BeamSearchParSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    beamsearchsat!(bs.sat, ctx, bs.sat.root, bs.config[].bs, q, res)

allknn_single_search!(bs::BeamSearchParSat, ctx::SatContext, i::Integer, res) = travelsat!(bs, ctx, i, res)

## optimize_index! integration

"""
    BeamSearchParSatSpace(; α=[1f0], bs=BeamSearchSpace(...)) <: AbstractSolutionSpace

Autotuning search space for [`BeamSearchParSat`](@ref). `α` is currently held fixed (see
`combine`/`mutate` below) -- this preserves the upstream algorithm's own choice; only `bs`
(the wrapped `BeamSearch` hyperparameters) is actually tuned.
"""
@kwdef struct BeamSearchParSatSpace <: AbstractSolutionSpace
    α = [1f0]
    α_scale = (s=1.1, p1=0.5, p2=0.5, lower=0.5, upper=2.0)
    bs::BeamSearchSpace = BeamSearchSpace(;
        bsize = 8:16:64, Δ = [0.8, 1.0, 1.3, 1.5],
        bsize_scale = (s=1.5, p1=0.5, p2=0.5, lower=4, upper=256),
        Δ_scale = (s=1.2, p1=0.5, p2=0.5, lower=0.5, upper=2.0))
end

Base.hash(c::BeamSearchParSatConfig, h::UInt) = hash((round(c.α; digits=2), c.bs), h)
Base.isequal(a::BeamSearchParSatConfig, b::BeamSearchParSatConfig) =
    round(a.α; digits=2) == round(b.α; digits=2) && isequal(a.bs, b.bs)
Base.eltype(::BeamSearchParSatSpace) = BeamSearchParSatConfig
Base.rand(rng::AbstractRNG, space::BeamSearchParSatSpace) =
    BeamSearchParSatConfig(rand(rng, space.α), rand(rng, space.bs))

# α is held fixed at 1f0, matching the upstream algorithm's own choice (real tuning left
# commented out in the original implementation).
combine(a::BeamSearchParSatConfig, b::BeamSearchParSatConfig) =
    BeamSearchParSatConfig(1f0, combine(a.bs, b.bs))

mutate(space::BeamSearchParSatSpace, c::BeamSearchParSatConfig, iter) =
    BeamSearchParSatConfig(1f0, mutate(space.bs, c.bs, iter))

optimization_space(::BeamSearchParSat) = BeamSearchParSatSpace()
setconfig!(config::BeamSearchParSatConfig, index::BeamSearchParSat, perf) = (index.config[] = config)

runconfig(config::BeamSearchParSatConfig, index::BeamSearchParSat, ctx::SatContext, q, res::AbstractKnnQueue) =
    search(index, ctx, q, res)

function runconfig(config::BeamSearchParSatConfig, index::BeamSearchParSat, ctx::SatContext,
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
