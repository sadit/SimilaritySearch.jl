# This file is a part of SimilaritySearch.jl

export BeamSearchSpace

"""
    BeamSearchSpace(; bsize=2:2:16, Δ=0.9:0.025:1.1, bsize_scale=(...), Δ_scale=(...))

Defines the search space explored by [`SearchModels.jl`](https://github.com/sadit/SearchModels.jl)
when autotuning `BeamSearch`'s hyperparameters, used by [`optimize_index!`](@ref) (through
[`OptimizeParameters`](@ref)).

# Keyword Arguments
- `bsize`: range of candidate values for `BeamSearch`'s `bsize` (beam size) hyperparameter.
- `Δ`: range of candidate values for `BeamSearch`'s `Δ` (soft margin) hyperparameter; this
  strongly depends on the dataset, so it may need to be adjusted.
- `bsize_scale`: named tuple of scaling parameters `(s, p1, p2, lower, upper)` passed to
  `SearchModels.scale` to mutate `bsize` values.
- `Δ_scale`: named tuple of scaling parameters `(s, p1, p2, lower, upper)` passed to
  `SearchModels.scale` to mutate `Δ` values.

# Examples

```julia
space = BeamSearchSpace(; bsize=2:2:32)
optimize_index!(index, ctx; space)
```
"""
@kwdef struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 2:2:16
    Δ = 0.9:0.025:1.1                  # this really depends on the dataset, be careful
    bsize_scale = (s=1.1, p1=0.25, p2=0.5, lower=2, upper=20)  # all these are reasonably values
    Δ_scale = (s=1.05, p1=0.75, p2=0.75, lower=0.6, upper=1.75)  # that should work in most datasets
end

Base.hash(c::BeamSearch, h::UInt) = hash((c.bsize, c.Δ), h)
Base.isequal(a::BeamSearch, b::BeamSearch) = a.bsize == b.bsize && a.Δ == b.Δ
Base.eltype(::BeamSearchSpace) = BeamSearch
Base.rand(rng::AbstractRNG, space::BeamSearchSpace) = BeamSearch(bsize=rand(rng, space.bsize), Δ=rand(rng, space.Δ))

"""
    combine(a::BeamSearch, b::BeamSearch)

`SearchModels.jl` hook: creates a new `BeamSearch` configuration by averaging `a` and `b`'s hyperparameters. Internal function.
"""
function combine(a::BeamSearch, b::BeamSearch)
    bsize = ceil(Int, (a.bsize + b.bsize) / 2)
    Δ = round((a.Δ + b.Δ) / 2, digits=2)
    BeamSearch(; bsize, Δ)
end

"""
    mutate(space::BeamSearchSpace, c::BeamSearch, iter)

`SearchModels.jl` hook: creates a new `BeamSearch` configuration by randomly perturbing `c`'s hyperparameters within `space`. Internal function.
"""
function mutate(space::BeamSearchSpace, c::BeamSearch, iter)
    bsize = SearchModels.scale(c.bsize; space.bsize_scale...)
    Δ = SearchModels.scale(c.Δ; space.Δ_scale...)
    BeamSearch(; bsize, Δ)
end

mutable struct OptimizeParameters <: Callback
    kind::ErrorFunction
    initialpopulation
    maxiters::Int
    bsize::Int
    mutbsize::Int
    crossbsize::Int
    maxpopulation::Int
    ksearch::Int32
    queries
    queries_identifiers
    numqueries::Int32
    space::BeamSearchSpace
end

"""
    OptimizeParameters(kind=MinRecall(0.9);
        initialpopulation=16,
        maxiters=12,
        bsize=4,
        mutbsize=4bsize,
        crossbsize=2bsize,
        maxpopulation=initialpopulation,
        ksearch=10,
        queries=nothing,
        numqueries=32,
        space::BeamSearchSpace=BeamSearchSpace()
    )

Creates a hyperoptimization callback using the given parameters


# Arguments

- `kind`: The kind of error function, e.g. `MinRecall(0.9)`.
- `hints`: How search hints should be computed.
- `initialpopulation`: Optimization argument that determines the initial number of configurations.
- `maxiters`: Optimization argument that determines the number of iterations.
- `bsize`: Optimization argument that determines how many top configurations are allowed to mutate and cross.
- `mutbsize`: Number of elements to be generated from mutation
- `crossbsize`: Number of elements to be generated from crossing
- `maxpopulation`: The maximum size that the population can be
- `ksearch`: The number of neighbors to be retrived by the optimization process.
- `queries`: The queryset to be used during the optimization process.
- `numqueries`: The number of queries to be performed during the optimization process.
- `space`: The cofiguration search space

# See more

- See [`BeamSearchSpace`](@ref)
- [`SearchParams` arguments of `SearchModels.jl`](https://github.com/sadit/SearchModels.jl)
for more details
"""
function OptimizeParameters(kind=MinRecall(0.9);
    initialpopulation=16,
    maxiters=12,
    bsize=4,
    mutbsize=4bsize,
    crossbsize=2bsize,
    maxpopulation=initialpopulation,
    ksearch=10,
    queries=nothing,
    queries_identifiers=nothing,
    numqueries=32,
    space::BeamSearchSpace=BeamSearchSpace()
)
    OptimizeParameters(kind, initialpopulation, maxiters, bsize, mutbsize, crossbsize, maxpopulation, ksearch, queries, queries_identifiers, numqueries, space)
end

"""
    optimization_space(index::SearchGraph)

Returns the default [`BeamSearchSpace`](@ref) used to autotune `index`'s search algorithm. Internal function.
"""
optimization_space(index::SearchGraph) = BeamSearchSpace()

"""
    setconfig!(bs::BeamSearch, index::SearchGraph, perf)

Installs `bs` as `index`'s search algorithm, after adjusting its `maxvisits` limit from the observed performance `perf`. Internal function, used by [`optimize_index!`](@ref) to apply the best found configuration.
"""
function setconfig!(bs::BeamSearch, index::SearchGraph, perf)
    @reset bs.maxvisits = ceil(Int, 2 * perf.visited[end])
    @assert bs.maxvisits > 0
    index.algo[] = bs
end

"""
    TUNINGPOOLSIZE

How many identifiers [`tuningpool`](@ref) sets aside. Each optimization draws `numqueries` of
them, so the pool is what gives successive callbacks different queries instead of a fresh
random draw every time.
"""
const TUNINGPOOLSIZE = 1024

"""
    tuningpool(ctx::SearchGraphContext, lo::Integer, hi::Integer, objectfor=nothing) -> SearchGraphContext

The context an insertion runs its callbacks with: the same one, except that an
[`OptimizeParameters`](@ref) left to sample its own queries is handed a fixed pool of
identifiers drawn once from `lo:hi`, the range this insertion is about to fill. A callback
given queries or identifiers of its own keeps them.

`objectfor` is what the two insertion paths differ by, and all they differ by. A `SearchGraph`
leaves it `nothing`: the identifiers are enough, since the objects they name are in
`database(index)` and that is what the distance takes. An `AsymmetricSearchGraph` passes
`k -> encodequery(distance(g), items[k - offset])`, because its database holds codes -- naming
an identifier there would hand the query side something it does not take.

The identifiers are the point, not an extra. The objects pooled here are being inserted, so
they are in the index, and a query that is its own vertex reads its own adjacency at distance
0 -- approximately the answer, handed over for free. Pooling them without saying where they
live tunes against a problem nobody poses; on two SISAP 2025 benchmarks that cost recall@10
against real queries 0.90 -> 0.69 (see [`tuningmask`](@ref)).

Why a pool rather than a fresh draw per callback: the callbacks fire repeatedly while the index
fills, and a new sample each time scores successive optimizations on different populations, so
their results are not comparable to each other. It also makes the symmetric and asymmetric
paths tune the same way, which matters whenever the two are compared -- otherwise the tuning
procedure varies alongside the thing under study.

Identifiers the graph has not reached yet count as external queries for that call: their objects
exist, and none of them is its own vertex yet, so [`tuningmask`](@ref) gives them nothing to
mask. The pool names the range up front and the index arrives at it gradually, so the early
callbacks tune mostly on queries from outside the index, which is a real workload, and every
callback has `numqueries` queries however large the range. Skipping them instead left the early
callbacks of a 600K build with one or two queries and no valid configuration.
"""
function tuningpool(ctx::SearchGraphContext, lo::Integer, hi::Integer, objectfor=nothing)
    cb = ctx.hyperparameters_callback
    (cb isa OptimizeParameters && cb.queries === nothing && cb.queries_identifiers === nothing &&
     hi >= lo) || return ctx
    pick = unique(rand(lo:hi, min(TUNINGPOOLSIZE, hi - lo + 1)))
    queries = objectfor === nothing ? nothing : VectorDatabase([objectfor(k) for k in pick])
    cb2 = OptimizeParameters(cb.kind; cb.initialpopulation, cb.maxiters, cb.bsize, cb.mutbsize,
                             cb.crossbsize, cb.maxpopulation, cb.ksearch, queries,
                             queries_identifiers=UInt32.(pick), cb.numqueries, cb.space)
    @set ctx.hyperparameters_callback = cb2
end

const EMPTY_MASK = UInt32[]

"""
    runconfig(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractKnnQueue)

Runs a single query `q` search using the candidate configuration `bs` (with `maxvisits` doubled with respect to `index`'s current algorithm). Internal function, used while evaluating candidate configurations during optimization.
"""

function runconfig(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractKnnQueue)
    runconfig(bs, index, ctx, q, EMPTY_MASK, res)
end

"""
    runconfig(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, mask::AbstractVector{UInt32}, res::AbstractKnnQueue)

As above, and marks every identifier in `mask` visited before descending -- what
[`tuningmask`](@ref) listed for this query, empty when it comes from outside the index. A
stored object reached at distance 0 exposes its whole adjacency, approximately its own nearest
neighbors, in a single expansion: the answer handed over for free, which a query from outside
has to earn through links. Under folding its cluster is the same shortcut, since a member's
representative also sits at distance 0. Masking those, and nothing else, leaves every gold
neighbor reachable through ordinary links, so recall stays measurable while the shortcut is
gone.
"""
function runconfig(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, mask::AbstractVector{UInt32}, res::AbstractKnnQueue)
    @reset bs.maxvisits = 2 * index.algo[].maxvisits
    vstate = getvstate(length(index), ctx)
    for id in mask
        visit!(vstate, UInt64(id))
    end
    search(bs, index, ctx, q, res, index.hints, vstate)
end

"""
    execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::OptimizeParameters)

SearchGraph's callback for adjunting search parameters
"""
function execute_callback!(index::SearchGraph, ctx::SearchGraphContext, opt::OptimizeParameters)
    if opt.ksearch == 0
        ksearch = neighborhoodsize(ctx.neighborhood, length(index))
    else
        ksearch = opt.ksearch
    end

    params = SearchParams(; opt.maxpopulation, opt.bsize, opt.mutbsize, opt.crossbsize, opt.maxiters, verbose=verbose(ctx))
    optimize_index!(index, ctx, opt.kind;
        opt.space,
        ksearch,
        opt.queries,
        opt.queries_identifiers,
        opt.numqueries,
        opt.initialpopulation,
        params)
end
