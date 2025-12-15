# This file is a part of SimilaritySearch.jl

export BeamSearchSpace

"""
    BeamSearchSpace(; bsize, Δ, bsize_scale, Δ_scale)

Define search space for beam search autotuning
"""
@kwdef struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 2:2:16
    Δ = 0.9:0.025:1.1                  # this really depends on the dataset, be careful
    bsize_scale = (s=1.1, p1=0.25, p2=0.5, lower=2, upper=20)  # all these are reasonably values
    Δ_scale = (s=1.05, p1=0.75, p2=0.75, lower=0.6, upper=1.75)  # that should work in most datasets
end

Base.hash(c::BeamSearch) = hash((c.bsize, c.Δ, c.maxvisits))
Base.isequal(a::BeamSearch, b::BeamSearch) = a.bsize == b.bsize && a.Δ == b.Δ && a.maxvisits == b.maxvisits
Base.eltype(::BeamSearchSpace) = BeamSearch
Base.rand(rng::AbstractRNG, space::BeamSearchSpace) = BeamSearch(bsize=rand(rng, space.bsize), Δ=rand(rng, space.Δ))

function combine(a::BeamSearch, b::BeamSearch)
    bsize = ceil(Int, (a.bsize + b.bsize) / 2)
    Δ = round((a.Δ + b.Δ) / 2, digits=2)
    BeamSearch(; bsize, Δ)
end

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
    numqueries=32,
    space::BeamSearchSpace=BeamSearchSpace()
)
    OptimizeParameters(kind, initialpopulation, maxiters, bsize, mutbsize, crossbsize, maxpopulation, ksearch, queries, numqueries, space)
end

optimization_space(index::SearchGraph) = BeamSearchSpace()

function setconfig!(bs::BeamSearch, index::SearchGraph, perf)
    @reset bs.maxvisits = ceil(Int, 2perf.visited[end])
    @assert bs.maxvisits > 0
    index.algo[] = bs
end

function runconfig(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::AbstractKnn)
    @reset bs.maxvisits = 2index.algo[].maxvisits
    search(bs, index, ctx, q, res, index.hints)
end

"""
    execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::OptimizeParameters)

SearchGraph's callback for adjunting search parameters
"""
function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::OptimizeParameters)
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
        opt.numqueries,
        opt.initialpopulation,
        params)
end
