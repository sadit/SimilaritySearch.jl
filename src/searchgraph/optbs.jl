# This file is a part of SimilaritySearch.jl

export BeamSearchSpace

"""
    BeamSearchSpace(; bsize, Δ, bsize_scale, Δ_scale)

Define search space for beam search autotuning
"""
@with_kw struct BeamSearchSpace <: AbstractSolutionSpace
    bsize = 2:8:64
    Δ = [0.8, 0.9, 1.0, 1.1]                  # this really depends on the dataset, be careful
    bsize_scale = (s=1.5, p1=0.75, p2=0.75, lower=2, upper=512)  # all these are reasonably values
    Δ_scale = (s=1.05, p1=0.75, p2=0.75, lower=0.5, upper=1.99)  # that should work in most datasets
end

Base.hash(c::BeamSearch) = hash((c.bsize, c.Δ, c.maxvisits))
Base.isequal(a::BeamSearch, b::BeamSearch) = a.bsize == b.bsize && a.Δ == b.Δ && a.maxvisits == b.maxvisits
Base.eltype(::BeamSearchSpace) = BeamSearch
Base.rand(space::BeamSearchSpace) = BeamSearch(bsize=rand(space.bsize), Δ=rand(space.Δ))

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
    params::SearchParams
    ksearch::Int32
    numqueries::Int32
    space::BeamSearchSpace
    verbose::Bool
end

"""
    OptimizeParameters(kind=MinRecall(0.9);
        initialpopulation=16,
        maxiters=12,
        bsize=4,
        ksearch=10,
        numqueries=32,
        verbose=false,
        params=SearchParams(; maxpopulation=initialpopulation, bsize, mutbsize=4bsize, crossbsize=2bsize, maxiters, verbose),
        space::BeamSearchSpace=BeamSearchSpace()
    )

Creates a hyperoptimization callback using the given parameters


# Arguments

- `kind`: The kind of error function, e.g. `MinRecall(0.9)`.
- `hints`: How search hints should be computed.
- `initialpopulation`: Optimization argument that determines the initial number of configurations.
- `maxiters`: Optimization argument that determines the number of iterations.
- `bsize`: Optimization argument that determines how many top configurations are allowed to mutate and cross.
- `params`: The `SearchParams` arguments (if separated optimization arguments are not enough)
- `ksearch`: The number of neighbors to be retrived by the optimization process.
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
        ksearch=10,
        numqueries=32,
        verbose=false,
        params=SearchParams(; maxpopulation=initialpopulation, bsize, mutbsize=4bsize, crossbsize=2bsize, maxiters, verbose),
        space::BeamSearchSpace=BeamSearchSpace()
    )
    OptimizeParameters(kind, initialpopulation, params, ksearch, numqueries, space, verbose)
end

optimization_space(index::SearchGraph) = BeamSearchSpace()

function setconfig!(bs::BeamSearch, index::SearchGraph, perf)
    index.algo.bsize = bs.bsize
    index.algo.Δ = bs.Δ
    index.algo.maxvisits = ceil(Int, 2perf.visited[end])
end

function runconfig(bs::BeamSearch, index::SearchGraph, ctx::SearchGraphContext, q, res::KnnResult)
    search(bs, index, ctx, q, res, index.hints; maxvisits = 4index.algo.maxvisits)
    # search(bs, index, q, res, index.hints, caches)
end

"""
    execute_callback(index::SearchGraph, context::SearchGraphContext, opt::OptimizeParameters)

SearchGraph's callback for adjunting search parameters
"""
function execute_callback(index::SearchGraph, context::SearchGraphContext, opt::OptimizeParameters)
    queries = nothing
    optimize_index!(index, context, opt.kind; opt.space,
        queries, opt.ksearch, opt.numqueries, opt.initialpopulation, opt.verbose, opt.params)
end
