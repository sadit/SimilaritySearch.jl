# This file is a part of SimilaritySearch.jl

"""
struct SearchGraphCallbacks
    hints::Union{Nothing,Callback} = DisjointHints()
    hyperparameters::Union{Nothing,Callback} = OptimizeParameters(kind=ParetoRecall())
    logbase::Float32 = 1.5
    starting::Int32 = 8
end

Call insertions and indexing methods with `SearchGraphCallbacks` objects to control how the index structure is adjusted (callbacks are called when ``n > starting`` and ``\\lceil(\\log(logbase, n)\\rceil != \\lceil\\log(logbase, n+1)\\rceil``)
"""
@with_kw struct SearchGraphCallbacks
    hints::Union{Nothing,Callback} = DisjointHints()
    hyperparameters::Union{Nothing,Callback} = OptimizeParameters(kind=ParetoRecall())
    logbase::Float32 = 1.5
    starting::Int32 = 8
end

Base.copy(g::SearchGraph;
    dist=g.dist,
    db=g.db,
    links=g.links,
    locks=g.locks,
    hints=g.hints,
    search_algo=copy(g.search_algo),
    verbose=true
) = SearchGraph(; dist, db, links, locks, hints, search_algo, verbose)

"""
    SearchGraphCallbacks(kind::ErrorFunction;
        hints=DisjointHints(),
        logbase=1.5,
        starting=8,
        initialpopulation=16,
        maxpopulation=initialpopulation,
        maxiters=12,
        bsize=4,
        mutbsize=4bsize,
        crossbsize=2bsize,
        tol=-1.0,
        params=SearchParams(; maxpopulation, bsize, mutbsize, crossbsize, tol, maxiters),
        ksearch=10,
        numqueries=32,
        space::BeamSearchSpace=BeamSearchSpace()
    )

Convenient constructor function to create `SearchGraphCallbacks` structs. See [`SearchGraphCallbacks`](@ref), [`SearchParams`](@ref), and [`BeamSearchSpace`](@ref) for more details.

# Arguments

- `kind`: The kind of error function, e.g. `MinRecall(0.9)`.
- `hints`: How search hints should be computed.
- `logbase`: Controls the periodicity of callback executions.
- `starting`: Controls the minimum size of the index before execute callbacks.
- `initialpopulation`: Optimization argument that determines the initial number of configurations.
- `maxiters`: Optimization argument that determines the number of iterations.
- `bsize`: Optimization argument that determines how many top configurations are allowed to mutate and cross.
- `tol`: Optimization argument that determines the minimal tolerance improvement to stop the optimization.
- `params`: The `SearchParams` arguments (if separated optimization arguments are not enough)
- `ksearch`: The number of neighbors to be retrived by the optimization process.
- `numqueries`: The number of queries to be performed during the optimization process.
- `space`: The cofiguration search space
"""
function SearchGraphCallbacks(kind::ErrorFunction;
    hints=DisjointHints(),
    logbase=1.5,
    starting=8,
    initialpopulation=16,
    maxiters=12,
    bsize=4,
    tol=-1.0,
    params=SearchParams(; maxpopulation=initialpopulation, bsize, mutbsize=4bsize, crossbsize=2bsize, tol, maxiters),
    ksearch=10,
    numqueries=32,
    space::BeamSearchSpace=BeamSearchSpace()
)
hyperparameters = OptimizeParameters(; kind, initialpopulation, params, ksearch, numqueries, space)
SearchGraphCallbacks(; hints, hyperparameters, logbase, starting)
end

"""
    execute_callbacks(index::SearchGraph, n=length(index), m=n+1)

Process all registered callbacks
"""
function execute_callbacks(callbacks::SearchGraphCallbacks, index::SearchGraph, n=length(index), m=n+1; force=false)
    if force || (n >= callbacks.starting && ceil(Int, log(callbacks.logbase, n)) != ceil(Int, log(callbacks.logbase, m)))
        callbacks.hints !== nothing && execute_callback(callbacks.hints, index)
        callbacks.hyperparameters !== nothing && execute_callback(callbacks.hyperparameters, index)
    end
end