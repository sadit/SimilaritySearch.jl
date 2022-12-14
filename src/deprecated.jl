"""
optimize!(index::SearchGraph, opt::OptimizeParameters; kwargs...)

Specialized function to be called from `execute_callback`; this will be deprecated.
"""
function optimize!(
    index::SearchGraph,
    opt::OptimizeParameters;
    queries=nothing,
    numqueries=opt.numqueries,
    ksearch=opt.ksearch,
    initialpopulation=opt.initialpopulation,
    params=opt.params,
    verbose=index.verbose,
)
    optimize!(index, opt.kind, opt.space; queries, ksearch, numqueries, initialpopulation, verbose, params)
end

