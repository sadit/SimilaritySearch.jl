# This file is a part of SimilaritySearch.jl

"""
    execute_callbacks(index, context, n=length(index), m=n+1)

Process all registered callbacks
"""
function execute_callbacks(index::SearchGraph, context::SearchGraphContext, n=length(index), m=n+1; force=false)
    if force || (n >= context.starting_callback && ceil(Int, log(context.logbase_callback, n)) != ceil(Int, log(context.logbase_callback, m)))
        context.hints_callback !== nothing && execute_callback(context.hints_callback, index, context)
        context.hyperparameters_callback !== nothing && execute_callback(context.hyperparameters_callback, index, context)
    end
end
