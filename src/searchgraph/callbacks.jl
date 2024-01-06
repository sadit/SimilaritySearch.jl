# This file is a part of SimilaritySearch.jl

"""
    execute_callbacks(callbacks, index, n=length(index), m=n+1)

Process all registered callbacks
"""
function execute_callbacks(setup::SearchGraphSetup, index::SearchGraph, n=length(index), m=n+1; force=false)
    if force || (n >= setup.starting_callback && ceil(Int, log(setup.logbase_callback, n)) != ceil(Int, log(setup.logbase_callback, m)))
        setup.hints_callback !== nothing && execute_callback(setup.hints_callback, index)
        setup.hyperparameters_callback !== nothing && execute_callback(setup.hyperparameters_callback, index)
    end
end
