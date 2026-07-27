# This file is a part of SimilaritySearch.jl

"""
    execute_callbacks!(index::SearchGraph, context::SearchGraphContext, n=length(index), m=n+1; force=false)

Runs the registered callbacks (`context.hints_callback` and `context.hyperparameters_callback`)
whenever the index has grown enough to cross a `context.logbase_callback`-logarithmic size
threshold between `n` and `m`, and `n` is at least `context.starting_callback`. Internal
function, called during insertion.

# Arguments
- `index`: the search graph index.
- `context`: the context environment of the graph, see [`SearchGraphContext`](@ref).
- `n`: current (lower) size used to decide whether callbacks should fire.
- `m`: size used as the upper bound of the comparison, defaults to `n+1`.

# Keyword Arguments
- `force`: if `true`, callbacks are executed unconditionally.
"""
function execute_callbacks!(index::SearchGraph, context::SearchGraphContext, n=length(index), m=n+1; force=false)
    if force || (n >= context.starting_callback && ceil(Int, log(context.logbase_callback, n)) != ceil(Int, log(context.logbase_callback, m)))
        context.hints_callback !== nothing && execute_callback!(index, context, context.hints_callback)
        context.hyperparameters_callback !== nothing && execute_callback!(index, context, context.hyperparameters_callback)
    end
end
