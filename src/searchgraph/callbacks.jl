# This file is a part of SimilaritySearch.jl

"""
    execute_callbacks!(index::SearchGraph, context::SearchGraphContext, n=length(index), m=n+1; force=false)

Runs the registered callbacks (`context.hints_callback` and `context.hyperparameters_callback`)
whenever the index, growing from `n` to `m`, reaches `context.starting_callback` or crosses a
`context.logbase_callback`-logarithmic size threshold above it. Internal function, called during
insertion.

Reaching `starting_callback` counts on its own: a parallel insertion block can start below it and
end above it without crossing a logarithmic threshold. Until 1.6.9 the test read `n >=
starting_callback`, so a database inserted in one block that started at 1 (2000 objects under the
default `parallel_block = 32 * nthreads()` with 64 threads) never ran a callback, and its graph was
left with no hints and untuned.

# Arguments
- `index`: the search graph index.
- `context`: the context environment of the graph, see [`SearchGraphContext`](@ref).
- `n`: current (lower) size used to decide whether callbacks should fire.
- `m`: size used as the upper bound of the comparison, defaults to `n+1`.

# Keyword Arguments
- `force`: if `true`, callbacks are executed unconditionally.
"""
function execute_callbacks!(index::SearchGraph, context::SearchGraphContext, n=length(index), m=n+1; force=false)
    s = context.starting_callback
    if force || (m >= s && (n < s || ceil(Int, log(context.logbase_callback, n)) != ceil(Int, log(context.logbase_callback, m))))
        context.hints_callback !== nothing && execute_callback!(index, context, context.hints_callback)
        context.hyperparameters_callback !== nothing && execute_callback!(index, context, context.hyperparameters_callback)
    end
end
