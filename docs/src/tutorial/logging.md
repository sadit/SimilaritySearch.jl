```@meta
CurrentModule = SimilaritySearch
```

# Logs, and capturing neighbors as they're built

Every context (`ctx.logger`) holds a logging backend that insertion-related functions
(`push_item!`, `append_items!`, `index!`) report progress through. This page covers that
system, then uses it to answer a more specific question: can you capture a
`SearchGraph`'s neighbor lists *as they're computed*, incrementally, instead of only
reading them from the finished graph? The answer is yes, with one important caveat.

## The default: `InformativeLog`

[`InformativeLog`](@ref) is what both [`GenericContext`](@ref) and
[`SearchGraphContext`](@ref) use by default -- it prints a status line to `stderr`, at
most once every `dt` seconds (default `1.0` for `GenericContext`, `2.0` for
`SearchGraphContext`), no matter how often it's actually called:

```julia
using SimilaritySearch

ctx = SearchGraphContext(; logger=InformativeLog(; dt=0.5, prompt="[demo]"))
```

The throttling is time-based, not count-based: under a busy parallel insertion loop
calling it thousands of times, only an occasional line actually gets printed -- this
keeps output readable instead of flooding `stderr` from every parallel batch.

## `LOG` and its event names

Internally, these functions call [`LOG`](@ref)`(ctx.logger, event, index, ctx, sp, ep)`,
where `event` is a `Symbol` naming what just happened and `sp:ep` is the range of
positions affected. The event names actually used in this package: `:push_item!`,
`:append_items!`, `:index!` (any index), and `:add!` (`SearchGraph` insertion
specifically, both the sequential and parallel construction paths). [`rebuild`](@ref)
does **not** go through `LOG` at all -- it reports progress via a
`ProgressMeter.Progress` object instead (its own `progress` keyword).

## Writing your own logger

[`AbstractLog`](@ref)'s contract is one method -- for your own log type `MyLog`, you
implement `LOG(log::MyLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext,
sp::Integer, ep::Integer)`. So a custom logger is just a struct plus that method,
dispatching on `event` for whatever you care about:

```julia
struct CountingLog <: AbstractLog
    counts::Dict{Symbol,Int}
end
CountingLog() = CountingLog(Dict{Symbol,Int}())

function SimilaritySearch.LOG(log::CountingLog, event::Symbol, index, ctx, sp::Integer, ep::Integer)
    log.counts[event] = get(log.counts, event, 0) + (ep - sp + 1)
end
```

Attach it directly (`logger=CountingLog()`), or alongside the default via
`SimilaritySearch.LogList` (not exported, so needs the qualified name), which fans one
`LOG` call out to every logger in its list:

```julia
mylog = CountingLog()
ctx = SearchGraphContext(; logger=SimilaritySearch.LogList(AbstractLog[InformativeLog(; dt=2.0), mylog]))
```

## Capturing neighbors incrementally from `:add!`

Here's the interesting part. By the time `LOG(ctx.logger, :add!, index, ctx, sp, ep)`
fires, every object in `sp:ep` already has its computed neighborhood written into
`index.adj` -- `add!(index.adj, objID, ...)` happens *before* the `LOG` call, both in the
sequential and parallel insertion loops. So a logger can read `neighbors(index.adj, i)`
right there and capture it:

```julia
struct NeighborCaptureLog <: AbstractLog
    captured::Dict{Int,Vector{UInt32}}
end
NeighborCaptureLog() = NeighborCaptureLog(Dict{Int,Vector{UInt32}}())

function SimilaritySearch.LOG(log::NeighborCaptureLog, event::Symbol, index::SearchGraph, ctx::SearchGraphContext, sp::Integer, ep::Integer)
    event === :add! || return
    for i in sp:ep
        log.captured[i] = collect(neighbors(index.adj, i))
    end
end
```

```julia
X = MatrixDatabase(rand(Float32, 4, 500))
G = SearchGraph(Dist.L2(), X)
mylog = NeighborCaptureLog()
ctx = SearchGraphContext(; logger=SimilaritySearch.LogList(AbstractLog[InformativeLog(; dt=2.0), mylog]))
index!(G, ctx)

length(mylog.captured)   # 500 -- every object got captured at insertion time
```

## The caveat: this is a forward-only snapshot, not the final neighborhood

`SearchGraph` connects **reverse** links too (when object `j` picks `i` as a neighbor,
`i` also gets a back-link to `j`) -- and that happens for *every later insertion*, long
after `i`'s own `:add!` event already fired and got captured. So what
`NeighborCaptureLog` captures for an early object is only its neighbors *as of its own
insertion moment*, not the neighbors it goes on to accumulate afterwards:

```julia
i = 10
mylog.captured[i]                       # e.g. 4 neighbors -- captured right when object 10 was inserted
collect(neighbors(G.adj, i))            # e.g. 26 neighbors -- after every later object had a chance to link back
```

The captured list is always a *prefix* of the final one (nothing captured is ever wrong,
there's just more added later) -- but if you need the true, complete neighborhood, you
still need a final pass over the finished graph, not just what you captured along the
way:

```julia
final_neighbors = Dict(i => collect(neighbors(G.adj, i)) for i in 1:length(G))
```

This makes incremental capture the right tool for things like an audit trail of how
construction proceeded, or writing forward links to disk as they're computed to bound
memory on a very large build (with a final reconciliation pass once construction
finishes) -- not a way to get the complete, final adjacency without ever reading the
finished graph.
