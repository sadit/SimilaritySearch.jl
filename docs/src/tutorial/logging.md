```@meta
CurrentModule = SimilaritySearch
```

# Reporting, observing, and capturing neighbors as they're built

A context carries **two** logging slots, because indexing produces two different kinds of
event and mixing them makes both harder to use:

- `ctx.reporters` -- where messages go to be *read*: `stderr`, a file, a monitoring
  service. They receive [`INFORM`](@ref).
- `ctx.observers` -- what *reacts* to a structural change so something durable happens:
  persist the range that was just inserted, checkpoint, keep statistics. They receive
  [`OBSERVE`](@ref).

Keeping them apart is what lets you silence the console without disabling persistence, and
what lets a context the library builds for itself inherit your reporters without inheriting
your observers.

This page covers both, then uses the observation channel to answer a more specific
question: can you capture a `SearchGraph`'s neighbor lists *as they're computed*,
incrementally, instead of only reading them from the finished graph? The answer is yes,
with one important caveat.

## Silencing

The switch is an empty reporter list:

```julia
using SimilaritySearch

ctx = SearchGraphContext(; reporters=[])
```

That is total: no progress line, no optimization detail, nothing. It is also free -- with
no destination, a message is not even built, so a silenced context costs one emptiness
check per call site.

`verbose` is a different knob. It is a *level*, not a switch: it decides whether the
chatty, per-iteration messages (optimization progress, hint selection) are produced at all,
so that they do not crowd out the per-block progress. It defaults to `false`.

```julia
ctx = SearchGraphContext(; verbose=true)     # everything the library can say
ctx = SearchGraphContext(; reporters=[])     # nothing, regardless of verbose
```

A few functions -- [`fft`](@ref), [`dnet`](@ref) -- take no context at all. They accept a
`reporters` keyword so a caller that *has* a context can pass `ctx.reporters` and have its
silencing reach them.

Warnings are deliberately not part of this channel. `@warn` (a recall that collapsed
during optimization, an unrecognized environment variable, a non-metric distance) goes
through Julia's own logging, so silencing progress reporting never hides a diagnosis.

## Reporters: `InformativeLog`

[`InformativeLog`](@ref) is the default in every context. It renders a message as a status
line carrying the index length, live heap, max-RSS and a timestamp, throttled so it prints
at most once every `dt` seconds no matter how often it is called:

```julia
ctx = SearchGraphContext(; reporters=InformativeLog(; dt=0.5, prompt="[demo]"))
```

Its first argument is where to write. The default, `nothing`, means *whatever `stderr` is
bound to at print time*, so `redirect_stderr` follows it -- which is what capturing output
in a test or a rendered page needs. Pass an `IO` to fix a destination, and pass several
reporters to write to more than one:

```julia
io = open("build.log", "a")
ctx = SearchGraphContext(; reporters=[InformativeLog(), InformativeLog(io)])
```

Throttling is time-based, not count-based. It is also the *only* reason a message is ever
dropped: `dt <= 0` disables it, and then nothing is lost.

```julia
ctx = GenericContext(; reporters=InformativeLog(; dt=0))   # every message, in order
```

That has a cost worth knowing: with no throttle the reporter becomes a serialization
point. Invisible at a per-block call site, very much not invisible at a per-item one such
as an inverted file's `push_item!`.

## Observers and the `:add!` contract

Insertion-related functions (`push_item!`, `append_items!`, `index!`) call
[`OBSERVE`](@ref)`(ctx, :add!, index, sp, ep)`, where `sp:ep` is the exact, contiguous
range of ids affected. Every index type (`SearchGraph`,
`ExhaustiveSearch`/`ParallelExhaustiveSearch`, `InvertedFile`/`DictInvertedFile`,
`BM25InvertedFile`, `Sat`) emits that same event regardless of which function was the entry
point -- one canonical name for one canonical kind of mutation.

`OBSERVE` fires **exactly once** per logical batch: when one mutating function delegates
its work to another (e.g. `append_items!` growing `db` and then calling `index!` to do the
actual indexing), only the function that performs the mutation reports -- the delegating
wrapper stays silent, so the same batch is never reported twice. That precision is what
makes the `:add!` stream usable as a write-ahead log for incremental or crash-recoverable
indexing; see [`OBSERVE`](@ref)'s docstring for the full contract.

Anything *not* structural is not an event at all. `index!` on a brute-force index, where
`db` already is the index and there is nothing to build, sends a message instead, so the
observation channel carries no pings a consumer would have to learn to ignore.
[`rebuild`](@ref) does not go through either channel -- it reports through a
`ProgressMeter.Progress` object (its own `progress` keyword).

The ready-made observer is [`CallbackLog`](@ref):

```julia
ranges = Tuple{Int,Int}[]
ctx = SearchGraphContext(; observers=CallbackLog((index, sp, ep) -> push!(ranges, (Int(sp), Int(ep)))))
```

Writing your own is a struct plus one method:

```julia
struct CountingLog <: SimilaritySearch.AbstractObserver
    counts::Dict{Symbol,Int}
end
CountingLog() = CountingLog(Dict{Symbol,Int}())

function SimilaritySearch.OBSERVE(log::CountingLog, event::Symbol, index, ctx, sp::Integer, ep::Integer)
    log.counts[event] = get(log.counts, event, 0) + (ep - sp + 1)
end

ctx = SearchGraphContext(; observers=CountingLog())
```

Two rules go with observers. An observer belongs to **one** index -- its state is that
index's id ranges, and sharing it across indexes merges two histories into one stream. And
its callback runs inline, with exceptions propagating: if a durable write fails, the
insertion that triggered it fails too, which is the correct behaviour for a write-ahead
log. (A reporter, by contrast, is *meant* to be shared: that is what makes one throttle
govern one console.)

## What a context built by the library inherits

Some functions build a context of their own -- `neardup`'s convenience wrapper picks
between an exact and an approximate index depending on `recall`, and the `EpsilonHints`
callback builds one for a scratch index. Those contexts inherit the caller's **reporters**,
so your silencing reaches them, and deliberately do **not** inherit its observers: a
scratch index emits `:add!` for ids of an entirely different index, and letting those reach
your `CallbackLog` would corrupt the reconstruction it is maintaining.

Functions with nothing to inherit from take the configuration directly:

```julia
D = neardup(dist, X, ϵ; reporters=[], observers=CallbackLog(persist!))
```

## Capturing neighbors incrementally from `:add!`

Here is the interesting part. By the time `OBSERVE(ctx, :add!, index, sp, ep)` fires, every
object in `sp:ep` already has its computed neighborhood written into `index.adj` --
`add!(index.adj, objID, ...)` happens *before* the call, in both the sequential and the
parallel insertion loops. So an observer can read `neighbors(index.adj, i)` right there:

```julia
struct NeighborCaptureLog <: SimilaritySearch.AbstractObserver
    captured::Dict{Int,Vector{UInt32}}
end
NeighborCaptureLog() = NeighborCaptureLog(Dict{Int,Vector{UInt32}}())

function SimilaritySearch.OBSERVE(log::NeighborCaptureLog, event::Symbol, index::SearchGraph, ctx::SearchGraphContext, sp::Integer, ep::Integer)
    for i in sp:ep
        log.captured[i] = collect(neighbors(index.adj, i))
    end
end
```

```julia
X = MatrixDatabase(rand(Float32, 4, 500))
G = SearchGraph(Dist.L2(), X)
mylog = NeighborCaptureLog()
ctx = SearchGraphContext(; observers=mylog)
index!(G, ctx)

length(mylog.captured)   # 500 -- every object got captured at insertion time
```

Note what did *not* have to change: the default `InformativeLog` is still printing
progress, because it lives in the other slot. Attaching an observer neither replaces nor
disturbs it, and `reporters=[]` would silence it without touching the capture.

## The caveat: this is a forward-only snapshot, not the final neighborhood

`SearchGraph` connects **reverse** links too (when object `j` picks `i` as a neighbor, `i`
also gets a back-link to `j`) -- and that happens for *every later insertion*, long after
`i`'s own `:add!` event already fired and got captured. So what `NeighborCaptureLog`
captures for an early object is only its neighbors *as of its own insertion moment*, not
the neighbors it goes on to accumulate afterwards:

```julia
i = 10
mylog.captured[i]                       # e.g. 4 neighbors -- captured right when object 10 was inserted
collect(neighbors(G.adj, i))            # e.g. 26 neighbors -- after every later object had a chance to link back
```

The captured list is always a *prefix* of the final one (nothing captured is ever wrong,
there's just more added later) -- but if you need the true, complete neighborhood, you
still need a final pass over the finished graph, not just what you captured along the way:

```julia
final_neighbors = Dict(i => collect(neighbors(G.adj, i)) for i in 1:length(G))
```

This makes incremental capture the right tool for things like an audit trail of how
construction proceeded, or writing forward links to disk as they're computed to bound
memory on a very large build (with a final reconciliation pass once construction finishes)
-- not a way to get the complete, final adjacency without ever reading the finished graph.

Next: [Inverted files and posting list intersections](invertedfiles.md).
