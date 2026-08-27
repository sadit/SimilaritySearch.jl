# This file is part of SimilaritySearch.jl

import Dates
export AbstractLog, AbstractReporter, AbstractObserver, InformativeLog, CallbackLog,
       OBSERVE, INFORM, @inform, LOG

"""
    abstract type AbstractLog end

Root of the logging taxonomy. A log is stored in a context (a subtype of [`AbstractContext`](@ref))
and receives events emitted by index operations, so that those operations can report what they do
without depending on any particular backend.

There are two kinds of log, and a context holds them in **two separate slots** because they answer
two different questions:

- [`AbstractReporter`](@ref), in `ctx.reporters` — *renders* an event somewhere a human or a service
  will read it: `stderr`, a file, a monitoring endpoint. Receives [`INFORM`](@ref).
- [`AbstractObserver`](@ref), in `ctx.observers` — *reacts* to an event so that something durable
  happens: persist the range that was just inserted, checkpoint, update statistics. Receives
  [`OBSERVE`](@ref).

Keeping them apart is what makes it possible to silence the console without disabling persistence
(`reporters=[]` leaves `observers` intact), and what lets an internally built context inherit the
caller's reporters without inheriting its observers -- see [`OBSERVE`](@ref) for why that second
one matters.

# Rules

- **A reporter is meant to be shared**; that is what makes one throttle govern one console instead
  of several indexes fighting over the screen with a `dt` each.
- **An observer belongs to exactly one index.** Its state is *that* index's id ranges, and sharing
  it across indexes mixes two histories into one stream.
- **Never call `OBSERVE` or `INFORM` inside a [`@BATCHES`](@ref) block.** Backends carry no lock:
  every mutating entry point (`push_item!`, `append_items!`, `index!`) is serial by design and
  every call site today sits outside any parallel region. Breaking this costs a duplicated or
  garbled line rather than corruption, but it also breaks the exactly-once guarantee `OBSERVE`
  consumers rely on.
"""
abstract type AbstractLog end

"""
    abstract type AbstractReporter <: AbstractLog end

A log that renders events for reading: to `stderr`, to a file, to a service. Reporters live in
`ctx.reporters` and receive [`INFORM`](@ref). A concrete subtype must implement

    INFORM(r::MyReporter, ctx, msg::Function, index, data)

where `msg` is a zero-argument function returning the message text, `index` is the index the message
is about or `nothing`, and `data` is an arbitrary structured payload or `nothing`. `ctx` is the
context the message came from, or `nothing` when it came from a function that has no context (see
[`INFORM`](@ref)'s vector form).

**Call `msg()` only after deciding the message will be emitted.** That is the whole point of it
being a function: a throttled or filtered message must not pay for building its own text.

See [`InformativeLog`](@ref) for the reference implementation.
"""
abstract type AbstractReporter <: AbstractLog end

"""
    abstract type AbstractObserver <: AbstractLog end

A log that reacts to structural events so that something durable happens. Observers live in
`ctx.observers` and receive [`OBSERVE`](@ref). A concrete subtype must implement

    OBSERVE(o::MyObserver, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)

See [`CallbackLog`](@ref) for the reference implementation, and [`OBSERVE`](@ref) for the contract
the event stream obeys.
"""
abstract type AbstractObserver <: AbstractLog end

reporterlist(::Nothing) = AbstractReporter[]
reporterlist(x::AbstractReporter) = AbstractReporter[x]
reporterlist(x::AbstractVector) = convert(Vector{AbstractReporter}, x)
reporterlist(x::Tuple) = AbstractReporter[x...]

observerlist(::Nothing) = AbstractObserver[]
observerlist(x::AbstractObserver) = AbstractObserver[x]
observerlist(x::AbstractVector) = convert(Vector{AbstractObserver}, x)
observerlist(x::Tuple) = AbstractObserver[x...]

"""
    OBSERVE(ctx::AbstractContext, event::Symbol, index::AbstractSearchIndex, sp::Integer, ep::Integer)

Reports a **structural** event to every observer in `ctx.observers`, in order. Called by index
operations that mutate the index (`push_item!`, `append_items!`, `index!`).

# The event contract

`event` names *what happened*, not which Julia function was called -- it must not simply mirror the
name of the calling method (`:push_item!`, `:append_items!`, `:index!`, ...). It is one of a small,
curated set of event kinds. Today there is exactly one:

- `:add!` -- one or more objects were added to the index, and `sp:ep` is the exact, contiguous range
  of ids affected by this call. Every index type (`SearchGraph`,
  `ExhaustiveSearch`/`ParallelExhaustiveSearch`, `InvertedFile`/`DictInvertedFile`,
  `BM25InvertedFile`, `Sat`) emits this same event, regardless of whether the call arrived via a
  single-item `push_item!` or a batch `append_items!`/`index!` -- one canonical name for one
  canonical kind of mutation, not one name per entry point.

Anything that is *not* structural -- an `index!` on a brute-force index where `db` already **is** the
index, so there is nothing to build -- is not an event at all: it is an [`@inform`](@ref) message.
The observation channel carries no informative pings, so a consumer never has to learn which event
kinds to ignore.

**Exactly-once**: when one mutating function calls another mutating function internally (e.g. an
`append_items!` that delegates its actual work to `index!`), only the function that performs/owns
the mutation may call `OBSERVE` for that range -- a caller that purely delegates must stay silent,
never reporting the same range again under a different (or the same) event name. See
`SimilaritySearch.InvertedFiles`'s `append_items!`/`index!` (or `SearchGraph`'s, in
`searchgraph/insertions.jl`) for the reference pattern: the outer function delegates without
observing, and the inner function it calls is the sole emitter.

**Why this precision matters**: every current index type is append-only (ids are assigned
monotonically and are never removed or modified), so a correctly-behaving stream of `:add!` events --
exactly one per logical batch, with an accurate, gap-free `sp:ep` -- is on its own sufficient for a
consumer to reconstruct or checkpoint which ids are durably indexed at any point in time. This is
what makes the mechanism usable as a write-ahead log for incremental or crash-recoverable indexing:
a consumer can replay the `:add!` stream instead of re-deriving state from the index itself.
Duplicate events, an event misnamed as if it were a different action, or an inaccurate `sp:ep`
silently break that invariant.

**Observers do not travel into internally built contexts.** Several functions build a context of
their own for a scratch index (see `hints.jl`'s `EpsilonHints` callback). That scratch index emits
`:add!` for ids of a *different* index; letting those reach the caller's observers would corrupt
the very reconstruction described above. Reporters do travel, which is what makes silencing reach
the whole call tree. Inherit observers only when the new context drives the same index the caller's
observers are already watching.

# Examples

```julia
using SimilaritySearch

struct Recorder <: AbstractObserver
    events::Vector{Tuple{Symbol,Int,Int}}
end

SimilaritySearch.OBSERVE(o::Recorder, event, index, ctx, sp, ep) = push!(o.events, (event, Int(sp), Int(ep)))

rec = Recorder([])
ctx = GenericContext(; reporters=[], observers=rec)   # silent, but still observed
idx = ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(rand(Float32, 4, 0)))
append_items!(idx, ctx, MatrixDatabase(rand(Float32, 4, 10)))
rec.events   # [(:add!, 1, 10)]
```
"""
function OBSERVE(ctx::AbstractContext, event::Symbol, index::AbstractSearchIndex, sp::Integer, ep::Integer)
    for o in ctx.observers
        OBSERVE(o, event, index, ctx, sp, ep)
    end

    nothing
end

"""
    INFORM(ctx::AbstractContext, msg; index=nothing, data=nothing)

Sends a free-form message to every reporter in `ctx.reporters`, in order. Unlike [`OBSERVE`](@ref)
this carries no contract at all: the message does not have to say what it affects, and neither
`index` nor `data` is required.

`msg` may be a `String` or a zero-argument function returning one; a `String` is wrapped. Prefer
[`@inform`](@ref) at call sites: it also skips building the message when there are no reporters at
all.

# Keyword Arguments
- `index`: the index the message is about, when there is one. A reporter may use it (e.g. to append
  the current length); it is never required.
- `data`: an arbitrary structured payload for a reporter that does not want text -- a service
  emitting JSON takes this, `stderr` takes the string.
"""
function INFORM(ctx::AbstractContext, msg::Function; index=nothing, data=nothing)
    _fanout(ctx.reporters, ctx, msg, index, data)
end

INFORM(ctx::AbstractContext, msg::AbstractString; index=nothing, data=nothing) =
    INFORM(ctx, () -> msg; index, data)

INFORM(reporters::AbstractVector, msg::Function; index=nothing, data=nothing) =
    _fanout(reporters, nothing, msg, index, data)

INFORM(reporters::AbstractVector, msg::AbstractString; index=nothing, data=nothing) =
    INFORM(reporters, () -> msg; index, data)

INFORM(reporter::AbstractReporter, msg::Function; index=nothing, data=nothing) =
    INFORM(reporter, nothing, msg, index, data)

INFORM(reporter::AbstractReporter, msg::AbstractString; index=nothing, data=nothing) =
    INFORM(reporter, () -> msg; index, data)

function _fanout(reporters, ctx, msg, index, data)
    for r in reporters
        INFORM(r, ctx, msg, index, data)
    end

    nothing
end

"""
    SimilaritySearch._reporters(target) -> AbstractVector

The reporter list of `target`, which is a context, a bare vector of reporters, or a single
reporter. Used by [`@inform`](@ref) so that a function without a context -- `fft`, `dnet` -- can
still report through whatever its caller handed it.
"""
_reporters(ctx::AbstractContext) = ctx.reporters
_reporters(reporters::AbstractVector) = reporters
_reporters(reporter::AbstractReporter) = (reporter,)

"""
    @inform ctx "message \$(interpolated)"
    @inform ctx "message" index=idx data=(; k, n)
    @inform reporters "message"

Call-site form of [`INFORM`](@ref). Expands to an emptiness check on the reporter list followed by
an `INFORM` whose message is a closure, so that a silenced context pays neither for the message nor
for the closure that would have built it. Use it in preference to calling `INFORM` directly,
especially at sites that run once per indexed item.

The first argument is a context, or a bare vector of reporters for a function that has none.
"""
macro inform(ctx, msg, kwargs...)
    kws = map(kwargs) do kw
        (kw isa Expr && kw.head === :(=)) ||
            throw(ArgumentError("@inform: trailing arguments must be keyword arguments, got `$kw`"))
        Expr(:kw, kw.args[1], esc(kw.args[2]))
    end

    ctxe = esc(ctx)
    empty = Expr(:call, :isempty, Expr(:call, GlobalRef(@__MODULE__, :_reporters), ctxe))
    call = Expr(:call, GlobalRef(@__MODULE__, :INFORM), ctxe, Expr(:->, Expr(:tuple), esc(msg)), kws...)
    Expr(:||, empty, call)
end

"""
    InformativeLog(io=nothing; dt::Real=1.0, prompt::AbstractString="LOG")

The reference [`AbstractReporter`](@ref): renders a message as a status line, throttled so that it
prints at most once every `dt` seconds. The line carries the message, the index length when the
message names an index, live heap and max-RSS, and a timestamp.

# Arguments
- `io`: where to write. `nothing` (the default) means *whatever `stderr` is bound to at print time*,
  so `redirect_stderr` follows it; pass an `IO` (an open file handle, `stdout`, ...) to fix a
  destination. The stream is flushed after every line.

# Keyword Arguments
- `dt`: minimum number of seconds between two printed lines. **`dt <= 0` disables throttling
  entirely, and then no message is ever dropped** -- at the cost of making the reporter a
  serialization point, which is invisible at a per-block call site and very much not invisible at a
  per-item one. With `dt > 0`, throttling is the only reason a message is ever lost.
- `prompt`: a prefix printed at the beginning of every line, useful to tell apart the reporters of
  different indexes or stages.

# Examples

```julia
using SimilaritySearch

ctx = GenericContext()                                          # prints to stderr, dt=1
ctx = GenericContext(; reporters=InformativeLog(; dt=10))       # once every 10 seconds
ctx = GenericContext(; reporters=InformativeLog(open("build.log", "a")))  # to a file
ctx = GenericContext(; reporters=[InformativeLog(), InformativeLog(io)])  # to both
ctx = GenericContext(; reporters=[])                            # silent
```
"""
mutable struct InformativeLog <: AbstractReporter
    io::Union{Nothing,IO}
    dt::Float64
    prompt::String
    last::Union{Nothing,UInt64}
end

InformativeLog(io::Union{Nothing,IO}=nothing; dt::Real=1.0, prompt::AbstractString="LOG") =
    InformativeLog(io, convert(Float64, dt), String(prompt), nothing)

"""
    INFORM(log::InformativeLog, ctx, msg::Function, index, data)

Prints one status line, unless `log.dt > 0` and fewer than `log.dt` seconds have elapsed since the
previous printed line, in which case nothing happens and `msg` is never called. See
[`InformativeLog`](@ref) for the `dt <= 0` case.
"""
function INFORM(log::InformativeLog, ctx, msg::Function, index, data)
    now = time_ns()
    log.dt > 0 && log.last !== nothing && (now - log.last) < log.dt * 1e9 && return nothing
    log.last = now

    io = log.io === nothing ? stderr : log.io
    print(io, log.prompt, " ", msg())
    index === nothing || print(io, " n=", length(index))
    data === nothing || print(io, " ", data)
    println(io, " mem=", ceil(Int, Base.gc_live_bytes() / 2^20), "MB max-rss=",
            ceil(Int, Sys.maxrss() / 2^20), "MB ", Dates.now())
    flush(io)
    nothing
end

"""
    CallbackLog(callback::Function)

The reference [`AbstractObserver`](@ref): calls `callback(index, sp, ep)` on every event it
receives. This is the mechanism to persist, checkpoint, or account for a range of ids the moment it
becomes part of the index.

The callback runs inline and its exceptions propagate: if the durable write fails, the insertion
that triggered it fails too, which is the correct behaviour for a write-ahead log. It is not called
concurrently -- see [`AbstractLog`](@ref)'s rules -- so it needs no locking of its own, but it does
belong to exactly one index.

# Examples

```julia
using SimilaritySearch

ranges = Tuple{Int,Int}[]
ctx = GenericContext(; observers=CallbackLog((index, sp, ep) -> push!(ranges, (Int(sp), Int(ep)))))
```
"""
struct CallbackLog <: AbstractObserver
    callback::Function
end

"""
    OBSERVE(log::CallbackLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)

Invokes `log.callback(index, sp, ep)`. See [`CallbackLog`](@ref).
"""
function OBSERVE(log::CallbackLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)
    log.callback(index, sp, ep)
    nothing
end

"""
    LOG(args...)

Removed. The single logging call was split in two, along with the single `ctx.logger` slot that
received it:

- [`OBSERVE`](@ref)`(ctx, event, index, sp, ep)` reports a structural event to `ctx.observers`.
- [`INFORM`](@ref)`(ctx, msg)` / [`@inform`](@ref) sends a free-form message to `ctx.reporters`.

A backend that used to implement `LOG` now implements one of the two, and declares itself
[`AbstractReporter`](@ref) or [`AbstractObserver`](@ref) accordingly. `LogList` is gone as well: the
context field is already a list, and `reporters=[]` is how a context is silenced.
"""
LOG(args...) = throw(ArgumentError(
    "LOG was split into OBSERVE (structural events, ctx.observers) and INFORM/@inform " *
    "(free-form messages, ctx.reporters); see the LOG docstring for how to migrate a backend"))
