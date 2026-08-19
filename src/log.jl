# This file is part of SimilaritySearch.jl

import Dates
export AbstractLog, InformativeLog, LOG

"""
    abstract type AbstractLog end

Base type for logging backends. A logger is stored (e.g., as `ctx.logger`) in a context object (a subtype
of `AbstractContext`) and is passed to [`LOG`](@ref) by index operations (`push_item!`, `append_items!`,
`index!`, etc.) so that they can report their progress without depending on any particular logging backend.
Concrete subtypes must implement:

    LOG(log::MyLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)

# The `event` contract

`event` names *what happened*, not which Julia function was called -- it must not simply mirror the
name of the calling method (`:push_item!`, `:append_items!`, `:index!`, ...). It is one of a small,
curated set of event kinds:

- `:add!` -- a **structural** event: one or more objects were added to the index, and `sp:ep` is the
  exact, contiguous range of ids affected by this call. Every current index type (`SearchGraph`,
  `ExhaustiveSearch`/`ParallelExhaustiveSearch`, `InvertedFile`/`DictInvertedFile`, `BM25InvertedFile`,
  `Sat`) emits this same event for "objects were added," regardless of whether the call arrived via a
  single-item `push_item!` or a batch `append_items!`/`index!` -- one canonical name for one canonical
  kind of mutation, not one name per entry point.
- `:info` -- a **non-structural**, purely informative ping: nothing about the index actually changed
  (e.g. `index!` on a brute-force index where `db` already *is* the index, so there is nothing to
  build). A consumer that only cares about real mutations should ignore `:info` events entirely.

**Exactly-once**: when one mutating function calls another mutating function internally (e.g. an
`append_items!` that delegates its actual work to `index!`), only the function that performs/owns the
mutation may call `LOG` for that range -- a caller that purely delegates must stay silent, never log
the same range again under a different (or the same) event name. See `SimilaritySearch.InvertedFiles`'s
`append_items!`/`index!` (or `SearchGraph`'s, in `searchgraph/insertions.jl`) for the reference pattern:
the outer function delegates without logging, and the inner function it calls is the sole emitter.

**Why this precision matters**: every current index type is append-only (ids are assigned
monotonically and are never removed or modified), so a correctly-behaving stream of `:add!` events --
exactly one per logical batch, with an accurate, gap-free `sp:ep` -- is on its own sufficient for a
consumer to reconstruct or checkpoint which ids are durably indexed at any point in time. This is what
makes the mechanism usable as a write-ahead log for incremental or crash-recoverable indexing: a
consumer can replay the `:add!` stream instead of re-deriving state from the index itself. Duplicate
events, an event misnamed as if it were a different action, or an inaccurate `sp:ep` silently break
that invariant even though nothing looks wrong in `InformativeLog`'s printed output -- it never
validates `event`/`sp`/`ep`, it only prints them (see [`LOG(::InformativeLog, ...)`](@ref)).
"""
abstract type AbstractLog end

"""
    LogList(list::Vector{AbstractLog}=AbstractLog[InformativeLog()])

An [`AbstractLog`](@ref) backend that fans a single log event out to every logger in
`list`, in order -- use it to attach more than one logger (e.g. the default
[`InformativeLog`](@ref) plus a custom one) to the same context at once.
"""
struct LogList <: AbstractLog
    list::Vector{AbstractLog}

    LogList() = new(AbstractLog[InformativeLog()])
    LogList(v) = new(v)
end

"""
    LOG(log::LogList, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)

Forwards the log event to every logger contained in `log.list`. See [`LOG`](@ref) for the general contract.
"""
function LOG(log::LogList, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)
    for log in log.list
        LOG(log, event, index, ctx, sp, ep)
    end
end

"""
    InformativeLog(; dt::Float64=1.0, prompt::String="LOG")

An [`AbstractLog`](@ref) backend that prints a short status line to `stderr` reporting the current size of
the index, available/used memory, and a timestamp, throttled so that it prints at most once every `dt`
seconds (i.e., calls happening less than `dt` seconds after the previous printed message are silently
skipped). This avoids flooding the output when logging happens inside tight or highly parallel loops.

# Keyword Arguments
- `dt`: minimum number of seconds between two consecutive printed messages
- `prompt`: a string prefix printed at the beginning of every logged line (useful to tell apart loggers of different indexes/stages)

# Examples

```julia
using SimilaritySearch

logger = InformativeLog(; dt=2.0, prompt="[my-index]")
ctx = GenericContext(; logger)
```
"""
struct InformativeLog <: AbstractLog
    dt::Float64
    prompt::String
    last::Ref{Float64}
    lock::Threads.SpinLock

    InformativeLog(; dt::Float64=1.0, prompt="LOG") = new(dt, prompt, Ref(0.0), Threads.SpinLock())
end

function timed_log_fun(fun::Function, log::InformativeLog)
    if trylock(log.lock)
        now = time()
        if log.last[] + log.dt < now
            fun()
            log.last[] = now
        end

        unlock(log.lock)
    end
end

"""
    LOG(log::InformativeLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)

Prints, at most once every `log.dt` seconds, a status line to `stderr` with `event`, the type of `index`,
the given range `sp:ep` (the start and end positions of the operation being logged, e.g., of an
`append_items!` call), the current index size, memory usage, and a timestamp. Calls that arrive before the
throttling interval has elapsed since the previous printed message do nothing.
"""
function LOG(log::InformativeLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)
    timed_log_fun(log) do 
        n = length(index)
        mem = ceil(Int, Sys.total_memory() / 2^20)
        maxrss = ceil(Int, Sys.maxrss() / 2^20)
        println(stderr, log.prompt, " $event $(typeof(index)) sp=$sp ep=$ep n=$n mem=$(mem) max-rss=$(maxrss) $(Dates.now())")
    end
end
