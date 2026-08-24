# 2026-08-24 — splitting the log into reporters and observers

Design decided in full and **implemented the same day**; see *What the implementation changed*
at the end for where reality differed from the plan. The package version stays at `1.2.0` even
though the change breaks: no shims, no aliases, no deprecation path.

## What is wrong today

Every context carries one slot, `logger::AbstractLog`, and that slot serves two unrelated jobs at
once: rendering progress for a human, and reacting to a structural event so something durable
happens. The best evidence is that a downstream package already had to invent the distinction by
hand. `SimilaritySearchEngine.jl`, `index_engine.jl:1013`:

```julia
loggers = AbstractLog[InformativeLog()]                          # renders to stderr
log_io    === nothing || push!(loggers, FileLog(log_io))         # renders to a file
on_change === nothing || push!(loggers, CallbackLog(on_change))  # persists the range sp:ep
return length(loggers) == 1 ? loggers[1] : LogList(loggers)
```

Three objects of two different kinds in one list, and the comment above it has to *say in prose*
what the types cannot: "adding a `FileLog` changes nothing about what gets persisted."

Three concrete failures follow from the single slot:

- **Silencing destroys persistence.** There is no way to stop the console output of that context
  without rebuilding the list minus the informative entries — which requires knowing what is in it.
- **`LogList()` is not empty.** Its zero-argument constructor returns `[InformativeLog()]`, so the
  spelling that looks like "no logging" is the spelling that turns logging on.
- **`verbose` and the logger are independent knobs.** `opt.jl` and `hints.jl` print to `stderr`
  through `verbose(ctx)`, entirely outside the logger, so emptying the logger does not silence
  the library.

## The taxonomy

```julia
abstract type AbstractLog end                       # root; exists to name the taxonomy
abstract type AbstractReporter <: AbstractLog end   # receives INFORM
abstract type AbstractObserver <: AbstractLog end   # receives OBSERVE
```

The library ships exactly two concrete types: `InformativeLog <: AbstractReporter` and
`CallbackLog <: AbstractObserver`. `FileLog` and `CallbackLog` move up from the engine, which
deletes its own copies.

`LogList` is deleted — the context's vector *is* the list. `LOG` is kept only as a method that
throws, explaining the split, so a downstream update reads a reason instead of an `UndefVarError`.

## Two calls, two contracts

```julia
OBSERVE(ctx, event::Symbol, index, sp::Integer, ep::Integer)      # fans out over ctx.observers
OBSERVE(o::AbstractObserver, event::Symbol, index, ctx, sp, ep)   # what a backend implements

INFORM(ctx, msg; index=nothing, data=nothing)                     # fans out over ctx.reporters
INFORM(r::AbstractReporter, ctx, msg, index, data)                # what a backend implements
```

`OBSERVE` keeps the whole strict contract that `AbstractLog`'s docstring carries today: one
canonical event name per canonical kind of mutation, exactly-once, an accurate gap-free `sp:ep`,
so that a consumer can replay the stream as a write-ahead log instead of re-deriving state.

`INFORM` is deliberately loose: the message need not say what it affects, and neither `index` nor
`data` is required. `msg` reaches the backend as a zero-argument function; `INFORM(ctx, "text")`
wraps a literal. The backend realizes it *after* deciding it will emit, so a throttled message
never pays for its own interpolation. `data` is for a reporter that does not want text — a
service emitting JSON takes the `NamedTuple`, `stderr` takes the string.

A macro carries the call sites:

```julia
@inform ctx "inserted $sp:$ep, |voc|=$(vocsize(idx))"
```

expanding to `isempty(ctx.reporters) || INFORM(ctx, () -> ...)`. The guard is what makes silencing
free: with no reporters the closure is never even built. This matters because `invfile.jl:217`
informs **per item**.

**`:info` leaves the observation contract.** The two sites that emit it
(`sequential-exhaustive.jl:52`, `parallel-exhaustive.jl:121`) mean literally "nothing structural
happened" — that is an `INFORM`. The observer channel is then `:add!` only, and the invariant
stops carrying non-structural noise a consumer must learn to ignore.

**No sugar that fires both.** Seven sites will call `OBSERVE` and then `@inform`. A combined
`OBSERVE(...; inform=...)` was considered and rejected: it re-fuses exactly what the change
separates.

## `InformativeLog`: one type, and no lock

`FileLog`'s own docstring said it "prints the exact same throttled status line `InformativeLog`
does, but to a caller-given `io`". When a type is documented as *identical to another except for
one field*, the field is the parameter and the second type is redundant. They merge:

```julia
mutable struct InformativeLog <: AbstractReporter
    io::Union{Nothing,IO}     # nothing = whatever stderr is bound to at print time
    dt::Float64
    prompt::String
    last::Float64
end
```

- **`io` stays lazy by default.** `FileLog` captured the handle at construction; `InformativeLog`
  resolves `stderr` at print time, so `redirect_stderr` follows it. That is what capturing output
  in a test or a rendered tutorial page needs. `nothing` keeps that semantics, `InformativeLog(io)`
  gives the other.
- **`flush(io)` after writing**, which `InformativeLog` never needed for `stderr` and a file handle
  does.
- **The `Ref` goes away** with the struct becoming mutable.

### The lock is removed, and `dt <= 0` becomes meaningful

The current body drops messages by *two* independent mechanisms, and `dt` only governs one:

```julia
if trylock(log.lock)              # contention -> message discarded, silently
    now = time()
    if log.last[] + log.dt < now  # dt=0 degenerates to log.last[] < now
        fun(); log.last[] = now
    end
    unlock(log.lock)
end
```

So `dt = 0` does **not** guarantee delivery: `trylock` discards on contention regardless, the
comparison is strict `<` so two messages inside one `time()` tick lose one, and `time()` is wall
clock — an NTP step backwards freezes the logger until the clock catches up with `last`.

The lock turns out to be defending nothing. All twelve `LOG` call sites are outside any `@BATCHES`
block: `sat.jl:146,187` come after the batch loops opened at 134 and 169; `invfile.jl:262` comes
after `_index_block!`, which is where the two `@BATCHES` live (278, 283); `insertions.jl:70` is
after the block's `end end`; the remaining six are in `push_item!`/`append_items!`/`index!` with no
parallelism at all. The new `INFORM` sites are serial too: `opt.jl:160` runs under
`search_models(...; parallel=:none)` (`opt.jl:305`) — the parallel part is the `searchbatch`
inside — and `neardup_`'s three `@info` sit in a serial `for range in Iterators.partition(...)`.

Per the package's own design, `push_item!` and `append_items!` on one index never run
concurrently, nor overlap each other. That is a contract to **document, not to defend**. The
`trylock` was not defending it either — when the contract was violated it hid the violation by
dropping the message.

Without the lock the body becomes `dt > 0 && now - last < dt && return`, and the semantics is one
line: **`dt > 0` throttles by time and that becomes the only reason a message is ever lost;
`dt <= 0` does not throttle and loses none.** `time_ns()` replaces `time()`. The lock leak the
lock itself created — `trylock` / `fun()` / `unlock` with no `try`/`finally`, so an exception
inside the closure holds the spinlock forever and mutes that logger for the rest of the process —
disappears with it.

Cost to document next to `dt <= 0`: it makes the reporter a serialization point. Invisible at a
per-block site, real at `invfile.jl:217`, which runs per item.

## Three rules to write down

**Contributors: no `OBSERVE` or `INFORM` inside a `@BATCHES`.** The context *is* copied per batch —
`@set ctx.batchid = @batchid()` at `insertions.jl:57` and `allknn.jl:92` — and those copies share
the same reporter object by reference. Today that is harmless only because no call site is inside
a batch: the invariant is held by *where the calls are*, not by the shape of the data. Without a
lock, breaking the rule costs a duplicated or lost line, not corruption — Julia's `println` already
serializes the bytes. That benign failure mode is another argument against the lock.

**Sharing is asymmetric.** An observer belongs to one index: its state is *that* index's `sp:ep`,
and sharing it mixes two histories. A reporter should be shared — that is what makes one throttle
govern one console instead of three indexes fighting over the screen with a `dt` each.

**Propagation happens through keywords, not through a helper.** An earlier draft of this note
proposed a `reporting(ctx)` NamedTuple to splat into the child constructor. Keywords are better:
the default sits in the signature, where a reader sees it, and each site picks its own default
instead of obeying one global rule. Two shapes, by whether the method has a context to inherit
from.

*The method receives no context and decides which one to build.* `neardup(dist, X, ϵ; recall)`
picks a `SearchGraphContext` when `recall < 1.0` and a `GenericContext` otherwise, and
`closestpair_buildindex(dist, A, recall)` does the same — its docstring already says it mirrors
`neardup`'s pattern. There is nothing to inherit, so the logging configuration belongs to the
method's own signature:

```julia
neardup(dist, X, ϵ; recall=1.0, verbose=true,
        reporters=[InformativeLog()], observers=AbstractObserver[], kwargs...)
```

The reporter default is constructed *in the signature*, so it is evaluated once per call — never a
`const` global, which the no-sharing rule forbids. The observer default is empty: the library never
invents an observer. For `closestpair_buildindex` the keywords have to be threaded from the public
wrappers that call it (`bichromatic_kclosestpairs(dist, A, B; ...)` and its siblings), since those
are the entry points a user actually holds.

*The method receives a context and may build another.* Then the keyword defaults to the received
context, so the inheritance happens even when no caller passes anything:

```julia
function execute_callback!(index, ctx::SearchGraphContext, opt::EpsilonHints;
        reporters=ctx.reporters,
        observers=AbstractObserver[])   # deliberately not ctx.observers -- see below
    ...
    neardup(E, GenericContext(; reporters, observers), sample, ϵ)
```

This shape also survives a fixed dispatch protocol like `execute_callback!`, whose caller
(`execute_callbacks!`) will never pass the keywords: the default does the inheriting by itself.

**The default depends on the role, and for observers it is almost always empty.** A scratch index
built internally emits `:add!` for a *different* index; letting that reach the caller's
`CallbackLog` corrupts the view of a consumer reconstructing which ids are durably indexed. The
rule: **inherit observers only when the new context drives the same index the caller's observers
are already watching.** No current site qualifies — every internally built context drives a scratch
index — so today every observer default is `AbstractObserver[]`. Reporters inherit everywhere;
that is what makes silencing reach the whole call tree. A single slot cannot express this split,
which is a stronger argument for it than silencing is.

**Two of the five sites are inert and need nothing.** `staticindexing.jl:225` and
`multirand.jl:119` build a context used only for `searchbatch`/`searchbatch!`. Search never logs —
all twelve call sites are mutations — so those contexts emit nothing at all and need no keyword.
Recorded here so nobody "fixes" them later. The three live ones are `neardup.jl:75`/`:78`,
`datasetwrapper.jl:77` (both case A) and `hints.jl:313` (case B).

**And one of them floods today.** When `EpsilonHints` is the chosen hints callback,
`hints.jl:313` calls `neardup(E, GenericContext(), sample, ϵ)`, and `neardup_`'s own
`verbose::Bool=true` default fires three `@info` lines per invocation — through Julia's logging, so
they survive even a run that redirected `stderr` — while the callback runs again and again as the
graph grows. Nothing a user does to their own context can currently stop it.

## Rejected: compile-time-typed log lists

Considered replacing `Vector{AbstractReporter}` with a tuple / cons list so the element types fix
at compile time and the dispatch resolves statically. Rejected. Orders of magnitude, estimated,
not measured:

- On the path that **emits**, a message costs `Sys.maxrss()` (a `/proc` read, a syscall),
  `Dates.now()`, interpolation and a `println` — microseconds. Dynamic dispatch is tens of
  nanoseconds: two to three orders of magnitude below.
- On the **throttled** path the dispatch does dominate — ~50% relative overhead over a `time_ns()`
  and a comparison — but the absolute number is ~30–60 ns per reporter per call, and only one site
  calls per item.
- At real scale: the 105.3M-paragraph corpus, two reporters, is ~6 seconds spread over a job that
  runs for hours. The same `push_item!` walks the document's tokens and calls `sort_postinglist!`
  per token — tens of microseconds. The dispatch is ~0.1% of what that site already costs.

Against that, fixing the types costs two more parameters on four contexts:
`SearchGraphContext{KnnType,VSType}` goes to four, every method taking a context specializes per
logger combination (compile latency is already this package's bottleneck), the three existing
`constructorof` overrides — which exist precisely because phantom parameters break Accessors'
default reconstruction — get worse, `ContextPool` and `SearchGraphEngine` store contexts in fields
and would either parameterize in cascade or declare the field abstract and bring the dynamic
dispatch back anyway, and a reporter can no longer be added at runtime with `push!`.

For comparison, the field today is `logger::AbstractLog` — an abstract field, which is *worse*
than a vector: same dynamic dispatch plus a boxed access. The vector regresses nothing.

The only case that must be genuinely free is the silenced one, and the `@inform` guard already
gives that without touching any type.

## Site inventory

Ten structural `LOG` calls become `OBSERVE` unchanged in meaning: `insertions.jl:70,154`,
`invfile.jl:217,262`, `sat.jl:146,187`, `sequential-exhaustive.jl:36,44`,
`parallel-exhaustive.jl:106,114`.

Two `LOG(:info)` become `@inform`: `sequential-exhaustive.jl:52`, `parallel-exhaustive.jl:121`.

Sixteen loose `println`/`@info`/`@show` sites enter the channel; the full audit is in *What
enters the reporter channel*, below.

`verbose` stays as a verbosity *level*, not an output switch — the switch is `reporters=[]`. That
removes the need for the coupling (`verbose(ctx) && !isempty(ctx.reporters)`) that an earlier draft
of this design proposed: with `INFORM` as its own call, emptying the reporters silences by
construction.

## What gets deleted

- `src/searchgraph/log.jl`, entirely. It exists only because the reporter needed to know about
  neighborhoods: it dispatches `LOG(::InformativeLog, ::SearchGraph, ::SearchGraphContext)` to
  print neighborhood-size quantiles. With a free-form `INFORM` the insertion site — which does know
  about neighborhoods — supplies the text, and the quantiles are computed only if something will
  read them. A dispatch surprise goes away.
- `LogList` and its non-empty zero-argument constructor.
- The `:info` event kind, from the contract.
- `FileLog` and `CallbackLog` in `SimilaritySearchEngine.jl`, plus `_engine_logger` in full: it
  becomes a literal list.

## Tests

`test/testlog.jl` exists and covers the exactly-once `:add!` contract per index type with a
`RecorderLog`; it ports to `AbstractObserver` with a two-token change. What must be added:

- silencing is total — with `reporters=[]`, capturing `stderr` over a full `index!` of every index
  type yields nothing, *including* the `opt.jl`/`hints.jl` paths that today bypass the logger;
- silencing does not disturb observation — `reporters=[]` with an observer still records the exact
  `:add!` stream;
- `dt <= 0` loses nothing: N informs produce N lines;
- observers do not leak into internally-built contexts: an observer on the outer context records
  no event from the scratch index inside `hints.jl:313`;
- reporters do reach it: the same run with a capturing reporter sees the inner progress.

## What enters the reporter channel

Every `@info`, `@warn`, `@show` and `println` outside a `Base.show` method was audited. Result:
sixteen convert, three must not, three are dead code.

**Fourteen convert directly — the context is already in scope.**

- `opt.jl:160,240,244,308,312` and `hints.jl:389` are already written as `verbose(ctx) && ...`;
  they become `verbose(ctx) && @inform ctx ...`, mechanically.
- `neardup.jl:112,132,153` — `neardup_` holds a context *and* its own `verbose::Bool` keyword, so
  `verbose && @inform ctx ...`. These are the three that flood from `hints.jl:313` today.
- `staticindexing.jl:128,143,158` (`@info`) and `:161,174` (`@show`) fire on **every** call with no
  gate at all, and `@show` writes to `stdout`, which nothing in this design can silence. The reason
  they were never gated is visible in the signature: `index!(idx::SearchGraph, ::SearchGraphContext,
  ::Val{:knr}, ...)` — the context argument is **unnamed**, so the author had no `ctx` to test.
  Naming it converts all five.

**Two convert through a keyword — no context to inherit from.** `fft.jl:71` and `dnet.jl:69` take a
plain `verbose::Bool` and no context. This settles what the first draft of this note deferred: they
should take `reporters` as well, because every caller can supply it — `hints.jl:377` already passes
`ctx.verbose`, `staticindexing.jl:212` passes its own `verbose`, `neighborhood.jl:170` passes
`verbose=false`. Left alone they are the last corner of the library that ignores silencing.

**Three must stay as `@warn`.** `opt.jl:142` (recall below 0.3), `parallel.jl:188` (unrecognized
`SIMSEARCH_BATCH_SCHEDULER`), `basket-list.jl:24` (a non-metric distance in `BasketList`). The
principle: **the reporter channel carries progress; a warning carries diagnosis.** Routing a warning
into it means `reporters=[]` hides a problem the user needs to act on, and it throws away what
`@warn` already provides — log level, `maxlog`, the caller's active Julia logger, and `@test_logs`
in a test suite. Silence should cost you progress reporting, never a diagnosis.

One thing to fix in passing rather than convert: `opt.jl:143` is a bare `@show cov` next to that
warning, dumping to `stdout` unconditionally whenever recall drops below 0.3, with its siblings at
145/149/152 already commented out. It should be folded into the `@warn` message or removed, not
turned into an `@inform`.

**Three are dead code, found while auditing.** `umerge.jl:4`'s `show_list_state` is a debug dumper
with zero callers; `hints.jl:141` sits inside a `#=...=#` block; `SimilaritySearch.jl:379` is inside
a docstring example. Unrelated to this change, but worth a separate broom.

**And one live path is broken, found the same way.** `neighborhood.jl:170` calls
`fft(distance(G), S, k; threads=false, verbose=false, scheduler=:sequential)`, but `fft` has exactly
one method and it takes `start`, `verbose` and `scheduler` — no `threads`. Any call to
`KCentersNeighborhood`'s filter throws on the keyword. The type is exported, documented in
`docs/src/api.md` with a usage example, and has zero test coverage, which is why nobody noticed.
Also a separate fix, but it must not be "fixed" silently while editing that line for `verbose`.

**Excluded, and not logging at all:** roughly forty `println(io, prefix, ...)` calls across
`src/db/*`, `invfile.jl`, `SearchGraph.jl` and the SAT types. Those are `Base.show` implementations
writing to a caller-given `io`.

## The `verbose` keywords after the change

The rule that falls out: **a `verbose` keyword survives only where there is no context to read it
from.**

**Two get removed.** `neardup(idx, ctx, X, ϵ; ..., verbose=true)` and its `neardup_` duplicate
`ctx.verbose` while a context is right there in the signature — and that `true` default is exactly
what makes `hints.jl:313` flood. It becomes `verbose(ctx)`, and propagation then carries the
caller's silence into it. `index!(idx, ctx, ::Val{:knr}; ..., verbose=true)` exists only to forward
to `fft` at `staticindexing.jl:212`; it is also the signature whose context argument has to be
named anyway, so the two edits are the same edit.

**Three survive, all of them where no context exists.** `fft` and `dnet` gain `reporters` and keep
`verbose`: the pair is exactly what a context would have carried — a level and a destination — and
`verbose && @inform ...` is the same shape used everywhere else. In `neardup(dist, X, ϵ; ...)`, the
case-A wrapper, `verbose` is an argument for the context it builds, not a second channel.

**`ctx.verbose` itself stays.** It is tempting to drop it now that `reporters=[]` is the switch,
but on one channel with one `dt` a chatty per-configuration site competes for the same throttle
slot as the per-block insertion progress and can crowd it out. `verbose` decides who is allowed to
compete; the reporters decide whether anyone is heard at all.

Its defaults should be made uniform in passing: `GenericContext` defaults to `verbose=true` while
`SearchGraphContext` and `SatContext` default to `false`. All three become `false` — progress still
prints through the reporters, per-configuration optimization detail does not.

## Order of work

Eight steps. Each one leaves the tree in a state whose breakage is *known*, which matters because
the type rename breaks everything at once if the steps are interleaved.

1. **`src/log.jl`, rewritten.** The three abstract types; `OBSERVE`/`INFORM` and the `@inform`
   macro; `InformativeLog` merged with `FileLog` (lazy `io`, `flush`, mutable, no lock, `dt <= 0`
   meaning no throttle, `time_ns()`); `CallbackLog`; `LOG` reduced to a method that throws with an
   explanation. `LogList` deleted. `src/searchgraph/log.jl` deleted and its `include` dropped. At
   this point `src/` does not compile — nothing else has been touched yet.

2. **The four contexts.** `GenericContext` (`SimilaritySearch.jl`), `SearchGraphContext`
   (`searchgraph/context.jl`), `InvertedFileContext` (`invertedfiles/InvertedFiles.jl`),
   `SatContext` (`SpatialAccessTree/context.jl`): `logger` becomes `reporters`/`observers`, with
   the constructor normalizing `nothing` / one / a vector. `InvertedFileContext` gains `verbose`,
   which it lacks today, so the four are uniform. Copy constructors and the three `constructorof`
   overrides follow. Docstrings say how to silence.

3. **The twelve structural sites.** Ten `LOG` → `OBSERVE`; the two `:info` → `@inform`. The event
   contract in `AbstractLog`'s docstring loses `:info` and gains the threading rule (no `OBSERVE`
   or `INFORM` inside a `@BATCHES`) and the sharing rule (an observer belongs to one index; a
   reporter is meant to be shared). Compilation is restored here.

4. **The sixteen reporter sites**, per the audit above: `opt.jl` (five, plus folding `@show cov`
   into its `@warn`), `hints.jl:389`, `neardup_`'s three, `staticindexing.jl`'s five — which needs
   the context argument *named* in the `Val{:knr}` signature first — and `fft`/`dnet`, which gain a
   `reporters` keyword threaded from their three callers. The three `@warn`s are left alone,
   deliberately. The two redundant `verbose` keywords go here too (`neardup_`, `index!` for
   `Val{:knr}`).

5. **Propagation.** Case A: `neardup(dist, X, ϵ; ...)` and `closestpair_buildindex`, whose keywords
   thread up to the public wrappers that call it. Case B: `execute_callback!` for `EpsilonHints`.
   The two inert sites (`staticindexing.jl:225`, `multirand.jl:119`) are left untouched, on
   purpose.

6. **Tests.** Port `test/testlog.jl` to `AbstractObserver`, then add the five properties listed
   above. This is the first point where running the suite proves anything about the change.

7. **Documentation.** `docs/src/tutorial/logging.md` is a 142-line page built entirely on the
   old model — "Every context (`ctx.logger`) holds a logging backend" — and needs a rewrite, not a
   patch; it is also the page whose rendered output the silencing has to work for. Then
   `docs/src/api.md`, `docs/src/tutorial/persistence.md`, `docs/src/tutorial/index.md`, the context
   docstrings, and `AGENTS.md`'s architecture map, which names `src/log.jl` and the context types.

8. **`SimilaritySearchEngine.jl`.** Delete its `FileLog` and `CallbackLog` and all of
   `_engine_logger`; `create_engine`/`restore_engine` pass `reporters=`/`observers=` directly. Net
   negative in lines. Its own suite is the last gate.

Steps 1–5 are one commit's worth of work but should land as at least two: the mechanism (1–3) and
the sites (4–5). Steps 7 and 8 are separate commits regardless.

## What the implementation changed about this plan

**TextSearch.jl is a consumer too, and the plan missed it.** Both it and the engine are dev-pathed
into each other's manifests, so all three move together. It was small -- `bm25/invfile.jl:278,351`
became `OBSERVE` plus an `@inform`, and three test sites ported -- but one of them is the best
possible advertisement for the change: `testprofileindex.jl` had

```julia
quietctx() = InvertedFileContext(logger=SimilaritySearch.LogList(SimilaritySearch.AbstractLog[]))
```

which is now `InvertedFileContext(reporters=[])`.

**Three live paths turned out to be broken, all of them never exercised by a test.** They were
found by touching their signatures, not by looking for them:

- `neardup(dist, X, ϵ; recall < 1.0)` called `OptimizeParametes(...)` -- a typo, present in the
  docstring too, so the approximate branch of a public wrapper threw `UndefVarError`.
- `EpsilonHints`'s callback called `DistanceWithIdentifiers` unqualified; it lives in
  `Dist.Hacks`. The exported, documented hints callback threw on use. This surfaced because the new
  propagation test calls it directly.
- `KCentersNeighborhood`'s filter called `fft(...; threads=false, ...)`, and `fft` has exactly one
  method, which takes `start`, `verbose` and `scheduler`. An exported, documented neighborhood
  filter that could not run. `threads=false` was a leftover next to the `scheduler=:sequential`
  that already says the same thing, so removing it is the whole fix.

All three are fixed, and reported rather than fixed silently. What they have in common is worth
more than the fixes: each is a *variant* -- an alternative callback, an alternative filter, an
alternative recall branch -- and the test suite only ever exercises the defaults.

**A seventeenth reporter site.** `staticindexing.jl:212` wrapped its `fft` call in
`@time "fft (center selection)"`, which prints to `stdout` unconditionally and no design here can
silence. It became `@elapsed` plus an `@inform`.

**`@inform` accepts a bare reporter, not just a context or a vector.** `fft`/`dnet` default
`reporters=InformativeLog()`, and requiring them to normalize that into a vector first was churn
for nothing; `_reporters` gained a one-line method returning a 1-tuple.

**The two `SearchGraph` messages deliberately omit `index=`.** At both sites `index.len[]` has not
been updated yet, so a reporter reading `length(index)` would print `n=0`. The event's *position*
cannot move -- the engine's `direct_neighbors` depends on `:add!` firing before
`connect_reverse_links!` -- so the message drops the field instead, and `sp:ep` says everything
anyway.

**`Logging` joined the test target.** The silencing test asserts that a `reporters=[]` build writes
*nothing at all* to `stderr`, which means neutralizing Julia's own logger for the duration --
`with_logger(NullLogger())`. That is also the test that documents the boundary: a `@warn` is a
different channel and silence must never hide one.

### Result

`Pkg.test()` green on all three packages (SimilaritySearch's full suite, TextSearch's, and the
engine's 109/109), the Documenter build clean of any new warning, and the Quarto manual re-rendered
(`manual/docs/architecture.html`). `src/index_engine.jl` lost about 120 lines: its `FileLog`, its
`CallbackLog`, and all of `_engine_logger`.

## Deferred

- Staying at `1.2.0` means someone updating from `1.2.0` meets the throwing `LOG` with no version
  signal. If this ever goes to a registry, that is where the major bump belongs.
