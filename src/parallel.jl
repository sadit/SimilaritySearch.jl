# This file is part of SimilaritySearch.jl

export @BATCHES, @BEGIN, @BEGINBATCH, @LOOP, @ENDBATCH, @END, @nbatches, @batchid,
       set_batch_scheduler!, get_batch_scheduler

# --- marker macros -----------------------------------------------------------------
#
# These are only ever meant to be consumed, unexpanded, directly out of the raw syntax
# tree by @BATCHES's own section-splitter (see `_batches_parse_body`) before anything is
# macro-expanded. They exist as real macros purely as a defensive fallback: if a marker
# ends up somewhere the splitter doesn't recognize it (nested inside an `if`/`let`,
# duplicated, used standalone outside of any @BATCHES call, ...), normal macro expansion
# reaches these definitions instead of leaving a confusing UndefVarError/MethodError.

"""
    @BEGIN

Marks the start of the `@BATCHES` section that runs once, in the caller's own scope,
before any batch starts. See [`@BATCHES`](@ref).
"""
macro BEGIN(args...)
    error("@BEGIN may only appear as a bare, top-level marker directly inside a @BATCHES block (used standalone, nested inside another construct, or duplicated?)")
end

"""
    @BEGINBATCH

Marks the start of the `@BATCHES` section that runs once per batch, at the start of that
batch's task, before its `@LOOP` iterations. See [`@BATCHES`](@ref).
"""
macro BEGINBATCH(args...)
    error("@BEGINBATCH may only appear as a bare, top-level marker directly inside a @BATCHES block (used standalone, nested inside another construct, or duplicated?)")
end

"""
    @LOOP for i in range ... end

Marks the mandatory `@BATCHES` section: the per-element body, run once for every element
of the batch's chunk. Must be immediately followed by exactly one `for i in range ... end`
loop. See [`@BATCHES`](@ref).
"""
macro LOOP(args...)
    error("@LOOP may only appear as a top-level marker, immediately followed by a `for i in range ... end` loop, directly inside a @BATCHES block")
end

"""
    @ENDBATCH

Marks the start of the `@BATCHES` section that runs once per batch, after that batch's
`@LOOP` iterations finish (same task, before it joins). See [`@BATCHES`](@ref).
"""
macro ENDBATCH(args...)
    error("@ENDBATCH may only appear as a bare, top-level marker directly inside a @BATCHES block (used standalone, nested inside another construct, or duplicated?)")
end

"""
    @END

Marks the start of the `@BATCHES` section that runs once, in the caller's own scope,
after all batches have joined. See [`@BATCHES`](@ref).
"""
macro END(args...)
    error("@END may only appear as a bare, top-level marker directly inside a @BATCHES block (used standalone, nested inside another construct, or duplicated?)")
end

# --- context macros ----------------------------------------------------------------

"""
    @nbatches()

Inside a [`@BATCHES`](@ref) call (any section: `@BEGIN`, `@BEGINBATCH`, `@LOOP`,
`@ENDBATCH`, `@END`), expands to the total number of batches/chunks used for that call
(always `>= 1`; `1` when the fast/serial path was taken). Typically used in `@BEGIN` to
size a shared, `@batchid()`-indexed array. Raises `UndefVarError` if used outside of
`@BATCHES`.
"""
macro nbatches()
    esc(:__batch_nbatches)
end

"""
    @batchid()

Inside a `@BATCHES` call's `@BEGINBATCH`, `@LOOP`, or `@ENDBATCH` section, expands to the
current batch's fixed, 1-based ordinal index (stable for the whole lifetime of that
batch's task). Since batch ids are disjoint -- no two concurrently-running batches ever
share one -- indexing a shared, `@nbatches()`-sized array by `@batchid()` is race-free by
construction, regardless of scheduler (`:static`/`:default`/`:greedy`); this is safer than
indexing by `Threads.threadid()`, which can alias/migrate under non-`:static` schedulers.
Not meaningful in `@BEGIN`/`@END` (those run once, globally, not per batch) -- using it
there raises `UndefVarError`.

!!! note "Always call it like a function: `@batchid()`, not bare `@batchid`"
    A bare (parenthesis-free) macro call followed directly by a unary `-` is parsed as
    the macro being passed that `-...` as an argument, not as subtraction on its result --
    e.g. `2 * @batchid - 1` parses as `2 * @batchid(-1)`, which errors. This is why every
    signature, docstring, and call site in this package writes `@batchid()`/`@nbatches()`
    with explicit empty parentheses, even though they take no arguments: it is exactly
    equivalent to the bare form, but the `()` closes the argument list right at the call
    site, so a following `- 1` can never be swallowed into it. Prefer that style over
    wrapping a bare call in parentheses yourself (`(@batchid) - 1`) -- it reads as what it
    is, a function-like call, and cannot be misparsed regardless of what follows it.
"""
macro batchid()
    esc(:__batch_id)
end

# --- global scheduler selection -----------------------------------------------------

"""
    SCHEDULER

Global selector (a `Ref{Symbol}`, seeded at package load time from the
`SIMSEARCH_BATCH_SCHEDULER` environment variable, default `:static`) for the
`Threads.@threads` scheduler kind used by [`@BATCHES`](@ref) when a call site does not
specify its own `scheduler=` override. One of `:default`, `:static`, `:greedy`,
`:sequential` (`:greedy` only valid on Julia >= 1.11). Read/write it via
[`get_batch_scheduler`](@ref)/[`set_batch_scheduler!`](@ref) rather than directly, since
the latter validates its argument.
"""
const SCHEDULER = Ref{Symbol}(:static)

"""
    set_batch_scheduler!(sched::Symbol)

Sets the global `Threads.@threads` schedule kind used by [`@BATCHES`](@ref) whenever a
call site does not give its own `scheduler=` override. Must be one of:

- `:static` (**the default**): one task per thread, never migrates mid-execution. This
  package no longer has any `Threads.threadid()`-indexed shared state on its own parallel
  paths: `searchgraph/context.jl`'s `vstates`/`beams`, `searchgraph/rebuild.jl`,
  `searchgraph/insertions.jl`, `closestpair.jl`, and `exact/parallel-exhaustive.jl` all use
  `@batchid()`-indexing (safe under every scheduler); `dist/seqs.jl`'s `Levenshtein`/`LCS`,
  which can't reach a `@batchid()` at all (their scratch buffer is needed inside the
  generic, context-free `evaluate(dist, a, b)`), use a `Channel`-based buffer pool
  instead of thread-indexing. `:static` remains the default for its simpler, more
  predictable scheduling, not because anything in this package still depends on it for
  correctness. Trade-off: throws immediately if a `@BATCHES` call is ever nested inside
  another already-threaded region, or invoked from a non-main thread.
- `:dynamic`/`:default`: whatever `Threads.@threads` itself currently defaults to
  (currently `:dynamic`; passed through as `:default` here so this package does not hard-
  code a name that Julia itself reserves the right to change).
- `:greedy`: spawns up to `Threads.threadpoolsize()` tasks that each greedily pull the
  next batch of work as they finish; best for very uneven per-batch cost. **Requires
  Julia >= 1.11** (raises `ArgumentError` on older versions, at the point this is set, not
  merely when a `@BATCHES` call later tries to use it).
- `:sequential`: disables threading entirely. Every `@BATCHES` call site that does not give
  its own `scheduler=` override runs its whole range as a single batch, in the caller's own
  task -- exactly the existing small-`n`/single-thread fast path, just forced regardless of
  `Threads.nthreads()` or how `range` compares to `minbatch`. [`@nbatches()`](@ref) is `1`
  and [`@batchid()`](@ref) is `1` for the entire call.

!!! warning
    `:default`/`:greedy` use migratable `Task`s: `Threads.threadid()` can change *during*
    a single batch's execution. Switching away from `:static` is **unsafe** for any code
    that indexes per-thread state by `Threads.threadid()` -- unlike `:static`'s nesting
    restriction, this failure mode is a **silent data race**, not an error. Nothing in
    this package's own `@BATCHES`-parallelized paths does this anymore (see above); this
    still matters for any *new* code you write. Prefer indexing by [`@batchid()`](@ref)
    (safe under every scheduler); when no `@batchid()` is reachable at all (e.g. a
    context-free interface like `evaluate`), use a `Channel`-based buffer pool instead
    (see `dist/seqs.jl`'s `Levenshtein`) -- both avoid `Threads.threadid()` entirely.

See also [`get_batch_scheduler`](@ref).
"""
function set_batch_scheduler!(sched::Symbol)
    sched === :default || sched === :static || sched === :greedy || sched === :sequential ||
        throw(ArgumentError("invalid @BATCHES scheduler `:$sched`; expected :default, :static, :greedy, or :sequential"))
    sched === :greedy && VERSION < v"1.11" &&
        throw(ArgumentError("@BATCHES: scheduler=:greedy requires Julia >= 1.11 (native Threads.@threads :greedy does not exist before that)"))
    SCHEDULER[] = sched
end

"""
    get_batch_scheduler() -> Symbol

Returns the current global scheduler used by [`@BATCHES`](@ref) when a call site does not
specify its own `scheduler=` override. One of `:default`, `:static`, `:greedy`,
`:sequential`. See [`set_batch_scheduler!`](@ref) for what each means and how to change it.
"""
get_batch_scheduler() = SCHEDULER[]

function __init__()
    s = Symbol(get(ENV, "SIMSEARCH_BATCH_SCHEDULER", "static"))
    if s === :static || s === :default || s === :sequential || (s === :greedy && VERSION >= v"1.11")
        SCHEDULER[] = s
    else
        @warn "unrecognized or unsupported SIMSEARCH_BATCH_SCHEDULER=$(repr(String(s))); falling back to :static" maxlog=1
        SCHEDULER[] = :static
    end
end

# --- dispatch targets: ordinary, hand-written functions, never macro-generated -----
#
# CRITICAL: Threads.@threads must only ever appear inside a plain, hand-written function
# body like these, never spliced directly into @BATCHES's own generated quote/esc() tree.
# An earlier draft that inlined `Threads.@threads for id in ...; f(id); end` straight into
# the macro's own AST (with `id` an internal, unescaped bookkeeping symbol) passed casual
# testing but failed ~2.5% of trials in a randomized stress test with silently wrong
# results, no error: Threads.@threads re-emits its loop variable as part of its own
# generated per-iteration binding, and an unescaped symbol coming from another macro's
# hygiene context resolved to a single shared variable instead of a fresh per-task local,
# so concurrent tasks raced on it. Ordinary function bodies are never touched by macro
# hygiene tagging, so keeping Threads.@threads confined to plain functions like these
# sidesteps the whole bug class by construction.

function _batches_run_static(f::F, n::Int) where {F}
    Threads.@threads :static for id in 1:n
        f(id)
    end
end

function _batches_run_default(f::F, n::Int) where {F}
    Threads.@threads for id in 1:n
        f(id)
    end
end

@static if VERSION >= v"1.11"
    function _batches_run_greedy(f::F, n::Int) where {F}
        Threads.@threads :greedy for id in 1:n
            f(id)
        end
    end
else
    function _batches_run_greedy(::F, ::Int) where {F}
        error("@BATCHES: scheduler=:greedy requires Julia >= 1.11 (native Threads.@threads :greedy does not exist before that)")
    end
end

function _batches_dispatch(f::F, n::Int, sched::Symbol) where {F}
    if sched === :greedy
        _batches_run_greedy(f, n)
    elseif sched === :static
        _batches_run_static(f, n)
    else
        _batches_run_default(f, n)
    end
end

# Runtime counterpart of the eager, macro-expansion-time validation `_batches_parse_args`
# performs on a literal `scheduler=:sym` -- used for a `scheduler=<expr>` call site (a
# variable, `ctx.scheduler`, a function call, ...) whose value can only be known once the
# expression is actually evaluated.
function _batches_validate_scheduler(sched::Symbol)
    (sched === :default || sched === :static || sched === :greedy || sched === :sequential) ||
        throw(ArgumentError("@BATCHES: `scheduler` must be one of :default, :static, :greedy, :sequential; got `:$sched` (resolved at run time)"))
    sched === :greedy && VERSION < v"1.11" &&
        throw(ArgumentError("@BATCHES: scheduler=:greedy requires Julia >= 1.11"))
    sched
end

# --- macro argument parsing (no MacroTools needed) ----------------------------------

function _batches_parse_args(args)
    length(args) >= 2 ||
        throw(ArgumentError("@BATCHES requires a minbatch expression and a for-loop or begin...end block, e.g. `@BATCHES minbatch for i in range ... end`"))
    minbatch_expr = args[1]
    if Meta.isexpr(minbatch_expr, :(=), 2) && minbatch_expr.args[1] === :minbatch
        throw(ArgumentError("@BATCHES: minbatch is now a positional argument -- write `@BATCHES $(minbatch_expr.args[2]) for ...` instead of `@BATCHES minbatch=... for ...`"))
    end
    body_ex = args[end]
    scheduler = nothing
    for kw in args[2:end-1]
        Meta.isexpr(kw, :(=), 2) ||
            throw(ArgumentError("@BATCHES: expected `key=value`, got `$kw`"))
        key, val = kw.args
        if key === :scheduler
            if val isa QuoteNode && val.value isa Symbol
                # literal, e.g. `scheduler=:static` -- validate eagerly and keep it known at
                # macro-expansion time (lets `dispatch`/`sequential_expr` below specialize
                # without any runtime branch). Kept wrapped in its `QuoteNode` so it stays
                # distinguishable from the "runtime expression" case just below -- a bare
                # variable name (`scheduler=myvar`) is *also* a plain `Symbol` at the AST
                # level, but never a `QuoteNode`.
                sym = val.value
                (sym === :default || sym === :static || sym === :greedy || sym === :sequential) ||
                    throw(ArgumentError("@BATCHES: `scheduler` must be one of :default, :static, :greedy, :sequential; got `$val`"))
                sym === :greedy && VERSION < v"1.11" &&
                    throw(ArgumentError("@BATCHES: scheduler=:greedy requires Julia >= 1.11"))
                scheduler = val
            else
                # arbitrary expression (a variable, `ctx.scheduler`, a function call, ...) --
                # cannot be validated until it is evaluated, so it is escaped and checked once
                # at run time via `_batches_validate_scheduler` instead.
                scheduler = val
            end
        elseif key === :per
            throw(ArgumentError("@BATCHES: `per` is not a recognized keyword (per=thread/core was a Polyester-only concept; @BATCHES no longer uses Polyester at all)"))
        else
            throw(ArgumentError("@BATCHES: unrecognized keyword `$key`"))
        end
    end
    minbatch_expr, scheduler, body_ex
end

function _batches_for_parts(ex)
    Meta.isexpr(ex, :for, 2) ||
        throw(ArgumentError("@BATCHES: @LOOP must wrap a single `for i in range ... end` loop"))
    header = ex.args[1]
    Meta.isexpr(header, :(=), 2) ||
        throw(ArgumentError("@BATCHES only supports a single induction variable (`for i in range`); block-form `for i in A, j in B` is not supported"))
    loopvar = header.args[1]
    loopvar isa Symbol ||
        throw(ArgumentError("@BATCHES only supports a plain loop variable, got `$loopvar`"))
    loopvar, header.args[2], ex.args[2]
end

const _BATCHES_MARKERS = (:BEGIN, :BEGINBATCH, :LOOP, :ENDBATCH, :END)
_marker_macroname(name::Symbol) = Symbol("@", name)

_is_marker_call(ex, name::Symbol) =
    Meta.isexpr(ex, :macrocall) && ex.args[1] === _marker_macroname(name)

# Splits the raw block passed to @BATCHES into its (up to 5) sections. Returns
# (beginblock, beginbatchblock, loopvar, range, loopbody, endbatchblock, endblock), with
# each `*block` either `nothing` (section absent -- no code generated for it) or an
# `Expr(:block, ...)` of that section's raw statements.
function _batches_parse_body(ex)
    if Meta.isexpr(ex, :for, 2)
        # backward-compatible simple form: no markers at all, the whole thing is @LOOP
        loopvar, range, loopbody = _batches_for_parts(ex)
        return nothing, nothing, loopvar, range, loopbody, nothing, nothing
    end

    Meta.isexpr(ex, :block) ||
        throw(ArgumentError("@BATCHES: expected a `for i in range ... end` loop, or a `begin ... end` block containing @BEGIN/@BEGINBATCH/@LOOP/@ENDBATCH/@END sections"))

    seen = Dict{Symbol,Bool}(name => false for name in _BATCHES_MARKERS)
    buffers = Dict{Symbol,Vector{Any}}(name => Any[] for name in _BATCHES_MARKERS)
    order = Symbol[]
    current = nothing

    for stmt in ex.args
        matched = false
        for name in _BATCHES_MARKERS
            _is_marker_call(stmt, name) || continue
            seen[name] && throw(ArgumentError("@BATCHES: `@$name` may only appear once"))
            seen[name] = true
            push!(order, name)
            if length(stmt.args) == 3
                # "docked" form, e.g. `@LOOP for i in range ... end` or `@BEGIN begin ... end`
                arg = stmt.args[3]
                if name === :LOOP
                    Meta.isexpr(arg, :for, 2) ||
                        throw(ArgumentError("@BATCHES: @LOOP must be followed by a `for i in range ... end` loop"))
                    push!(buffers[name], arg)
                elseif Meta.isexpr(arg, :block)
                    append!(buffers[name], arg.args)
                else
                    push!(buffers[name], arg)
                end
                current = nothing
            elseif length(stmt.args) == 2
                current = name  # bare marker: subsequent statements belong to this section
            else
                throw(ArgumentError("@BATCHES: `@$name` does not accept arguments"))
            end
            matched = true
            break
        end
        matched && continue

        if stmt isa LineNumberNode && current === nothing
            continue
        end

        current === nothing &&
            throw(ArgumentError("@BATCHES: statement found outside of any @BEGIN/@BEGINBATCH/@LOOP/@ENDBATCH/@END section: `$stmt`"))
        push!(buffers[current], stmt)
    end

    let idx = 0
        for name in order
            pos = findfirst(==(name), _BATCHES_MARKERS)
            pos >= idx ||
                throw(ArgumentError("@BATCHES: sections must appear in order @BEGIN, @BEGINBATCH, @LOOP, @ENDBATCH, @END (got `@$name` out of order)"))
            idx = pos
        end
    end

    seen[:LOOP] || throw(ArgumentError("@BATCHES: @LOOP is mandatory"))
    loopstmts = buffers[:LOOP]
    length(loopstmts) == 1 && Meta.isexpr(loopstmts[1], :for, 2) ||
        throw(ArgumentError("@BATCHES: @LOOP's section must contain exactly one `for i in range ... end` loop"))
    loopvar, range, loopbody = _batches_for_parts(loopstmts[1])

    mkblock(name) = seen[name] ? Expr(:block, buffers[name]...) : nothing

    mkblock(:BEGIN), mkblock(:BEGINBATCH), loopvar, range, loopbody, mkblock(:ENDBATCH), mkblock(:END)
end

# --- @BATCHES itself -----------------------------------------------------------------

"""
    @BATCHES minbatch [scheduler=:default|:static|:greedy] for i in range ... end
    @BATCHES minbatch [scheduler=...] begin
        @BEGIN ... end            # optional, runs once, before dispatch
        @BEGINBATCH ... end       # optional, runs once per batch, before its elements
        @LOOP for i in range ... end   # mandatory
        @ENDBATCH ... end         # optional, runs once per batch, after its elements
        @END ... end              # optional, runs once, after all batches join
    end

Splits `range` into consecutive chunks ("batches") of (approximately) `minbatch`
elements each and processes each batch as one task, using `Threads.@threads` under the
selected [`scheduler`](@ref get_batch_scheduler). No `Polyester` dependency is involved
(unlike this package's earlier `@batch`-based macros).

The simple, one-argument form above (no `@BEGIN`/`@BEGINBATCH`/`@ENDBATCH`/`@END`) is
exactly equivalent to using only `@LOOP`; it exists so straightforward per-element loops
don't need any of the section machinery.

# Sections
- `@BEGIN`: runs once, in the *caller's own scope*, before any batch starts. Variables
  declared here are plain local variables of the enclosing function -- visible later in
  `@END`, and (via ordinary closure capture) inside every batch's `@BEGINBATCH`/`@LOOP`/
  `@ENDBATCH` too. [`@nbatches()`](@ref) is available here (typically to size a shared,
  per-batch array, e.g. `results = Vector{Float32}(undef, @nbatches())`).
- `@BEGINBATCH`: runs once **per batch**, at the start of that batch's task, before its
  `@LOOP` iterations. [`@batchid()`](@ref)/[`@nbatches()`](@ref) and `@BEGIN`'s variables are
  available.
- `@LOOP for i in range ... end`: **mandatory**. The per-element body, run once for every
  `i` in this batch's chunk of `range`. Shares one lexical/closure scope with
  `@BEGINBATCH`/`@ENDBATCH` (of the *same* batch), so a variable declared in
  `@BEGINBATCH` can be read and updated here directly.
- `@ENDBATCH`: runs once **per batch**, after that batch's `@LOOP` iterations finish
  (same task, before it joins). Sees `@BEGIN`'s variables plus whatever `@BEGINBATCH`/
  `@LOOP` left in the per-batch scope. Writing into `results[@batchid()]` here is race-free
  by construction (batch ids are disjoint, unlike `Threads.threadid()` which can
  alias/migrate under non-`:static` schedulers -- see [`set_batch_scheduler!`](@ref)).
- `@END`: runs once, in the *caller's own scope*, after **all** batches have joined. Sees
  `@BEGIN`'s variables (e.g. to reduce the now fully-populated `results` array).

Sections that are omitted generate no code at all. When present, sections must appear in
the order `@BEGIN`, `@BEGINBATCH`, `@LOOP`, `@ENDBATCH`, `@END` (each except `@LOOP` may
be individually omitted).

# Arguments
- `minbatch`: (approximate) number of iterations processed per batch; the first
  positional argument. Use [`getminbatch`](@ref) to compute a reasonable value (aims for
  ~8 batches per thread) instead of hand-picking one.
- `scheduler`: overrides the global [`get_batch_scheduler`](@ref)/
  [`set_batch_scheduler!`](@ref) selection for this call site only. One of `:default`,
  `:static`, `:greedy`, or `:sequential` -- `scheduler=:sequential` forces this call site to
  run its whole `range` as a single, unthreaded batch (`@nbatches()` is `1`, `@batchid()` is
  `1`), regardless of `Threads.nthreads()` or how `range` compares to `minbatch`; see
  [`set_batch_scheduler!`](@ref). May be given either as a literal (`scheduler=:static`,
  validated immediately, at macro-expansion time) or as an arbitrary runtime expression --
  e.g. `scheduler=ctx.scheduler` for a context-typed caller that stores its own scheduler
  choice (see [`GenericContext`](@ref)/[`SearchGraphContext`](@ref)) -- which is evaluated
  and validated once, right before this call's batches start.

!!! warning
    **`:static` is the global default scheduler; switching to `:default`/`:greedy` is
    unsafe for code that indexes per-thread state by `Threads.threadid()`** (a silent
    data race, not an error, since those two schedulers use migratable `Task`s). Prefer
    [`@batchid()`](@ref)-indexed scratch space in new code -- it is safe under every
    scheduler. See [`set_batch_scheduler!`](@ref) for the full explanation.

!!! danger "The tagged-handle hazard: passing the wrong *object*, not the wrong index"
    A second, more insidious hazard shows up whenever `@batchid()`-indexed state is resolved
    **indirectly**, through a shared object that a callee re-derives batch-local state
    from several call frames below where the batch was tagged -- e.g.
    `searchgraph/context.jl`'s `getvstate`/`getbeam`, which read `ctx.batchid` deep inside
    `find_neighborhood!`/`search`, not at the `@BATCHES` call site itself (see
    `SearchGraphContext`). The pattern that makes this safe is: mint a tagged, per-batch
    copy once in `@BEGINBATCH` (`bctx = @set ctx.batchid = @batchid()`, via `Accessors.@set`)
    and use *that* copy -- never the original, outer object -- for every call made from
    inside that batch. **If even one call inside `@LOOP`/`@ENDBATCH` is accidentally
    passed the untagged original instead of the tagged copy, every batch silently
    resolves to the same hardcoded slot** (whatever the untagged object's default
    `batchid` is, typically `1`). This is unsafe under **every** scheduler, including
    `:static` -- unlike `Threads.threadid()`-aliasing above, it has nothing to do with
    task migration: batches running concurrently on genuinely different threads simply
    all read and write the same slot. It type-checks, compiles, and runs without error,
    returning plausible-looking (just silently wrong/corrupted) results, so it is easy to
    miss in a quick test. A real instance of exactly this bug was caught and fixed in
    `searchgraph/rebuild.jl` and `searchgraph/insertions.jl` during development: both
    files' `@BEGINBATCH` correctly minted `bctx`, but their `@LOOP` body still called
    `find_neighborhood!(N, g, ctx, ...)` (the outer, untagged context) instead of
    `find_neighborhood!(N, g, bctx, ...)` (the tagged one) --

    ```julia
    # BUGGY: every batch's find_neighborhood! call resolves getvstate/getbeam via the
    # SAME outer `ctx` (batchid always 1) -- a live race across concurrently-running
    # batches, on every scheduler, despite `tmp`/`N` themselves being correctly
    # @batchid()-sliced right above it.
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
        tmp = knnqueue(bctx, view(qcache, 1:ksearch, 2 * @batchid() - 1))
        N = knnqueue(bctx, view(qcache, 1:ksearch, 2 * @batchid()))
    @LOOP for objID in 1:n
        find_neighborhood!(N, g, ctx, database(g, objID), tmp, 1:-1; hints=...)  # bug: ctx, not bctx
    end

    # FIXED
    @LOOP for objID in 1:n
        find_neighborhood!(N, g, bctx, database(g, objID), tmp, 1:-1; hints=...)
    end
    ```

    **When reviewing/writing a `@BATCHES` body that mints a tagged per-batch handle, grep
    the diff for the original untagged variable's name inside `@LOOP`/`@ENDBATCH` -- it
    should not appear there at all.**

!!! note "Julia 1.10 and stack-allocated scratch buffers"
    This macro no longer uses `Polyester.@batch` at all (on any Julia version), so its
    stack-allocated, non-GC-tracked `threadlocal=`-style buffers are not available here.
    Since v0.15, `Polyester`/`StrideArraysCore` are no longer dependencies of this
    package at all (removed for Julia 1.12+ compatibility and better static/binary
    deployment support). If you relied on that for performance, initialize your own
    `@BEGIN`/`@BEGINBATCH` scratch arrays as a `StrideArraysCore.PtrArray` instead of a
    plain `Array` to get comparable non-GC-tracked, stack-friendly behavior -- you'll
    need to add `StrideArraysCore` to your own project's dependencies to do so.

# Examples

```julia
julia> using SimilaritySearch

julia> n = 100_000; out = zeros(Int, n);

julia> @BATCHES getminbatch(n) for i in 1:n
           out[i] = i^2
       end

julia> out == [i^2 for i in 1:n]
true

julia> function sumsq(n, minbatch)
           local total
           @BATCHES minbatch begin
               @BEGIN
                   partial = zeros(Float64, @nbatches())
               @BEGINBATCH
                   acc = 0.0
               @LOOP for i in 1:n
                   acc += abs2(i)
               end
               @ENDBATCH
                   partial[@batchid()] = acc
               @END
                   total = sum(partial)
           end
           total
       end;

julia> sumsq(1000, getminbatch(1000)) == sum(abs2, 1:1000)
true
```
"""
macro BATCHES(args...)
    minbatch_expr, scheduler, body_ex = _batches_parse_args(Any[args...])
    beginblock, beginbatchblock, loopvar, range, loopbody, endbatchblock, endblock =
        _batches_parse_body(body_ex)

    beginbatch_code = beginbatchblock === nothing ? nothing : esc(beginbatchblock)
    endbatch_code = endbatchblock === nothing ? nothing : esc(endbatchblock)
    begin_code = beginblock === nothing ? nothing : esc(beginblock)
    end_code = endblock === nothing ? nothing : esc(endblock)

    # `scheduler`, from `_batches_parse_args`, is one of:
    #   - `nothing`             : no override -- use the mutable global `SCHEDULER[]`
    #   - `QuoteNode(sym)`      : a literal (`scheduler=:static`), already validated, known
    #                             at macro-expansion time
    #   - anything else         : a runtime expression (`scheduler=ctx.scheduler`, a
    #                             variable, a call, ...), escaped and validated once via
    #                             `_batches_validate_scheduler` when this call runs
    scheduler_sym = scheduler isa QuoteNode ? scheduler.value : nothing
    scheduler_is_runtime = scheduler !== nothing && !(scheduler isa QuoteNode)

    dispatch = if scheduler_sym === :greedy
        :(_batches_run_greedy(__batch_f, $(esc(:__batch_nbatches))))
    elseif scheduler_sym === :static
        :(_batches_run_static(__batch_f, $(esc(:__batch_nbatches))))
    elseif scheduler_sym === :default
        :(_batches_run_default(__batch_f, $(esc(:__batch_nbatches))))
    elseif scheduler_sym === :sequential
        nothing  # never reached -- __batch_sequential forces __batch_fastpath below
    elseif scheduler_is_runtime
        :(_batches_dispatch(__batch_f, $(esc(:__batch_nbatches)), __batch_scheduler_runtime))
    else
        :(_batches_dispatch(__batch_f, $(esc(:__batch_nbatches)), SCHEDULER[]))
    end

    # Whether this call site is forced to skip threading entirely, i.e. run as a single
    # batch regardless of `Threads.nthreads()`/`range` vs `minbatch`. Known at macro-
    # expansion time when `scheduler=:sequential` is given directly; otherwise (no override,
    # or a runtime expression) it can only change at run time, so it is re-checked there.
    sequential_expr = if scheduler_sym === :sequential
        true
    elseif scheduler_is_runtime
        :(__batch_scheduler_runtime === :sequential)
    elseif scheduler === nothing
        :(SCHEDULER[] === :sequential)
    else
        false
    end

    # Evaluates a runtime `scheduler=<expr>` exactly once (before any batch starts) and
    # validates it; absent for the literal/no-override cases, which need no such binding.
    runtime_sched_code = scheduler_is_runtime ?
        :(__batch_scheduler_runtime = _batches_validate_scheduler($(esc(scheduler)))) : nothing

    quote
        $(runtime_sched_code)
        __batch_range = $(esc(range))
        __batch_n = length(__batch_range)
        __batch_minbatch = max(1, Int($(esc(minbatch_expr))))
        __batch_sequential = $sequential_expr
        __batch_fastpath = __batch_sequential || Threads.nthreads() == 1 || __batch_n <= __batch_minbatch
        __batch_parts = __batch_fastpath ? nothing : collect(Iterators.partition(__batch_range, __batch_minbatch))
        $(esc(:__batch_nbatches)) = __batch_fastpath ? 1 : length(__batch_parts)

        __batch_f = function ($(esc(:__batch_id)),)
            __batch_chunk = __batch_fastpath ? __batch_range : __batch_parts[$(esc(:__batch_id))]
            $(beginbatch_code)
            for $(esc(loopvar)) in __batch_chunk
                $(esc(loopbody))
            end
            $(endbatch_code)
        end

        $(begin_code)

        if __batch_fastpath
            __batch_f(1)
        else
            $dispatch
        end

        $(end_code)
        nothing
    end
end
