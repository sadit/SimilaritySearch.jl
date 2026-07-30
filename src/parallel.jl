# This file is part of SimilaritySearch.jl

export @BATCH

@static if VERSION < v"1.11"

"""
    @BATCH [minbatch=N] for i in range ... end

On Julia < 1.11 (no native `Threads.@threads :greedy` scheduler exists yet), `@BATCH` is
literally `Polyester.@batch`: this macro forwards its arguments, unexamined, straight to
`Polyester.@batch`, so every keyword `Polyester.@batch` itself understands (`per=thread`,
`per=core`, `minbatch=N`, `threadlocal=`, `reduction=`, `stride=`, ...) works exactly as
documented there. See `?Polyester.@batch` for the full keyword reference.

On Julia >= 1.11, `@BATCH` is a different, native implementation on top of
`Threads.@threads` with a smaller, different keyword set (`minbatch=`/`scheduler=`, no
`per=`) — see that method's own docstring. Code that needs to run unmodified on both
needs to stick to the keywords common to both (in practice: just `minbatch=`).
"""
macro BATCH(args...)
    # Delegate to Polyester's own macro *function* directly (not via an unexpanded nested
    # macrocall Expr), passing through the same __source__/__module__ this macro itself
    # received. Polyester's macro then sees `args` exactly as the caller wrote them
    # (needed for its own internal `:(=)` kwarg pattern-matching to work) and applies its
    # own, already-correct hygiene relative to the original call site; nothing here needs
    # (or should) wrap the result in an extra esc() -- Julia's ordinary nested-macro
    # hygiene composition already handles this correctly.
    Polyester.var"@batch"(__source__, __module__, args...)
end

else # VERSION >= v"1.11"

export set_batch_scheduler!, get_batch_scheduler

"""
    SCHEDULER

Global selector (a `Ref{Symbol}`, seeded at package load time from the
`SIMSEARCH_BATCH_SCHEDULER` environment variable, default `:static`) for the
`Threads.@threads` scheduler kind used by [`@BATCH`](@ref) on Julia >= 1.11 when a call
site does not specify its own `scheduler=` override. One of `:default`, `:static`,
`:greedy`. Read/write it via [`get_batch_scheduler`](@ref)/[`set_batch_scheduler!`](@ref)
rather than directly, since the latter validates its argument.
"""
const SCHEDULER = Ref{Symbol}(:static)

"""
    set_batch_scheduler!(sched::Symbol)

Sets the global `Threads.@threads` schedule kind used by [`@BATCH`](@ref) on Julia >=
1.11 whenever a call site does not give its own `scheduler=` override (has no effect on
Julia < 1.11, where `@BATCH` always forwards to `Polyester.@batch` instead). Must be one
of:

- `:static` (**the default**): one task per thread, never migrates mid-execution. Chosen
  as the default specifically because it preserves the same non-migration guarantee this
  package's `Threads.threadid()`/`Threads.maxthreadid()`-indexed per-thread scratch
  buffers (e.g. in `searchgraph/context.jl`, `searchgraph/rebuild.jl`,
  `searchgraph/staticindexing.jl`, `searchgraph/insertions.jl`, `closestpair.jl`,
  `dist/hacks.jl`, `dist/seqs.jl`) already implicitly depend on today via Polyester.
  Trade-off: throws immediately if a `@BATCH` call is ever nested inside another already-
  threaded region, or invoked from a non-main thread.
- `:dynamic`/`:default`: whatever `Threads.@threads` itself currently defaults to
  (currently `:dynamic`; passed through as `:default` here so this package does not hard-
  code a name that Julia itself reserves the right to change).
- `:greedy`: spawns up to `Threads.threadpoolsize()` tasks that each greedily pull the
  next chunk of work as they finish; best for very uneven per-chunk cost.

!!! warning
    `:default`/`:greedy` use migratable `Task`s: `Threads.threadid()` can change *during*
    a single chunk's execution. Switching away from `:static` is **unsafe** for any code
    that indexes per-thread state by `Threads.threadid()` (see the files listed above) —
    unlike `:static`'s nesting restriction, this failure mode is a **silent data race**,
    not an error. Only switch schedulers for call sites you have checked do not rely on
    `threadid()`-indexed state.

See also [`get_batch_scheduler`](@ref).
"""
function set_batch_scheduler!(sched::Symbol)
    sched === :default || sched === :static || sched === :greedy ||
        throw(ArgumentError("invalid @BATCH scheduler `:$sched`; expected :default, :static, or :greedy"))
    SCHEDULER[] = sched
end

"""
    get_batch_scheduler() -> Symbol

Returns the current global scheduler used by [`@BATCH`](@ref) on Julia >= 1.11 when a
call site does not specify its own `scheduler=` override. One of `:default`, `:static`,
`:greedy`. See [`set_batch_scheduler!`](@ref) for what each means and how to change it.
"""
get_batch_scheduler() = SCHEDULER[]

function __init__()
    s = Symbol(get(ENV, "SIMSEARCH_BATCH_SCHEDULER", "static"))
    if s === :default || s === :static || s === :greedy
        SCHEDULER[] = s
    else
        @warn "unrecognized SIMSEARCH_BATCH_SCHEDULER=$(repr(String(s))); falling back to :static" maxlog=1
        SCHEDULER[] = :static
    end
end

# --- dispatch targets: one per literal scheduler, defined once (not per @BATCH call site) ---

function _batch_run_greedy(f::F, parts) where {F}
    Threads.@threads :greedy for chunk in parts
        f(chunk)
    end
end

function _batch_run_static(f::F, parts) where {F}
    Threads.@threads :static for chunk in parts
        f(chunk)
    end
end

function _batch_run_default(f::F, parts) where {F}
    Threads.@threads for chunk in parts
        f(chunk)
    end
end

function _batch_dispatch(f::F, parts, sched::Symbol) where {F}
    if sched === :greedy
        _batch_run_greedy(f, parts)
    elseif sched === :static
        _batch_run_static(f, parts)
    else
        _batch_run_default(f, parts)
    end
end

# --- macro argument parsing (no MacroTools needed) ---

function _batch_parse_kwargs(kwargs)
    minbatch_expr = 1
    scheduler = nothing  # nothing => resolve SCHEDULER[] at run time (no per-call override)
    for kw in kwargs
        Meta.isexpr(kw, :(=), 2) ||
            throw(ArgumentError("@BATCH: expected `key=value`, got `$kw`"))
        key, val = kw.args
        if key === :minbatch
            minbatch_expr = val
        elseif key === :scheduler
            sym = val isa QuoteNode ? val.value : val
            (sym isa Symbol && (sym === :default || sym === :static || sym === :greedy)) ||
                throw(ArgumentError("@BATCH: `scheduler` must be one of :default, :static, :greedy; got `$val`"))
            scheduler = sym
        elseif key === :per
            throw(ArgumentError("@BATCH: `per` is not a recognized keyword here (per=thread/core is a Polyester-only concept -- Threads.@threads has no core-vs-thread notion -- and is only valid inside a literal Polyester.@batch call); remove it when migrating a @batch call site to @BATCH"))
        else
            throw(ArgumentError("@BATCH: unrecognized keyword `$key`"))
        end
    end
    minbatch_expr, scheduler
end

function _batch_extract_for(ex)
    Meta.isexpr(ex, :for, 2) ||
        throw(ArgumentError("@BATCH must wrap a single `for i in range ... end` loop"))
    header = ex.args[1]
    Meta.isexpr(header, :(=), 2) ||
        throw(ArgumentError("@BATCH only supports a single induction variable (`for i in range`); block-form `for i in A, j in B` is not supported"))
    loopvar = header.args[1]
    loopvar isa Symbol ||
        throw(ArgumentError("@BATCH only supports a plain loop variable, got `$loopvar`"))
    loopvar, header.args[2], ex.args[2]
end

"""
    @BATCH [minbatch=N] [scheduler=:default|:static|:greedy] for i in range ... end

Evaluates the loop on multiple threads, using Julia's native `Threads.@threads`
(available since Julia 1.11) instead of `Polyester.@batch`. The iteration space is split
into consecutive chunks of (approximately) `minbatch` elements each, via
`Iterators.partition`, and each chunk is processed as one unit of work by a single
task/thread; the execution of the loop waits for all chunks to finish before continuing.

# Keyword Arguments
- `minbatch`: (approximate) number of iterations processed per chunk/task; defaults to
  `1`. Use [`getminbatch`](@ref) to compute a reasonable value (aims for ~8 chunks per
  thread) instead of hand-picking one.
- `scheduler`: overrides the global [`get_batch_scheduler`](@ref)/[`set_batch_scheduler!`](@ref)
  selection for this call site only. One of `:default`, `:static` (the global default),
  `:greedy`.

!!! warning
    **`:static` is the global default scheduler, and switching to `:default`/`:greedy` is
    unsafe for code that indexes per-thread state by `Threads.threadid()`** (a silent
    data race, not an error, since those two schedulers use migratable `Task`s). See
    [`set_batch_scheduler!`](@ref) for the full explanation and the list of files in this
    package that currently rely on `threadid()`-indexed buffers and must not be switched
    without individual review.

!!! note
    Unlike `Polyester.@batch`, `@BATCH` does not accept `per=thread`/`per=core` at all —
    `Threads.@threads` has no core-vs-hardware-thread concept — so a `per=` keyword
    raises `ArgumentError` here rather than being silently ignored. Drop it when
    migrating a call site from `@batch` to `@BATCH`.

# Examples

```julia
julia> using SimilaritySearch

julia> n = 100_000; out = zeros(Int, n);

julia> @BATCH minbatch=getminbatch(n) for i in 1:n
           out[i] = i^2
       end

julia> out == [i^2 for i in 1:n]
true

julia> @BATCH scheduler=:greedy minbatch=1000 for i in 1:n  # one-off override
           out[i] = i^2
       end
```
"""
macro BATCH(args...)
    isempty(args) && throw(ArgumentError("@BATCH requires a `for` loop"))
    ex = args[end]
    minbatch_expr, scheduler = _batch_parse_kwargs(args[1:end-1])
    loopvar, range, body = _batch_extract_for(ex)

    dispatch = if scheduler === :greedy
        :(_batch_run_greedy(__batch_f, __batch_parts))
    elseif scheduler === :static
        :(_batch_run_static(__batch_f, __batch_parts))
    elseif scheduler === :default
        :(_batch_run_default(__batch_f, __batch_parts))
    else
        :(_batch_dispatch(__batch_f, __batch_parts, SCHEDULER[]))
    end

    quote
        __batch_range = $(esc(range))
        __batch_n = length(__batch_range)
        __batch_minbatch = max(1, Int($(esc(minbatch_expr))))
        if Threads.nthreads() == 1 || __batch_n <= __batch_minbatch
            for $(esc(loopvar)) in __batch_range
                $(esc(body))
            end
        else
            __batch_parts = collect(Iterators.partition(__batch_range, __batch_minbatch))
            __batch_f = function (__batch_chunk)
                for $(esc(loopvar)) in __batch_chunk
                    $(esc(body))
                end
            end
            $dispatch
        end
        nothing
    end
end

end # @static if
