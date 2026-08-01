# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

SimilaritySearch.jl is a Julia library for nearest-neighbor search. Its flagship index
is `SearchGraph`, an approximate, incrementally-built graph index; it also ships exact
baselines (`ExhaustiveSearch`, `ParallelExhaustiveSearch`), scalar quantization
(`ScalarQuant`), random/Hadamard projections and bit sketches (`Projections`), and
supporting utilities (k-NN result queues, batch parallelism, distance functions).

See `README.md`/`docs/src/index.md` for the research background and citations.

## Build / test / run

```sh
# from the repo root
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first-time setup
julia -t8 --project=. -e 'using Pkg; Pkg.test()'       # run the full test suite
```

**Always pass `-tN` (e.g. `-t8`) when testing anything threading-related.** The default
session is single-threaded (`Threads.nthreads() == 1`), which silently takes every fast
serial path in `@BATCHES` and never exercises real parallelism, races, or
scheduler-specific behavior.

Individual test files live in `test/*.jl` and are `include`d from `test/runtests.jl`; to
run just one, `include` it directly after `using SimilaritySearch` in a REPL/script rather
than editing `runtests.jl`. `Aqua.jl` ambiguity/quality checks only run under
`VERSION == v"1.10"` (see the top of `runtests.jl`).

### Julia version matrix

CI (`.github/workflows/ci.yml`) only officially tests **Julia 1.10**. The package also
supports 1.11 and 1.12 (verified by hand repeatedly during development, not by CI) via
`@static if VERSION >= v"1.11"` gates, mainly in `src/parallel.jl` (native
`Threads.@threads :greedy` doesn't exist before 1.11). If `juliaup` has other versions
installed, cross-check with:

```sh
julia +1.11 -t8 --project=. -e 'using Pkg; Pkg.test()'
julia +1.12 -t8 --project=. -e 'using SimilaritySearch'   # at least a load smoke-test
```

## Architecture map (`src/`)

- `SimilaritySearch.jl` — top-level module; defines `AbstractSearchIndex`,
  `AbstractContext`, the `search`/`searchbatch`/`push_item!`/`append_items!`/`index!`
  generic-function interface, and `getminbatch` (see Parallelism below).
- `parallel.jl` — the `@BATCHES` macro (see below). **Read this file's docstrings before
  touching any parallel loop** — it documents real hygiene pitfalls, not just API.
  `include`d early in `SimilaritySearch.jl`, right after the module opens.
- `dist/` (`Dist` submodule) — distance functions (`L2`, `SqL2`, `Cosine`, `Angle`,
  sequences, sets, "hacks" like `DistanceWithIdentifiers`).
- `db/` — database containers (`MatrixDatabase`, `VectorDatabase`, `SubDatabase`) wrapping
  the actual point storage.
- `pqueue/` — k-NN result containers (`KnnSorted`, `KnnHeap`), both `AbstractKnn`
  subtypes sharing one interface (`push_item!`, `nearest`, `frontier`, `viewitems`,
  `reuse!`, `maxlength`). Construct via `knnqueue(KnnSorted, k_or_vec)`, never the raw
  struct constructor.
- `exact/` — `ExhaustiveSearch` (sequential) and `ParallelExhaustiveSearch` (parallel,
  `@BATCHES`-based, lock-free per-batch buffers).
- `searchgraph/` — `SearchGraph` itself: construction/insertion (`insertions.jl`),
  rebuild-from-scratch (`rebuild.jl`), beam search (`beamsearch.jl`), neighborhood
  filters (`neighborhood.jl`), adjacency backends (`../adj/`), per-call state
  (`context.jl` → `SearchGraphContext`).
- `sq/` (`ScalarQuant` submodule) — per-column (`SQu2`/`SQu4`/`SQu8`) and global
  (`SQgu4`/`SQgu8`) scalar quantization, each its own nested submodule.
- `proj/` (`Projections` submodule) — `RandomProjections` (gaussian/QR),
  `HadamardProjection`, and `bitsketch` (SimHash-style binary sketches).
- `allknn.jl`, `closestpair.jl`, `neardup.jl`, `hsp.jl`, `rerank.jl`, `fft.jl`, `opt.jl` —
  higher-level algorithms built on top of the index interface.

## Conventions worth knowing before writing code

- **Most index/context constructors are positional, not keyword**, despite some
  docstrings suggesting otherwise: `SearchGraph(dist, db)`,
  `ParallelExhaustiveSearch(dist, db)`, `ExhaustiveSearch(dist, db)` — **not**
  `SearchGraph(; dist, db)`. Check the actual method definition before assuming a
  keyword form exists; a couple of docstrings document keyword forms that were never
  actually implemented.
- Build a context with `GenericContext()` / `SearchGraphContext()` directly — there is no
  working `getcontext(index)` despite it being referenced in some docstrings/deprecated.jl
  shims; don't rely on it existing.
- Distance functions live under `Dist` (e.g. `SimilaritySearch.Dist.SqL2()`), not at
  top level.
- `IdDist(id, dist)` is the fundamental `(identifier, distance)` pair type; `IdView`/
  `DistView` give zero-copy column-style views over collections of it.

## Parallelism: `@BATCHES` and `getminbatch`

`@BATCHES` (in `src/parallel.jl`) is this package's only parallel-for construct — `Polyester`/
`@batch` was fully removed from `src/` (don't reintroduce it); `Project.toml` still lists
`Polyester` as a dependency as of this writing, but nothing in `src/` uses it anymore.
`@BATCHES` is a single native `Threads.@threads`-based macro on every supported Julia version.

Simple form (equivalent to today's plain per-element loop):

```julia
@BATCHES minbatch for i in range
    ...
end
```

Full form, all sections but `@LOOP` optional:

```julia
@BATCHES minbatch begin
    @BEGIN
        results = Vector{Float32}(undef, @nbatches)   # runs once, before dispatch
    @BEGINBATCH
        acc = 0.0f0                                    # runs once per batch
    @LOOP for i in range
        acc += f(i)                                    # runs once per element
    end
    @ENDBATCH
        results[@batchid] = acc                         # runs once per batch, after its elements
    @END
        total = sum(results)                            # runs once, after all batches join
end
```

Key facts an agent must know before editing anything here:

- **Index scratch buffers by `@batchid`, never by `Threads.threadid()`.** Batch ids are
  fixed, disjoint ordinals — race-free under *every* scheduler (`:static`/`:default`/
  `:greedy`). `Threads.threadid()`-indexing is only safe under `:static` (the default) and
  is a silent data race under the others. No remaining call site in `src/` still does
  this: `dist/seqs.jl`'s `Levenshtein`/`LCS` were the one case that couldn't use
  `@batchid` at all (their scratch buffer is needed inside `evaluate(dist, a, b)`, the
  generic, context-free interface shared by *every* distance function in this package —
  no `ctx`/`@batchid` reaches it), so instead of thread-indexing they use a `Channel`-based
  buffer pool (`take!`/`put!`, sized from `ctx.maxbatches` when a context is given via
  `Levenshtein(ctx; ...)`/`LCS(ctx)`) — safe under *any* concurrency model, not just
  `@BATCHES`, since it has no dependency on thread identity at all. A smaller pool only
  costs throughput (a `take!` blocks until a buffer is returned), never correctness — this
  is the preferred pattern over thread/batch-indexing whenever the caller can't supply a
  `@batchid` at all (e.g. a context-free interface like `evaluate`).
- **`GenericContext`/`SearchGraphContext` carry `batchid`/`maxbatches` fields** (see
  `searchgraph/context.jl`) precisely so `@batchid`-indexing can flow through the existing
  `search`/`find_neighborhood!` call graph without changing any of those functions'
  signatures: mint a per-batch context once per batch (in `@BEGINBATCH`, not per element)
  via `bctx = @set ctx.batchid = @batchid` (`Accessors.@set`; already `using Accessors`),
  then use `bctx` — never the outer `ctx` — for every call made from inside that batch.
  `getvstate`/`getbeam` (`context.jl`) read `ctx.batchid` to pick their scratch slot.
  Both context structs have a *phantom* type parameter (`KnnType`, not derivable from any
  field), so `@set` requires a `ConstructionBase.constructorof` override for each — already
  defined right after each struct; don't remove it.
- **The "tagged-handle" hazard (found live in this codebase — read this before touching
  any `@BATCHES` body that mints a `bctx`/similar per-batch handle).** Unlike
  `Threads.threadid()`-aliasing, this bug is unsafe under **every** scheduler, including
  `:static` — it has nothing to do with task migration. It happens when `@BEGINBATCH`
  correctly mints a tagged per-batch copy, but a call inside `@LOOP`/`@ENDBATCH` is
  accidentally passed the original, untagged object instead. Every batch then silently
  resolves to the *same* hardcoded slot (whatever the untagged object's default `batchid`
  is, typically `1`) — a live data race between concurrently-running batches on genuinely
  different threads. It type-checks, compiles, and runs without error, producing
  plausible-looking but silently corrupted results, so it's easy to miss in a quick test.
  Exactly this bug was caught and fixed in `searchgraph/rebuild.jl` and
  `searchgraph/insertions.jl`:

  ```julia
  # BUGGY: tmp/N are correctly @batchid-sliced, but find_neighborhood! still gets the
  # outer, untagged `ctx` — its internal getvstate/getbeam calls all resolve to slot 1,
  # for every batch, concurrently.
  @BEGINBATCH
      bctx = @set ctx.batchid = @batchid
      tmp = knnqueue(bctx, view(qcache, 1:ksearch, 2 * (@batchid) - 1))
      N = knnqueue(bctx, view(qcache, 1:ksearch, 2 * (@batchid)))
  @LOOP for objID in 1:n
      find_neighborhood!(N, g, ctx, database(g, objID), tmp, 1:-1; hints=...)  # ctx, not bctx
  end

  # FIXED — use the tagged bctx, not the outer ctx
  @LOOP for objID in 1:n
      find_neighborhood!(N, g, bctx, database(g, objID), tmp, 1:-1; hints=...)
  end
  ```

  **Rule of thumb: once `@BEGINBATCH` mints a `bctx`, grep the rest of that `@BATCHES`
  body for the original variable's name (`ctx`) — it should not appear inside
  `@LOOP`/`@ENDBATCH` at all.** This is also why a freshly-built/empty index can mask the
  bug in a quick test: `find_neighborhood!` only touches `ctx` at all when the target
  index already has elements (`length(index) > 0`).
- **`@batchid`/`@nbatches` followed directly by a unary `-` misparses.** `2 * @batchid - 1`
  parses as `2 * @batchid(-1)` (the bare macro slurps the following `-1` as an argument)
  and errors. Always parenthesize: `2 * (@batchid) - 1`.
- **Never splice `Threads.@threads` directly inside a macro's own generated
  quote/`esc()` tree.** A hygiene interaction between the two macros can silently make the
  per-iteration binding resolve to a single shared variable instead of a fresh per-task
  local — a real, intermittent, silent data race was caught this way (see the comment
  above `_batches_run_static` in `parallel.jl`). Keep `Threads.@threads` confined to
  plain, hand-written, non-macro functions.
- `getminbatch(n, nt=Threads.nthreads(); blocks_per_thread=8, maxbatches=n)` is the
  underlying, always-valid way to compute `minbatch` for `@BATCHES`. `maxbatches` (a plain
  `Int`, deliberately with **no special/sentinel value** — no `0`-means-off, no
  `Union{Nothing,Int}` — to stay type-stable *and* simple to reason about) is just a hard
  ceiling, always in effect; it defaults to `n` because `n` is already the largest a
  batch count could sensibly be, so that default is a genuine no-op, not a disguised
  "disabled" flag. Pass anything smaller and it directly reduces the batch count (raising
  `minbatch` correspondingly) — there is no other case to remember, and every `Int`
  (including `0` or negative) produces a well-defined result. **Prefer the context-aware
  overload,
  `getminbatch(ctx::AbstractContext, n)`** (`searchgraph/context.jl`), whenever a context
  object is available — it derives `maxbatches` from `ctx.maxbatches` (default
  `8 * Threads.nthreads()` for both context types), so the cap stays consistent with the
  capacity of that context's own caches (`vstates`/`beams`, for `SearchGraphContext`)
  instead of being an independent, easy-to-drift number. `ParallelExhaustiveSearch`'s
  `search` and `rebuild` used to each carry their own bespoke `maxbatches` keyword; both
  were dropped in favor of this single, context-level knob — set it via
  `GenericContext(; maxbatches=...)`/`SearchGraphContext(; maxbatches=...)` instead. Read
  `getminbatch`'s docstring's "Extreme cases" warning before picking a
  `maxbatches`/`blocks_per_thread` value — capping too aggressively can leave threads idle.
- When a scratch buffer's width is tied to `maxbatches`, derive the cap from the buffer's
  *actual* allocated size (`size(buf, 2) ÷ slots_per_batch`), not from an assumed
  relationship with some other parameter (e.g. a caller-supplied block size) — a mismatch
  between an assumed relationship and the real buffer size caused a real out-of-bounds
  crash during development (`searchgraph/insertions.jl`'s `qcache`, which is why its width
  is sized directly from `ctx.maxbatches` in `index!`).
- Some search methods (e.g. `ParallelExhaustiveSearch`'s `search`) are commonly invoked
  from *within* another `@BATCHES`-parallelized outer loop (`searchbatch!`/`allknn`/
  `closestpair` all do this generically). Native `:static` throws if nested/concurrent;
  such inner call sites force `scheduler=:default` explicitly rather than relying on the
  global default.

## Git / commit conventions

Recent history favors concise, single-focus commits explaining *why* a change was made,
not a line-by-line what — see `git log --oneline` for the house style. Don't commit or
push unless explicitly asked to.
