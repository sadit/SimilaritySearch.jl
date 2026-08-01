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
  is a silent data race under the others; several older call sites still do this and are
  candidates for the same migration (search for `Threads.threadid()` in `src/`).
- **`@batchid`/`@nbatches` followed directly by a unary `-` misparses.** `2 * @batchid - 1`
  parses as `2 * @batchid(-1)` (the bare macro slurps the following `-1` as an argument)
  and errors. Always parenthesize: `2 * (@batchid) - 1`.
- **Never splice `Threads.@threads` directly inside a macro's own generated
  quote/`esc()` tree.** A hygiene interaction between the two macros can silently make the
  per-iteration binding resolve to a single shared variable instead of a fresh per-task
  local — a real, intermittent, silent data race was caught this way (see the comment
  above `_batches_run_static` in `parallel.jl`). Keep `Threads.@threads` confined to
  plain, hand-written, non-macro functions.
- `getminbatch(n, nt=Threads.nthreads(); blocks_per_thread=8, maxbatches=0)` is the only
  sanctioned way to compute `minbatch` for `@BATCHES`. `maxbatches` (a plain `Int`, `0`
  means uncapped — deliberately not `Union{Nothing,Int}`, to stay type-stable) directly
  bounds the resulting batch count, for call sites where each batch allocates its own
  `@nbatches`-sized scratch (e.g. `exact/parallel-exhaustive.jl`, `searchgraph/rebuild.jl`,
  `searchgraph/insertions.jl`). Read `getminbatch`'s docstring's "Extreme cases" warning
  before picking a `maxbatches`/`blocks_per_thread` value — capping too aggressively can
  leave threads idle.
- When a scratch buffer's width is tied to `maxbatches`, derive the cap from the buffer's
  *actual* allocated size (`size(buf, 2) ÷ slots_per_batch`), not from an assumed
  relationship with some other parameter (e.g. a caller-supplied block size) — a mismatch
  between an assumed relationship and the real buffer size caused a real out-of-bounds
  crash during development (`searchgraph/insertions.jl`'s `qcache`).
- Some search methods (e.g. `ParallelExhaustiveSearch`'s `search`) are commonly invoked
  from *within* another `@BATCHES`-parallelized outer loop (`searchbatch!`/`allknn`/
  `closestpair` all do this generically). Native `:static` throws if nested/concurrent;
  such inner call sites force `scheduler=:default` explicitly rather than relying on the
  global default.

## Git / commit conventions

Recent history favors concise, single-focus commits explaining *why* a change was made,
not a line-by-line what — see `git log --oneline` for the house style. Don't commit or
push unless explicitly asked to.
