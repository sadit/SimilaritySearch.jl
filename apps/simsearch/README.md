# simsearch

A command-line application for building, searching, evaluating, and inspecting
[SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl) indexes, driven entirely by
HDF5/JLD2-stored datasets. It lives under `apps/` as the first of what may become several
independent applications built on top of the library, each with its own environment that depends
on `SimilaritySearch` as a local, in-development package (`[sources] SimilaritySearch = {path =
"../.."}` in `Project.toml` — the same mechanism `Pkg.develop` uses, so tests and runs always
exercise the checked-out source, not a registered release).

## Requirements

**Julia ≥ 1.12.** This is a deliberate, stricter requirement than the parent `SimilaritySearch`
package's own `julia = "^1.10"` compat — a dependent application is free to require a newer
Julia than its dependency.

## Install

`simsearch` is a [Julia app](https://pkgdocs.julialang.org/dev/apps/) (Pkg's
app support is currently experimental). Install it once, from the repo root:

```sh
julia -e 'using Pkg; pkg"app develop apps/simsearch"'
```

This registers a `simsearch` executable under `~/.julia/bin/`, managed by
Pkg. Make sure `~/.julia/bin` is on your `PATH` (Pkg does not add it for you),
then run it directly:

```sh
simsearch <subcommand> [options]
```

`app develop` (as opposed to `app add`) points the installed app at this
checked-out source, mirroring the `[sources] SimilaritySearch = {path =
"../.."}` local-dev mechanism already used by this project — edits to
`apps/simsearch` or the parent `SimilaritySearch` package take effect without
reinstalling.

If you'd rather not install anything, run it as a plain script under an
activated project instead:

```sh
julia --project=apps/simsearch apps/simsearch/src/main.jl <subcommand> [options]
```

The examples below use the installed-executable form for brevity.

## The `path:key` dataset convention

Every dataset/query/gold/results argument that reads a matrix from disk uses a `path:key`
mini-format: `h5file.h5:key` or `jld2file.jld2:key`, meaning "read the array stored under `key`
in this HDF5 or JLD2 file". Columns are treated as the individual object/query vectors (i.e. a
`(dim, nobjects)` matrix), matching `SimilaritySearch.MatrixDatabase`'s convention.

- `--dataset` and `--queries` **always require** the `:key` suffix — there's no fixed schema for
  raw input data, so the key must be given explicitly.
- `--gold` and `--results` (on `evaluate`) accept the colon **optionally**: without one, the
  file is assumed to follow the fixed schema written by `simsearch search` (see below); with one,
  `key` is treated as an externally-produced ids matrix, and a sibling key `"<key>_dists"` is
  used for distances if present.

## Persistence: indexes never embed their dataset

`simsearch build` always saves an index with a placeholder dataset — the real data is *not*
duplicated into the `.jld2` index file. As a consequence, `simsearch search` and
`simsearch analyze` always **require** a `--dataset path:key` flag to reattach the real data
before they can do anything useful. This keeps saved index files small and keeps "what dataset is
this index built from" explicit at every later invocation, at the cost of always having to pass
`--dataset` — usually a non-issue since the same dataset is reused across many `search`/`analyze`
calls against one saved index.

## Usage

### `build` — build and save an index

```sh
simsearch build --type ExhaustiveSearch --dataset h5file.h5:key \
    --distance Dist.SqL2 --save index.jld2
```

```sh
simsearch build --type SearchGraph --dataset h5file.h5:key \
    --distance Dist.SqL2 --save index.jld2 --minrecall 0.95
```

| flag | required | default | description |
|---|---|---|---|
| `--type` | yes | — | `ExhaustiveSearch`, `ParallelExhaustiveSearch`, or `SearchGraph` |
| `--dataset` | yes | — | `path:key` dataset spec |
| `--distance` | yes | — | distance name, e.g. `Dist.SqL2` or `SqL2` (see below) |
| `--save` | yes | — | output path, must end in `.jld2` (the dataset is never embedded) |
| `--minrecall` | no | — | target recall (0–1) for `SearchGraph` autotuning via `optimize_index!`/`MinRecall`; ignored (with a warning) for exact index types |
| `--logbase` | no | `1.3` | `SearchGraph` neighborhood growth log-base (`Neighborhood.logbase`) — controls how many candidate neighbors are considered per insertion as the index grows; ignored for exact index types |
| `--logbase-callback` | no | `1.5` | `SearchGraph` periodic-callback log-base (`SearchGraphContext.logbase_callback`) — controls how often hyperparameter re-optimization/hints recomputation fires during incremental insertion; ignored for exact index types |
| `--hints-logbase` | no | `1.1` | `SearchGraph` entry-point hints log-base (`RandomHints.logbase`) — controls how many entry points are kept as the index grows; ignored for exact index types |

These three `--logbase*` flags map to three genuinely distinct knobs in the underlying package
(see `docs/src/tutorial/searchgraph.md` and `docs/src/tutorial/logging.md` in the parent
package) — each defaults to the package's own default and can be tuned independently.

### `search` — load an index and run queries

```sh
simsearch search index.jld2 --queries h5file.h5:key --dataset h5file.h5:key \
    --results out-res.jld2 -k 10
```

| flag | required | default | description |
|---|---|---|---|
| `index` (positional) | yes | — | saved index path |
| `--queries` | yes | — | `path:key` queries spec |
| `--dataset` | yes | — | `path:key` spec used to reattach the real dataset to the loaded index |
| `--results` | yes | — | output path, `.h5` or `.jld2` |
| `-k`/`--k` | no | `10` | number of neighbors per query |

### `evaluate` — score results against a gold standard

```sh
simsearch evaluate --gold h5file.h5:key --results res.h5:key -k 10 --html report.html
```

| flag | required | default | description |
|---|---|---|---|
| `--gold` | yes | — | gold results spec (`path` or `path:key`) |
| `--results` | yes | — | results spec, same grammar as `--gold` |
| `-k`/`--k` | no | `min(k_gold, k_results)` | number of neighbors to evaluate per query |
| `--html` | no | — | write a self-contained HTML report (stats table + SVG histograms) to this path |
| `--out` | no | — | also write the plain-text report to this path (always printed to stdout too) |

Reports: macro-recall (via `SimilaritySearch.macrorecall`), a per-query recall distribution
(mean/std/min/median/max + histogram), descriptive statistics and a histogram of the gold and
results distance distributions (when distances are available), and coverage counts (queries with
fewer than `k` neighbors found, per the package's `id == 0` sentinel convention).

### `analyze` — inspect a saved index

```sh
simsearch analyze index.jld2 --dataset h5file.h5:key --html report.html
```

| flag | required | default | description |
|---|---|---|---|
| `index` (positional) | yes | — | saved index path |
| `--dataset` | yes | — | `path:key` spec used to reattach the real dataset |
| `--html` | no | — | write a self-contained HTML report to this path |
| `--out` | no | — | also write the plain-text report to this path |

Reports index type, object count, dimension, and distance type for every index; for
`SearchGraph` specifically, also reports the node-degree distribution (mean/std/min/median/max +
histogram, computed from the adjacency list), the entry-point hints count, and the current
`BeamSearch` hyperparameters (`bsize`, `Δ`, `maxvisits`).

## The results-file schema

`search --results` and `evaluate --gold`/`--results` (when given without a `:key`) share one
fixed schema: key `"ids"` (an `(k, nqueries)` `Int32` matrix) and, when available, `"dists"` (an
`(k, nqueries)` `Float32` matrix), stored in either an `.h5` or `.jld2` file.

## Supported index types

- **`ExhaustiveSearch`** — exact, sequential. Best for small datasets or as a gold standard.
- **`ParallelExhaustiveSearch`** — exact, multi-threaded (needs Julia started with `-tN`). Same
  results as `ExhaustiveSearch`, faster on multi-core machines.
- **`SearchGraph`** — approximate, incrementally-built graph index. Scales to large datasets;
  tune with `--minrecall` and the `--logbase*` flags.

## Supported distances

The zero-argument numeric-vector distance family under `SimilaritySearch.Dist`: `L1`, `L2`,
`SqL2` (squared Euclidean — cheapest, same ranking as `L2`, recommended default), `LInfty`,
`Cosine`, `Angle`, `NormCosine`, `NormAngle`. The `Dist.` prefix is optional (`--distance SqL2` and
`--distance Dist.SqL2` are equivalent).

Not supported in this version: `Dist.Lp(p)` (needs an extra parameter) and the non-vector
distance families (`Dist.Sets.*`, `Dist.Bits.*`, `Dist.Seqs.*`), since the `path:key`
dataset convention is inherently a numeric matrix. Note also that `SearchGraph`'s greedy beam
search degrades on discrete/combinatorial distances with many exact ties — prefer
`ExhaustiveSearch`/`ParallelExhaustiveSearch` for those regardless of dataset size (see
`docs/src/tutorial/distances.md` in the parent package).

## Testing

```sh
julia --project=apps/simsearch apps/simsearch/test/runtests.jl
```

Runs an end-to-end smoke test (build → search → evaluate → analyze, plus dataset-IO round-trips)
against a small synthetic dataset, for all three index types and both `.h5`/`.jld2` results
files.
