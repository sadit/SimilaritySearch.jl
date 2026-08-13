using SimilaritySearch, Statistics, StatsBase, Random, JSON

#
# Compares three ways of building a SearchGraph on synthetic L2 data:
#
#   1. "incremental" -- the normal, online `index!(graph, ctx)`.
#   2. "knr"          -- fast, non-incremental `index!(graph, ctx, :knr; ...)`.
#   3. "knr+rebuild"  -- (2) followed by a `rebuild(graph, ctx)` refinement pass.
#
# For each, we report the indexing (build) time *and* the resulting search quality
# (recall + searchtime against an ExhaustiveSearch gold standard), after tuning every variant
# to the same target recall via `optimize_index!` so the comparison is apples-to-apples.
#
# The data has a low *intrinsic* dimension `dim` (e.g. 8 or 12), linearly embedded into a much
# higher `exdim`-dimensional space via a random projection -- same trick as `searchgraph.jl` --
# so plain L2 distance in `exdim` space still reflects the lower intrinsic dimensionality
# (dense vectors, not a sparse/low-rank corner case).
#

"""
    embed(rng, dim, exdim, n) -> MatrixDatabase

`n` dense `Float32` vectors of intrinsic dimension `dim`, embedded into `exdim` dimensions via a
random linear projection `P` (`exdim x dim`).
"""
function embed(rng, P, dim, n)
    MatrixDatabase(P * rand(rng, Float32, dim, n))
end

"""
    tune_and_search(graph, ctx, queries, gold_ids; ksearch, minrecall_search) -> (; opttime, searchtime, recall)

Tunes `graph`'s `BeamSearch` hyperparameters to `minrecall_search` via `optimize_index!` (which
samples its own validation queries internally from the indexed dataset -- `queries`/`gold_ids`
are deliberately *not* handed to it, so tuning never sees the held-out query set it's about to
be evaluated against), then measures `searchbatch` time and recall against `gold_ids`. Applied
identically to every variant so the comparison isn't skewed by one of them being tuned and
another not.
"""
function tune_and_search(graph, ctx, queries, gold_ids; ksearch, minrecall_search)
    opttime = @elapsed optimize_index!(graph, ctx, MinRecall(minrecall_search))
    searchtime = @elapsed knns_ids, knns_dists = searchbatch(graph, ctx, queries, ksearch)
    recall = macrorecall(gold_ids, knns_ids)
    (; opttime, searchtime, recall)
end

"""
    build_incremental(dist, db; minrecall, logbase) -> (graph, ctx, buildtime)

The normal, online construction path: `hyperparameters_callback` keeps `BeamSearch` tuned
towards `minrecall` *during* insertion, so `buildtime` here already includes continuous
online tuning -- unlike the `:knr` variants below, which do no tuning at all until the
explicit `optimize_index!` call in `tune_and_search`.
"""
function build_incremental(dist, db; minrecall, logbase)
    graph = SearchGraph(dist, db)
    ctx = SearchGraphContext(
        neighborhood=Neighborhood(; filter=SatNeighborhood(), logbase),
        hyperparameters_callback=OptimizeParameters(MinRecall(minrecall)),
        parallel_block=2^13
    )
    buildtime = @elapsed index!(graph, ctx)
    graph, ctx, buildtime, NamedTuple()
end

"""
    build_knr(dist, db; numrefs, k, n_neighbors, hints_size, start_factor) -> (graph, ctx, buildtime)

Fast, non-incremental construction via hierarchical reference clusters -- no online tuning
happens here at all (see `build_incremental`'s docstring).
"""
function build_knr(dist, db; numrefs, k, n_neighbors, hints_size, start_factor)
    graph = SearchGraph(dist, db)
    ctx = SearchGraphContext()
    buildtime = @elapsed index!(graph, ctx, :knr; numrefs, k, sample_method=:fft, n_neighbors, hints_size, start_factor)
    graph, ctx, buildtime, NamedTuple()
end

"""
    build_knr_rebuild(dist, db; numrefs, k, n_neighbors, hints_size, start_factor) -> (graph, ctx, buildtime, (; knrtime, rebuildtime))

`build_knr` followed by a `rebuild` pass (which re-derives every node's neighborhood against
the *whole* dataset instead of just the nodes seen so far, using the `:knr` graph's own
adjacency as search hints).
"""
function build_knr_rebuild(dist, db; numrefs, k, n_neighbors, hints_size, start_factor)
    graph, ctx, knrtime, _ = build_knr(dist, db; numrefs, k, n_neighbors, hints_size, start_factor)
    rebuildtime = @elapsed graph = rebuild(graph, ctx)
    graph, ctx, knrtime + rebuildtime, (; knrtime, rebuildtime)
end

function run_variant(D, name, buildfn, queries, gold_ids; ksearch, minrecall_search, n, m, dim, exdim)
    graph, ctx, buildtime, extra = buildfn()
    perf = tune_and_search(graph, ctx, queries, gold_ids; ksearch, minrecall_search)
    N = [neighbors_length(graph.adj, i) for i in eachindex(graph.adj)]
    NQ = quantile(N, 0:0.25:1)
    @info "== $name (n=$n, dim=$dim/$exdim) -> buildtime=$(round(buildtime; digits=3))s, opttime=$(round(perf.opttime; digits=3))s, searchtime=$(round(perf.searchtime; digits=4))s, recall=$(round(perf.recall; digits=4)), QpS=$(round(m/perf.searchtime; digits=1))"
    push!(D, (; name, n, m, dim, exdim, ksearch, minrecall_search, buildtime, perf.opttime, perf.searchtime, perf.recall, degree_quantiles=NQ, extra...))
end

function main_l2(D, n, m, dim, exdim; ksearch=16, minrecall=0.99, minrecall_search=0.9, logbase=1.3f0)
    @info "=== n=$n m=$m dim=$dim exdim=$exdim ksearch=$ksearch"
    @assert dim <= exdim
    rng = Xoshiro(n)
    dist = Dist.SqL2()
    P = randn(rng, Float32, exdim, dim)
    db = embed(rng, P, dim, n)
    queries = embed(rng, P, dim, m)

    seq = ExhaustiveSearch(dist, db)
    gold_ids, gold_dists = searchbatch(seq, GenericContext(), queries, ksearch)

    numrefs = max(64, ceil(Int, 4sqrt(n)))
    hints_size = ceil(Int, sqrt(n))
    knr_kwargs = (; numrefs, k=8, n_neighbors=4, hints_size, start_factor=0.97)

    run_variant(D, "incremental", () -> build_incremental(dist, db; minrecall, logbase), queries, gold_ids; ksearch, minrecall_search, n, m, dim, exdim)
    run_variant(D, "knr", () -> build_knr(dist, db; knr_kwargs...), queries, gold_ids; ksearch, minrecall_search, n, m, dim, exdim)
    run_variant(D, "knr+rebuild", () -> build_knr_rebuild(dist, db; knr_kwargs...), queries, gold_ids; ksearch, minrecall_search, n, m, dim, exdim)
end

D = []
for n in [50_000, 200_000], m in [1000], dim in [8, 12], exdim in [256]
    m = min(m, n)
    main_l2(D, n, m, dim, exdim)
end

for r in D
    println(JSON.json(r))
end
