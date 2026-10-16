# Issue #125: below which index size should SearchGraph's `search` scan everything instead of
# running the beam? T = 1 disables the shortcut; the shipped value was 64.
# Random subsets of ccnews train (n items), 1000 itest queries, gold by ExhaustiveSearch on the
# subset (NormCosine), k = min(10, n). Two builds: sequential (parallel_block=1, where the
# shortcut decides the first T-1 neighborhoods) and parallel (default parallel_block = 8 nthreads,
# whose first block is already exhaustive among itself). Two searches: the autotuned beam a user
# gets, and a fixed BeamSearch(bsize=8, Δ=1) that compares topologies alone.
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> threshold.jl
# It sets `SimilaritySearch.SMALLINDEX[]`, a `Ref` while the measurement ran; the package ships it
# as a constant, so restore `const SMALLINDEX = Ref(64)` in `src/searchgraph/beamsearch.jl` to rerun.
using SimilaritySearch, HDF5, Statistics, Random, Printf
const SS = SimilaritySearch
const path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
const X = h5read(path, "train")::Matrix{Float32}
const Q = MatrixDatabase(h5read(path, "itest/queries")[:, 1:1000])
const dist = Dist.NormCosine()

evals(ctx) = sum(ctx.costdists)

function run(n, rep, T, mode)
    db = MatrixDatabase(X[:, randperm(Xoshiro(rep), size(X, 2))[1:n]])
    k = min(10, n)
    gold, _ = searchbatch(ExhaustiveSearch(dist, db), GenericContext(), Q, k)
    SS.SMALLINDEX[] = T
    ctx = mode === :seq ? SearchGraphContext(parallel_block=1, reporters=[]) : SearchGraphContext(reporters=[])
    G = SearchGraph(dist, db)
    Random.seed!(rep)
    tbuild = @elapsed index!(G, ctx)
    edges = sum(i -> neighbors_length(G.adj, i), 1:n)
    row = [n, rep, T, mode, tbuild, edges]
    for algo in (G.algo[], BeamSearch(bsize=8, Δ=1.0))
        G.algo[] = algo
        fill!(ctx.costdists, 0)
        ids, _ = searchbatch(G, ctx, Q, k)
        push!(row, macrorecall(gold, ids), evals(ctx) / length(Q))
    end
    row
end

out = joinpath(@__DIR__, "results", "threshold.tsv")
open(out, "w") do f
    println(f, join(["n", "rep", "T", "mode", "build_s", "edges", "recall_tuned", "evals_tuned", "recall_fixed", "evals_fixed"], '\t'))
    for n in (16, 32, 48, 63, 128, 256, 1024, 10_000, 100_000), rep in 1:(n >= 100_000 ? 2 : 5), T in (1, 8, 16, 32, 64), mode in (:seq, :par)
        row = run(n, rep, T, mode)
        println(f, join(row, '\t')); flush(f)
        @printf("n=%d rep=%d T=%d %s build=%.2fs recall tuned %.3f (%.0f evals) fixed %.3f (%.0f evals)\n", n, rep, T, mode, row[5], row[7], row[8], row[9], row[10])
    end
end
