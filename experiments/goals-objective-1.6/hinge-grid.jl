# The transition zone of MinRecall's finite-support hinge on ccnews, same graph, external
# queries: centered on the target (-1, 1), below it (0, 2), above it (-2, 0), over three
# tradeoffs, 64/256 tuning queries, 8 seeds. (The original run of this script also measured a
# softplus hinge by monkeypatching the hinge; softplus was dropped because it overshot the
# target by two to three widths, and its numbers are in README.md.)
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> hinge-grid.jl
using SimilaritySearch, HDF5, Statistics, Random, Dates, Printf
const K = 10
const ZONES = (("centered", (-1, 1)), ("below", (0, 2)), ("above", (-2, 0)))

path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
X = h5read(path, "train")::Matrix{Float32}
Qi = h5read(path, "itest/queries"); Gi = h5read(path, "itest/knns")[1:K, :]
Qo = h5read(path, "otest/queries"); Go = h5read(path, "otest/knns")[1:K, :]
G = SearchGraph(Dist.SqL2(), MatrixDatabase(X))
ctx = SearchGraphContext(; reporters=[])
build_t = @elapsed index!(G, ctx)
@info "$(now()) built in $(round(build_t; digits=1)) s; construction left $(G.algo[])"
tunepool = MatrixDatabase(Qi[:, 1:500])
eval_i = MatrixDatabase(Qi[:, 501:end]);  gold_i = Gi[:, 501:end]
eval_o = MatrixDatabase(Qo[:, 1:500]);    gold_o = Go[:, 1:500]
function score!(G, ctx, Q, gold)
    before = copy(ctx.costdists)
    ids, _ = searchbatch(G, ctx, Q, K)
    (recall=bootstrapscore(recallscore, gold, ids; rng=Xoshiro(0)).mean, visits=distance_evaluations(ctx, before) / length(Q))
end
const TARGET = 0.9
rows = NamedTuple[]
csv = open("hinge-grid.csv", "w"); println(csv, "hinge,numqueries,tradeoff,seed,bsize,delta,maxvisits,recall_i,visits_i,recall_o,visits_o")
println("| hinge | numqueries | tradeoff | seed | tuned | recall_i | visits_i | recall_o | visits_o |"); println("|---|---|---|---|---|---|---|---|---|")
for (hinge, transition) in ZONES, numqueries in (64, 256), tradeoff in (1.2, 1.5, 3.0)
    for seed in 1:8
        rng = Xoshiro(seed)
        tq = SubDatabase(tunepool, randperm(rng, length(tunepool))[1:numqueries])
        G.algo[] = BeamSearch(; bsize=4, Δ=1.0)
        optimize_index!(G, ctx, MinRecall(TARGET; tradeoff, transition); queries=tq, rng)
        a = G.algo[]
        si = score!(G, ctx, eval_i, gold_i); so = score!(G, ctx, eval_o, gold_o)
        push!(rows, (; hinge, numqueries, tradeoff, seed, bsize=a.bsize, recall=si.recall, visits=si.visits, recall_o=so.recall, visits_o=so.visits))
        println(csv, join((hinge, numqueries, tradeoff, seed, a.bsize, a.Δ, a.maxvisits, si.recall, si.visits, so.recall, so.visits), ",")); flush(csv)
        @printf("| %s | %d | %.1f | %d | bsize=%d Δ=%.2f mv=%d | %.3f | %.0f | %.3f | %.0f |\n", hinge, numqueries, tradeoff, seed, a.bsize, a.Δ, a.maxvisits, si.recall, si.visits, so.recall, so.visits); flush(stdout)
    end
end
close(csv)
println(); println("SUMMARY (mean ± std across 8 seeds; target $TARGET; auto width)")
println("| hinge | numqueries | tradeoff | recall_i | visits_i | recall_o | visits_o | bsize range |"); println("|---|---|---|---|---|---|---|---|")
for (hinge, _) in ZONES, numqueries in (64, 256), tradeoff in (1.2, 1.5, 3.0)
    cell = filter(r -> r.hinge == hinge && r.numqueries == numqueries && r.tradeoff == tradeoff, rows)
    f(k) = (mean(getfield.(cell, k)), std(getfield.(cell, k)))
    @printf("| %s | %d | %.1f | %.3f ± %.3f | %.0f ± %.0f | %.3f ± %.3f | %.0f ± %.0f | %d-%d |\n", hinge, numqueries, tradeoff,
            f(:recall)..., f(:visits)..., f(:recall_o)..., f(:visits_o)..., minimum(r.bsize for r in cell), maximum(r.bsize for r in cell))
end
