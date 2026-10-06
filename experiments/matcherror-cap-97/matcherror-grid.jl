# MaxMatchError on ccnews: the three canonical transition zones, two targets, 64/256 tuning
# queries, 8 seeds; scored outside on held-out queries by the external macro match error (gold
# distances computed exhaustively under the graph's SqL2), recall@10 and visits. The per-position
# cap (maxdeviation) is the package's behaviour now; the original uncapped run, which showed why
# the cap was needed, is results/matcherror-grid.log and README.md.
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> matcherror-grid.jl
using SimilaritySearch, HDF5, Statistics, Random, Dates, Printf
const K = 10
path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
X = h5read(path, "train")::Matrix{Float32}
Qi = h5read(path, "itest/queries"); Gi = h5read(path, "itest/knns")[1:K, :]
db = MatrixDatabase(X)
G = SearchGraph(Dist.SqL2(), db)
ctx = SearchGraphContext(; reporters=[])
build_t = @elapsed index!(G, ctx)
@info "$(now()) built in $(round(build_t; digits=1)) s; construction left $(G.algo[])"
tunepool = MatrixDatabase(Qi[:, 1:500])
eval_i = MatrixDatabase(Qi[:, 501:end]); gold_i = Gi[:, 501:end]
# gold distances for the held-out queries, under the graph's own distance
E = ExhaustiveSearch(Dist.SqL2(), db)
_, goldD = searchbatch(E, GenericContext(), eval_i, K)
golddists = [goldD[:, i] for i in 1:length(eval_i)]
@info "$(now()) gold distances for $(length(eval_i)) held-out queries: k-th neighbor median $(median(goldD[K, :]))"

function score!(G, ctx, kind)
    before = copy(ctx.costdists)
    knns = [search(G, ctx, eval_i[i], knnqueue(KnnSorted, K)) for i in 1:length(eval_i)]
    visits = distance_evaluations(ctx, before) / length(eval_i)
    ids = reduce(hcat, [collect(IdView(r)) for r in knns])
    (match=macromatcherror(golddists, knns, kind), recall=macrorecall(gold_i, ids), visits=visits)
end

rows = NamedTuple[]
println("| maxerror | transition | numqueries | seed | tuned | match_i | recall_i | visits_i |"); println("|---|---|---|---|---|---|---|---|")
for maxerror in (0.02f0, 0.05f0), transition in ((-1, 1), (0, 2), (-2, 0)), numqueries in (64, 256)
    for seed in 1:8
        rng = Xoshiro(seed)
        tq = SubDatabase(tunepool, randperm(rng, length(tunepool))[1:numqueries])
        G.algo[] = BeamSearch(; bsize=4, Δ=1.0)
        kind = MaxMatchError(; maxerror, transition)
        optimize_index!(G, ctx, kind; queries=tq, rng)
        a = G.algo[]
        sc = score!(G, ctx, kind)
        push!(rows, (; maxerror, transition, numqueries, seed, bsize=a.bsize, sc...))
        @printf("| %.2f | %s | %d | %d | bsize=%d Δ=%.2f | %.4f | %.3f | %.0f |\n", maxerror, transition, numqueries, seed, a.bsize, a.Δ, sc.match, sc.recall, sc.visits); flush(stdout)
    end
end
println(); println("SUMMARY (mean ± std across 8 seeds; external macro match error against its target, recall@10, visits)")
println("| maxerror | transition | numqueries | match_i | recall_i | visits_i | bsize range |"); println("|---|---|---|---|---|---|---|")
for maxerror in (0.02f0, 0.05f0), transition in ((-1, 1), (0, 2), (-2, 0)), numqueries in (64, 256)
    cell = filter(r -> r.maxerror == maxerror && r.transition == transition && r.numqueries == numqueries, rows)
    f(k) = (mean(getfield.(cell, k)), std(getfield.(cell, k)))
    @printf("| %.2f | %s | %d | %.4f ± %.4f | %.3f ± %.3f | %.0f ± %.0f | %d-%d |\n", maxerror, transition, numqueries,
            f(:match)..., f(:recall)..., f(:visits)..., minimum(r.bsize for r in cell), maximum(r.bsize for r in cell))
end
