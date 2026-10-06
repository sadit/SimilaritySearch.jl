# The tradeoff/width grid for MinRecall's hinge on ccnews: one graph, tuned with external
# queries under each (tradeoff, width, numqueries) and several seeds, scored on held-out queries.
# width=1e-4 is the hard threshold of old. (Run originally against the softplus hinge; the
# current hinge is the finite-support one, so the numbers will move: see README.md.)
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> tradeoff-grid.jl
using SimilaritySearch, HDF5, Statistics, Random, Dates, Printf
const K = 10
path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
X = h5read(path, "train")::Matrix{Float32}
Qi = h5read(path, "itest/queries"); Gi = h5read(path, "itest/knns")[1:K, :]
Qo = h5read(path, "otest/queries"); Go = h5read(path, "otest/knns")[1:K, :]
dim, n = size(X)
@info "$(now()) ccnews n=$n dim=$dim threads=$(Threads.nthreads()); itest $(size(Qi, 2)) otest $(size(Qo, 2))"

G = SearchGraph(Dist.SqL2(), MatrixDatabase(X))
ctx = SearchGraphContext(; reporters=[])
build_t = @elapsed index!(G, ctx)
@info "$(now()) built in $(round(build_t; digits=1)) s; construction left $(G.algo[])"

tunepool = MatrixDatabase(Qi[:, 1:500])                  # tuning queries are drawn from here
eval_i = MatrixDatabase(Qi[:, 501:end]);  gold_i = Gi[:, 501:end]   # held out, in distribution
eval_o = MatrixDatabase(Qo[:, 1:500]);    gold_o = Go[:, 1:500]     # out of distribution

function score!(G, ctx, Q, gold)
    before = copy(ctx.costdists)
    ids, _ = searchbatch(G, ctx, Q, K)
    visits = distance_evaluations(ctx, before) / length(Q)
    b = bootstrapscore(recallscore, gold, ids; rng=Xoshiro(0))
    (recall=b.mean, std=b.std, visits=visits)
end

const TARGET = 0.9
rows = NamedTuple[]
csv = open("tradeoff-grid.csv", "w")
println(csv, "numqueries,width,tradeoff,seed,bsize,delta,maxvisits,recall_i,std_i,visits_i,recall_o,std_o,visits_o,tune_s")
println("| numqueries | width | tradeoff | seed | tuned | recall_i ± std | visits_i | recall_o | visits_o | tune s |")
println("|---|---|---|---|---|---|---|---|---|---|")
for numqueries in (64, 256), width in (nothing, 0.01f0, 0.04f0, 1f-4), tradeoff in (1.1, 1.5, 3.0, 10.0)
    for seed in 1:8
        rng = Xoshiro(seed)
        tq = SubDatabase(tunepool, randperm(rng, length(tunepool))[1:numqueries])
        G.algo[] = BeamSearch(; bsize=4, Δ=1.0)                       # the same start for every run
        tune_t = @elapsed optimize_index!(G, ctx, MinRecall(TARGET; tradeoff, width); queries=tq, rng)
        a = G.algo[]
        si = score!(G, ctx, eval_i, gold_i); so = score!(G, ctx, eval_o, gold_o)
        w = width === nothing ? "auto" : string(width)
        push!(rows, (; numqueries, width=w, tradeoff, seed, bsize=a.bsize, Δ=a.Δ, maxvisits=a.maxvisits, recall=si.recall, std=si.std, visits=si.visits, recall_o=so.recall, std_o=so.std, visits_o=so.visits))
        println(csv, join((numqueries, w, tradeoff, seed, a.bsize, a.Δ, a.maxvisits, si.recall, si.std, si.visits, so.recall, so.std, so.visits, tune_t), ","))
        flush(csv)
        @printf("| %d | %s | %.1f | %d | bsize=%d Δ=%.2f mv=%d | %.3f ± %.3f | %.0f | %.3f | %.0f | %.1f |\n",
                numqueries, w, tradeoff, seed, a.bsize, a.Δ, a.maxvisits, si.recall, si.std, si.visits, so.recall, so.visits, tune_t)
        flush(stdout)
    end
end
close(csv)

println()
println("SUMMARY (mean ± std across 8 seeds; target recall $TARGET on held-out in-distribution queries)")
println("| numqueries | width | tradeoff | recall_i | visits_i | recall_o | visits_o | bsize range |")
println("|---|---|---|---|---|---|---|---|")
for numqueries in (64, 256), w in ("auto", "0.01", "0.04", "0.0001"), tradeoff in (1.1, 1.5, 3.0, 10.0)
    cell = filter(r -> r.numqueries == numqueries && r.width == w && r.tradeoff == tradeoff, rows)
    isempty(cell) && continue
    m(f) = mean(getfield.(cell, f)); s(f) = std(getfield.(cell, f))
    @printf("| %d | %s | %.1f | %.3f ± %.3f | %.0f ± %.0f | %.3f ± %.3f | %.0f ± %.0f | %d-%d |\n",
            numqueries, w, tradeoff, m(:recall), s(:recall), m(:visits), s(:visits),
            mean(r.recall_o for r in cell), std(r.recall_o for r in cell), mean(r.visits_o for r in cell), std(r.visits_o for r in cell),
            minimum(r.bsize for r in cell), maximum(r.bsize for r in cell))
end
