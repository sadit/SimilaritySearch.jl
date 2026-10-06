# Why the external macro match error was unstable on ccnews: the per-query distribution, with
# and without the per-position cap (maxdeviation) that 1.6 added.
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> matcherror-tail.jl
using SimilaritySearch, HDF5, Statistics, Random, Printf
const K = 10
path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
X = h5read(path, "train")::Matrix{Float32}; Qi = h5read(path, "itest/queries")
db = MatrixDatabase(X); G = SearchGraph(Dist.SqL2(), db); ctx = SearchGraphContext(; reporters=[]); index!(G, ctx)
eval_i = MatrixDatabase(Qi[:, 501:end])
_, goldD = searchbatch(ExhaustiveSearch(Dist.SqL2(), db), GenericContext(), eval_i, K)
golddists = [goldD[:, i] for i in 1:length(eval_i)]
tq = SubDatabase(MatrixDatabase(Qi[:, 1:500]), randperm(Xoshiro(1), 500)[1:256])
kind = MaxMatchError(; maxerror=0.05f0)
G.algo[] = BeamSearch(; bsize=4, Δ=1.0); optimize_index!(G, ctx, kind; queries=tq, rng=Xoshiro(1))
knns = [search(G, ctx, eval_i[i], knnqueue(KnnSorted, K)) for i in 1:length(eval_i)]
pq = perqueryscores((g, r) -> matcherror(g, r, kind), golddists, knns)
spread = [g[end] - g[1] for g in golddists]
println("tuned ", G.algo[], "; held-out queries ", length(pq))
@printf("per-query match error: mean %.4f  median %.4f  q90 %.4f  q99 %.4f  max %.2f\n", mean(pq), median(pq), quantile(pq, 0.9), quantile(pq, 0.99), maximum(pq))
top = sort(pq; rev=true); n1 = cld(length(pq), 100)
@printf("share of the mean from the top 1%% of queries: %.0f%%;  from the top 10 queries: %.0f%%\n", 100sum(top[1:n1]) / sum(pq), 100sum(top[1:10]) / sum(pq))
@printf("gold spread d*_k - d*_1: median %.3f  q10 %.3f  q01 %.4f  min %.5f  (minspread 0.01)\n", median(spread), quantile(spread, 0.1), quantile(spread, 0.01), minimum(spread))
worst = sortperm(pq; rev=true)[1:5]
for i in worst
    @printf("  worst query: error %.2f, gold spread %.5f, gold d*_1 %.4f d*_k %.4f, returned d_1 %.4f d_k %.4f\n", pq[i], spread[i], golddists[i][1], golddists[i][end], first(DistView(knns[i])), last(DistView(knns[i])))
end
# the uncapped score, as it was before 1.6: maxdeviation=Inf32 removes the per-position cap
pc = perqueryscores((g, r) -> matcherror(g, r; maxdeviation=Inf32), golddists, knns)
@printf("uncapped (maxdeviation=Inf): mean %.4f  median %.4f  q99 %.4f  max %.2f\n", mean(pc), median(pc), quantile(pc, 0.99), maximum(pc))
pm = perqueryscores((g, r) -> matcherror(g, r; maxdeviation=Inf32, spreadfloor=0.05f0), golddists, knns)
@printf("uncapped, spreadfloor 0.05 instead of 0.01: mean %.4f  median %.4f  q99 %.4f  max %.2f\n", mean(pm), median(pm), quantile(pm, 0.99), maximum(pm))
