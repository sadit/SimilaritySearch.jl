# SQEncoder on ccnews (SISAP 2025): every ScalarQuant quantizer (SQgu8/4/2 global, SQu8/4/2
# per-vector), with a QR rotation and with none, through an AsymmetricSearchGraph; the
# encoder's exhaustive ceiling next to the graph with its own tuning and at two fixed beams
# (b = bsize of BeamSearch, the columns to compare across rows), and bytes per vector.
# Results on issue #86: the rotation moved recall by < 0.01 at every width.
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> sqencoder-ccnews.jl

using SimilaritySearch, HDF5, Statistics, LinearAlgebra, Dates
using SimilaritySearch: encodequery
const SQ = SimilaritySearch.ScalarQuant
using SimilaritySearch.ScalarQuant: SQEncoder, sqcodes
const Projections = SimilaritySearch.Projections

const K = 10
const NQ = 1000
path = expanduser("~/SISAP2025/data-sisap2025/benchmark-dev-ccnews.h5")
X = h5read(path, "train")::Matrix{Float32}
Q = Matrix{Float32}(h5read(path, "itest/queries")[:, 1:NQ])
gold = [Set(h5read(path, "itest/knns")[1:K, j]) for j in 1:NQ]
dim, n = size(X)
queries = collect(eachcol(Q))
@info "$(now()) ccnews: n=$n dim=$dim threads=$(Threads.nthreads())"

knn_ids(index, ctx, qs) = [collect(IdView(search(index, ctx, q, knnqueue(KnnSorted, K)))) for q in qs]
recall_of(got) = mean(length(intersect(gold[j], Set(got[j]))) / K for j in eachindex(got))
function timed_recall(index, ctx, qs)
    knn_ids(index, ctx, qs[1:5])
    t = @elapsed got = knn_ids(index, ctx, qs)
    recall_of(got), t / length(qs) * 1e6
end
r(x) = round(x; digits=4)

println("| graph | encode s | build s | tuned algo | recall@10 / μs per query, own tuning | b=8 | b=32 | bytes/vector |")
println("|---|---|---|---|---|---|---|---|")
for quant in (SQ.SQgu8, SQ.SQgu4, SQ.SQgu2, SQ.SQu8, SQ.SQu4, SQ.SQu2), rotname in ("QR", "no rotation")
    rot = rotname == "QR" ? Projections.qr(dim, dim) : nothing
    est = SQEncoder(quant, rot, X)
    name = "SQEncoder $(nameof(quant)), $rotname"
    t_enc = @elapsed codes = sqcodes(est, X)
    bytes = cld(dim, 8 ÷ SQ.codewidth(est)) + 8 + (SQ.isglobal(est) ? 0 : 8)
    rseq, tseq = timed_recall(ExhaustiveSearch(est, codes), GenericContext(), [encodequery(est, q) for q in queries])
    println("| $name, exhaustive | $(round(t_enc; digits=1)) | - | - | $(r(rseq)) / $(round(Int, tseq)) | - | - | $bytes |")
    G = AsymmetricSearchGraph(est, sqcodes(est))
    ctx = SearchGraphContext(; reporters=[])
    t_build = @elapsed append_items!(G, ctx, MatrixDatabase(X))
    tuned = G.graph.algo[]
    rt, tt = timed_recall(G, ctx, queries)
    G.graph.algo[] = BeamSearch(; bsize=8, Δ=1.0);  r8, t8 = timed_recall(G, ctx, queries)
    G.graph.algo[] = BeamSearch(; bsize=32, Δ=1.0); r32, t32 = timed_recall(G, ctx, queries)
    println("| $name, graph | $(round(t_enc; digits=1)) | $(round(t_build; digits=1)) | bsize=$(tuned.bsize), Δ=$(round(tuned.Δ; digits=2)) | $(r(rt)) / $(round(Int, tt)) | $(r(r8)) / $(round(Int, t8)) | $(r(r32)) / $(round(Int, t32)) | $bytes |")
    flush(stdout)
    G = nothing; codes = nothing; GC.gc()
end
