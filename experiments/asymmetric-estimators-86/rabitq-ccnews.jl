# RaBitQ as the distance of an AsymmetricSearchGraph on ccnews (SISAP 2025): raw vectors in,
# raw queries in, sign bits plus three scalars stored; then the two-level RaBitQRefined with an
# exact or a scalar-quantized fallback beside the bits. Each estimator's own exhaustive recall
# next to the graph with its own tuning and at two fixed beams (b = bsize of BeamSearch, the
# columns to compare across rows), and bytes per vector. Results on issue #86.
#
# usage: julia -t auto --project=<env with HDF5 and this SimilaritySearch> rabitq-ccnews.jl

using SimilaritySearch, SimilaritySearch.RaBitQ, HDF5, Statistics, LinearAlgebra, Dates
using SimilaritySearch: encodequery
const Projections = SimilaritySearch.Projections
const SQ = SimilaritySearch.ScalarQuant

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

coarse = RaBitQCosine(Projections.qr(dim, dim))
τ9 = refinethreshold(coarse, X, K; q=0.9)
τ5 = refinethreshold(coarse, X, K; q=0.5)
@info "$(now()) τ from the sample's $K-th neighbor distances: q=0.9 -> $τ9, q=0.5 -> $τ5"
sq4 = RaBitQVectorFallback(SQ.SQgu4, coarse, X)
nbytes(c::RaBitQCode) = 8 * length(c.bits) + 12
nbytes(t::Tuple) = nbytes(t[1]) + (t[2] isa AbstractVector ? sizeof(t[2]) : length(t[2].V) + 8)
for (name, est) in (("RaBitQ bits, QR rotation", coarse),
                    ("bits + exact Float32, τ=Inf", RaBitQRefined(coarse, RaBitQExactFallback())),
                    ("bits + exact Float32, τ=q0.9", RaBitQRefined(coarse, RaBitQExactFallback(); τ=τ9)),
                    ("bits + exact Float32, τ=q0.5", RaBitQRefined(coarse, RaBitQExactFallback(); τ=τ5)),
                    ("bits + exact Float16, τ=q0.9", RaBitQRefined(coarse, RaBitQExactFallback{Float16}(); τ=τ9)),
                    ("bits + SQ 4 bits, τ=Inf", RaBitQRefined(coarse, sq4)),
                    ("bits + SQ 4 bits, τ=q0.9", RaBitQRefined(coarse, sq4; τ=τ9)))
    t_enc = @elapsed codes = rabitqcodes(est, X)
    bytes = nbytes(codes[1])
    # the estimator's own ceiling: exhaustive, queries prepared by hand
    seq = ExhaustiveSearch(est, codes)
    rseq, tseq = timed_recall(seq, GenericContext(), [encodequery(est, q) for q in queries])
    println("| $name, exhaustive | $(round(t_enc; digits=1)) | - | - | $(r(rseq)) / $(round(Int, tseq)) | - | - | $bytes |")
    # the graph
    G = AsymmetricSearchGraph(est, rabitqcodes(est))
    ctx = SearchGraphContext(; reporters=[])
    t_build = @elapsed append_items!(G, ctx, MatrixDatabase(X))
    tuned = G.graph.algo[]
    rt, tt = timed_recall(G, ctx, queries)
    G.graph.algo[] = BeamSearch(; bsize=8, Δ=1.0);  r8, t8 = timed_recall(G, ctx, queries)
    G.graph.algo[] = BeamSearch(; bsize=32, Δ=1.0); r32, t32 = timed_recall(G, ctx, queries)
    println("| $name, graph | $(round(t_enc; digits=1)) | $(round(t_build; digits=1)) | bsize=$(tuned.bsize), Δ=$(round(tuned.Δ; digits=2)) | $(r(rt)) / $(round(Int, tt)) | $(r(r8)) / $(round(Int, t8)) | $(r(r32)) / $(round(Int, t32)) | $bytes |")
    flush(stdout)
end
