using SimilaritySearch
SimilaritySearch.getminbatch(::Int, ::Int, Int) = 8

function main(n, m, dim, k)
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))
    dist = Dist.SqL2()
    seq = ExhaustiveSearch(dist, db)
    ctx = GenericContext()
    knns_ids = zeros(UInt32, k, m)
    knns_dists = zeros(Float32, k, m)
    GC.enable(false)
    @time searchbatch!(seq, ctx, queries, knns_ids, knns_dists)
    GC.enable(true)
    @show n m dim k
    knns_ids, knns_dists
end

@info "warming"
main(100, 10, 8, 2)
@info "large benchmark"
knns_ids, knns_dists = main(1000_000, 1000, 8, 10)
size(knns_ids)
