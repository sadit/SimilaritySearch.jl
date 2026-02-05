using SimilaritySearch
SimilaritySearch.getminbatch(::Int, ::Int, Int) = 8

function main(n, m, dim, k)
    db = StrideMatrixDatabase(rand(Float32, dim, n))
    queries = StrideMatrixDatabase(rand(Float32, dim, m))
    dist = SqL2Distance()
    seq = ExhaustiveSearch(; db, dist)
    knns = zeros(IdWeight, k, m)
    GC.enable(false)
    @time searchbatch!(seq, getcontext(seq), queries, knns; sorted=false)
    GC.enable(true)
    @show n m dim k

end

@info "warming"
main(100, 10, 8, 2)
@info "large benchmark"
knns = main(1000_000, 1000, 8, 10)
size(knns)
