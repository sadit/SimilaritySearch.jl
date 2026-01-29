using SimilaritySearch
SimilaritySearch.getminbatch(::Int, ::Int, Int) = 8

function main(n, m, dim, k)
    db = StrideMatrixDatabase(rand(Float32, dim, n))
    queries = StrideMatrixDatabase(rand(Float32, dim, m))
    seq = ExhaustiveSearch(; db)
    @time searchbatch(seq, getcontext(seq), queries, k; sorted=false)
    @show n m dim k

end

@info "warming"
main(100, 10, 8, 2)
@info "large benchmark"
knns = main(1000_000, 1000, 8, 10)
size(knns)
