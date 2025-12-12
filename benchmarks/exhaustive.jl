using SimilaritySearch

function main(n, m, dim, k)
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))
    seq = ExhaustiveSearch(; db)
    @time searchbatch(seq, getcontext(seq), queries, k)
end

@info "warming"
main(100, 10, 8, 2)
@info "large benchmark"
knns = main(100_000, 1000, 8, 10)
size(knns)
