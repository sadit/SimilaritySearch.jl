using SimilaritySearch

function main(n, m, dim)
    db = MatrixDatabase(rand(Float32, dim, n))
    queries = MatrixDatabase(rand(Float32, dim, m))
    seq = ExhaustiveSearch(; db)
    k = 10
    @time searchbatch(seq, queries, k)
end

@info "warming"
main(100, 10, 3)
@info "large benchmark"
I, D = main(100_000, 1000, 3)
size(I), size(D)
