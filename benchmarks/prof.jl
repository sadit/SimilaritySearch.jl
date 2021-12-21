using SimilaritySearch

function main(n, m, dim)
    S = MatrixDatabase(rand(Float32, dim, n))
    Q = MatrixDatabase(rand(Float32, dim, m))

    seq = ExhaustiveSearch(SqL2Distance(), S)
    k = 10
    @info "running matrix knnresult"
    @time searchbatch(seq, Q, k)
    knnlist = [KnnResult(k) for _ in 1:m]
    @info "running vector knnresult"
    @time searchbatch(seq, Q, knnlist)
end

@info "warming"
@info size(main(100, 10, 3))
@info "large benchmark"
@info size(main(100_000, 1000, 3))