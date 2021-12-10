using SimilaritySearch

function main()
    S = MatrixDatabase(rand(Float32, 3, 1000))
    Q = MatrixDatabase(rand(Float32, 3, 3))
    seq = ExhaustiveSearch(SqL2Distance(), S)
    res = [KnnResult(10) for i in eachindex(Q)]
    searchbatch(seq, Q, res)
    for r in res
        empty!(r)
    end

    @timed begin
        searchbatch(seq, Q, res)
        nothing
    end
end

@info main()