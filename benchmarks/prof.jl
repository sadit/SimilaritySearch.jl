using SimilaritySearch

function main()
    S = MatrixDatabase(rand(Float32, 3, 1000))
    Q = MatrixDatabase(rand(Float32, 3, 3))
    seq = ExhaustiveSearch(SqL2Distance(), S)
    searchbatch(seq, Q, 10)
end

display(main())