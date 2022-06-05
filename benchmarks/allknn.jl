using SimilaritySearch, LinearAlgebra

function create_database()
    X = rand(Float32, 8, 100_000)
    for c in eachcol(X) normalize!(c) end
    MatrixDatabase(X)
end

function main()
    db = create_database()
    dist = NormalizedCosineDistance()
    k = 32
    goldsearchtime = @elapsed gI, gD = allknn(ExhaustiveSearch(; db, dist), k; parallel_block=512)
    H = SearchGraph(; db, dist, verbose=false)
    index!(H; parallel_block=128)
    optimize!(H, MinRecall(0.9), ksearch=k)
    searchtime = @elapsed hI, hD = allknn(H, k; parallel_block=512)
    n = length(db)
    @info "gold:" (; n, goldsearchtime, qps=n/goldsearchtime)
    @info "searchgraph:" (; n, searchtime, qps=n / searchtime)
    @info "recall:" macrorecall(gI, hI)
end

main()
