using SimilaritySearch, LinearAlgebra

function create_database()
    X = rand(Float32, 8, 100_000)
    for c in eachcol(X) normalize!(c) end
    MatrixDatabase(X)
end

function main()
    db = create_database()
    H = SearchGraph(; db, dist=NormalizedCosineDistance(), verbose=false)
    index!(H; parallel_block=128)
    searchtime = @elapsed hI, hD = allknn(H, 32; parallel=true)
    n = length(db)
    @show (; n, searchtime, qps=n / searchtime)
end

main()