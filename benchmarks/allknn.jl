using SimilaritySearch, LinearAlgebra

function create_database()
    X = rand(Float32, 4, 100_000)
    for c in eachcol(X) normalize!(c) end
    MatrixDatabase(X)
end

function main()
    db = create_database()
    H = SearchGraph(; db, dist=NormalizedCosineDistance(), verbose=false)
    index!(H; parallel_block=128)
    @time hI, hD = allknn(H, 32; parallel=true)
end

main()