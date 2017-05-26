using SimilaritySearch
using Base.Test


@testset "exact-indexes" begin
    n = 100
    dim = 3
    S = [rand(1:10, dim) for i in 1:n]
    index = Laesa(S, L2Distance(), 5)
        
    info("done; now testing")
    @test length(index.db) == n
    item = rand(1:10, dim)
    res = search(index, item, KnnResult(10))
    @show res
end

