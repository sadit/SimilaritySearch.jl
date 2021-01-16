# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

function test_pivs(pivs, queries, ksearch, gold)
    results = [search(pivs, q, KnnResult(ksearch)) for q in queries]

    S = scores.(gold, results)
    @test scores(S).macro_recall >= 0.99
end

@testset "indexing vectors with PivotedSearch" begin
    db = [rand(Float32, 4) for i in 1:1000]
    queries = [rand(Float32, 4) for i in 1:30]
    ksearch = 10
    dist = L2Distance()
    seq = ExhaustiveSearch(dist, db, ksearch)
    gold = [search(seq, q, KnnResult(ksearch)) for q in queries]

    # random pivots
    test_pivs(PivotedSearch(dist, db, 8), queries, ksearch, gold)

    # pivots selected with SSS criterion
    test_pivs(sss(dist, db), queries, ksearch, gold)

    # pivots selected with distant tournament criterion
    test_pivs(distant_tournament(dist, db, 8), queries, ksearch, gold)

    # Kvp    
    test_pivs(Kvp(dist, db), queries, ksearch, gold)
end
