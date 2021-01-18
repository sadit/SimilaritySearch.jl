# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

function test_pivs(perf, pivs)
    @test probe(perf, pivs).macrorecall >= 0.99
end

@testset "indexing vectors with PivotedSearch" begin
    db = [rand(Float32, 4) for i in 1:1000]
    queries = [rand(Float32, 4) for i in 1:30]
    ksearch = 10
    dist = L2Distance()
    seq = ExhaustiveSearch(dist, db)
    perf = Performance(seq, queries, ksearch)

    # random pivots
    test_pivs(perf, PivotedSearch(dist, db, 8))

    # pivots selected with SSS criterion
    test_pivs(perf, sss(dist, db))

    # pivots selected with distant tournament criterion
    test_pivs(perf, distant_tournament(dist, db, 8))

    # Kvp    
    test_pivs(perf, Kvp(dist, db))
end
