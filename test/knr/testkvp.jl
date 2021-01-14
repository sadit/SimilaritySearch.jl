# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

function test_vectors(create_index, dist::PreMetric, ksearch)
    n = 300 # number of items in the dataset
    m = 30  # number of queries
    dim = 3  # vector's dimension

    db = [rand(Float32, dim) for i in 1:n]
    queries = [rand(Float32, dim) for i in 1:m]

    index = create_index(db)
    # testing pushes
    push!(index, dist, rand(Float32, dim))
    @test length(index.db) == n + 1
    perf = Performance(dist, index.db, queries, expected_k=ksearch)
    p = probe(perf, index, dist)
    @show dist, p
    return p
end

@testset "indexing vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    local index
    @testset "indexing vectors with Kvp" begin
        for (recall_lower_bound, dist) in [
            (1.0, L2Distance()), # 1.0 -> metric, < 1.0 if dist is not a metric
        ]
            @show recall_lower_bound, dist
            p = test_vectors((db) -> fit(Kvp, dist, db, 3, 32), dist, ksearch)
            @test p.recall >= recall_lower_bound * 0.99 # not 1 to allow some "numerical" deviations
        end
    end
end