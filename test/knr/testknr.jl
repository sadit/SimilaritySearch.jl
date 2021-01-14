# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using LinearAlgebra
using Test


function test_vectors(create_index, dist::PreMetric, ksearch)
    n = 1000 # number of items in the dataset
    m = 100  # number of queries
    dim = 3  # vector's dimension

    db = [rand(Float32, dim) |> normalize! for i in 1:n]
    queries = [rand(Float32, dim) |> normalize! for i in 1:m]

    index = create_index(db)
    optimize!(index, dist, recall=0.9, k=10)
    perf = Performance(dist, index.db, queries, expected_k=ksearch)
    p = probe(perf, index, dist)
    @show dist, p
    @test p.recall > 0.8

    @info "adding more items"
    for item in queries
        push!(index, dist, item)
    end
    perf = Performance(dist, index.db, queries, expected_k=1)
    p = probe(perf, index, dist)
    @show dist, p
    @test p.recall > 0.999
    return p
end

@testset "indexing vectors" begin
    # NOTE: The following algorithms are complex enough to say we are testing it doesn't have syntax errors, a more grained test functions are required
    ksearch = 10
    σ = 127
    κ = 3

    for dist in [
        SqL2Distance(), # 1.0 -> metric, < 1.0 if dist is not a metric
    ]
        @testset "indexing vectors with Knr and $dist" begin
            p = test_vectors((db) -> fit(Knr, dist, db, numrefs=σ, k=κ), dist, ksearch)
        end
    end
end