# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using LinearAlgebra
using JSON
using Test

function test_knr_vectors(perf, knr, lower)
    @time p = probe(perf, knr)
    @test p.macrorecall >= lower
    p
end

@testset "indexing vectors with Knr" begin
    # NOTE: The following algorithms are complex enough to say we are testing
    # it doesn't have syntax errors and achieves some expected quality
    # a more grained test functions are required
    k = 10
    n = 10000 # number of items in the dataset
    m = 100  # number of queries
    dim = 8  # vector's dimension
    db = [rand(Float32, dim) |> normalize! for i in 1:n]
    queries = [rand(Float32, dim) |> normalize! for i in 1:m]

    dist = L2Distance()
    seq = ExhaustiveSearch(dist, db, k)
    perf = Performance(seq, queries, k)

    knr = Knr(dist, db; numrefs=64, kbuild=2)
    p = test_knr_vectors(perf, knr, 0.7)
    @info "-- kbuild=3 $(JSON.json(p))"
    optimize!(perf, knr; recall=0.95, ksearch=k)
    p = test_knr_vectors(perf, knr, 0.95)
    @info "-- Optimized kbuild=3 $(JSON.json(knr.opts)) -- $(JSON.json(p))"
end

