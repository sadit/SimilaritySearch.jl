using SimilaritySearch
using Test
#
# This file contains a set of tests for SearchGraph over databases of vectors (of Float32)
#

dist = l2_distance

function create_dataset(n, dim)
    [rand(Float32, dim) for i in 1:n]
end


@testset "kcenters" begin
    n = 10000
    db = create_dataset(n, 2)
    k = 100

    centers, epsilon = kcenters(dist, db, k)
    @test length(centers) == k
    @test sum([dist(db[centers[i]], db[centers[k]]) for i in 1:(k-1)] .>= epsilon) == k-1

    index = fit(SearchGraph, dist, db)
    for (i, p) in allknn(index, dist, db[centers], k=7) |> enumerate
        @show i p
    end
end
exit(0)

@testset "All K-NN Sequential" begin
    n = 10000
    db = create_dataset(n, 2)
    @time graph = fit(SearchGraph, dist, db)
    index = fit(Sequential, db)
    k = 3
    @time A = allknn(index, dist, k=k)
    @time B = allknn(graph, dist, k=k)

    f = 1 / n
    s = 0.0
    for i in 1:n
        a = Set(p.objID for p in A[i])
        b = Set(p.objID for p in B[i])
        # @show a b intersect(a, b)
        s += f * length(intersect(a, b)) / k
    end

    @test s > 0.9
end

