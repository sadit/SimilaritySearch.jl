# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

@testset "Result set" begin
    k = 4
    V = rand(Float32, k*k)
    Vsorted = sort(V)[1:k]

    res = SortedKnnResult(k)
    sres = SortedKnnResult(k)
    nn = SortedKnnResult(1)

    for i=1:length(V)
        push!(res, Item(i, V[i]))
        push!(sres, Item(i, V[i]))
        push!(nn, Item(i, V[i]))
    end
    
    @test [x.id for x in res] == [x.id for x in sres]
    @test [x.dist for x in res] == [x.dist for x in sres] == Vsorted
    @test nearest(nn).dist == Vsorted[1]
end
