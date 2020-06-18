# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

@testset "Result set" begin
    k = 4
    V = rand(Float32, 50)
    Vsorted = sort!([Item(i, v) for (i, v) in enumerate(V)])[1:k]
    res = KnnResult(k)

    for i in eachindex(V)
        push!(res, i, V[i])
    end

    @show res Vsorted
    @test nearestdist(res) == Vsorted[1].dist
    @test [x.id for x in res] == [x.id for x in Vsorted[1:k]]
    @test [x.dist for x in res] == [x.dist for x in Vsorted[1:k]]
    
end
