# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
using Test

@testset "Result set" begin
    k = 4
    V = rand(Float32, 50)
    Vsorted = sort!([i => v for (i, v) in enumerate(V)], by=x->x[2])[1:k]
    res = KnnResult(k)

    for i in eachindex(V)
        push!(res, i, V[i])
    end

    @show res Vsorted
    @test minimum(res) == first(Vsorted)[end]
    @test maximum(res) == last(Vsorted)[end]
    @test argmin(res) == first(Vsorted)[1]
    @test argmax(res) == last(Vsorted)[1]

    @test collect(keys(res)) == first.(Vsorted)
    @test collect(values(res)) == last.(Vsorted)
end
