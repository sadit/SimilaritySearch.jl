# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test

function testsorted(res, Vsorted)
    @info "========================", (typeof(res), length(res), maxlength(res)), length(Vsorted)
    @show res
    @show Vsorted
    @info "======================== starting ============="
    @test collect(res) == Vsorted
    @test minimum(res) == first(Vsorted)[end]
    @test maximum(res) == last(Vsorted)[end]
    @test argmin(res) == first(Vsorted)[1]
    @test argmax(res) == last(Vsorted)[1]

    @show res
    
    @test collect(res) == Vsorted

    pop!(Vsorted)
    pop!(res)
    @test collect(res) == Vsorted

    popfirst!(Vsorted)
    popfirst!(res)
    @test collect(res) == Vsorted

end

function create_random_array(n, k)
    V = rand(Float32, n)
    Vsorted = sort!([IdWeight(i, v) for (i, v) in enumerate(V)], by=x->x.weight)[1:k]
    V, Vsorted
end

@testset "shifted vector-based result set" begin
    k = 10
    res = KnnResult(k)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        push_item!(res, i, V[i])
    end
    
    testsorted(res, copy(Vsorted))
end