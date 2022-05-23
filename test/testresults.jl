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
    
    @test idview(res) == first.(Vsorted)
    @test distview(res) == last.(Vsorted)

    pop!(Vsorted)
    pop!(res)
    @test idview(res) == first.(Vsorted)
    @test distview(res) == last.(Vsorted)
    @test collect(res) == Vsorted

    popfirst!(Vsorted)
    popfirst!(res)

    @info "b   collect id:" => idview(res)
    @info "b collect dist:" => distview(res)
    @info "b ========" => first.(Vsorted)
    @info "b ========" => last.(Vsorted)
    @show idview(res)
    @show distview(res) 
    @test idview(res) == first.(Vsorted)
    @test distview(res) == last.(Vsorted)
    @test collect(res) == Vsorted

end

function create_random_array(n, k)
    V = rand(Float32, n)
    Vsorted = sort!([i => v for (i, v) in enumerate(V)], by=x->x[2])[1:k]
    V, Vsorted
end

@testset "shifted vector-based result set" begin
    k = 10
    for _ in 1:100
        res = KnnResultSingle(k)
        res2 = KnnResultView(KnnResultSet(k, 1), 1)
        V, Vsorted = create_random_array(50, k)
        for i in eachindex(V)
            push!(res, i, V[i])
            push!(res2, i, V[i])
        end
        
        testsorted(res, copy(Vsorted))
        testsorted(res2, copy(Vsorted))
    end
end
