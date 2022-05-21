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

function testsorted2(res, st, Vsorted)
    @info "========================", (typeof(res), length(res, st), length(res), maxlength(res), st), length(Vsorted)
    @show res
    @show Vsorted
    @info "======================== starting ============="
    @test length(res, st) == length(res)
    @test minimum(res, st) == minimum(res)
    @test maximum(res, st) == maximum(res)
    @test argmin(res, st) == argmin(res)
    @test argmax(res, st) == argmax(res)

    @show res.id, res.dist, st, Vsorted
    @test collect(res) == Vsorted
    @test minimum(res, st) == first(Vsorted)[end]
    @test maximum(res, st) == last(Vsorted)[end]
    @test argmin(res, st) == first(Vsorted)[1]
    @test argmax(res, st) == last(Vsorted)[1]
    
    @show res
    @show st

    @info "collect:" => collect(idview(res, st))
    @info "========" => first.(Vsorted)
    
    @test collect(idview(res, st)) == first.(Vsorted)
    @test collect(distview(res, st)) == last.(Vsorted)

    pop!(Vsorted)
    _, st = pop!(res, st)
    @test collect(idview(res, st)) == first.(Vsorted)
    @test collect(distview(res, st)) == last.(Vsorted)
    @test length(res, st) == length(res)
    @test collect(res) == Vsorted

    popfirst!(Vsorted)
    _, st = popfirst!(res, st)

    @info "b   collect id:" => collect(idview(res, st))
    @info "b collect dist:" => collect(distview(res, st))
    @info "b ========" => first.(Vsorted)
    @info "b ========" => last.(Vsorted)
    @show res.id, res.dist, st
    @info collect(res)
    @info "c   collect id:" => collect(idview(res, st))
    @info "c collect dist:" => collect(distview(res, st))
    
    @test collect(idview(res, st)) == first.(Vsorted)
    @test collect(distview(res, st)) == last.(Vsorted)
    @test length(res, st) == length(res)
    @test collect(res) == Vsorted

end

function create_random_array(n, k)
    V = rand(Float32, n)
    Vsorted = sort!([i => v for (i, v) in enumerate(V)], by=x->x[2])[1:k]
    V, Vsorted
end

@testset "shifted vector-based result set" begin
    k = 10
    res = KnnResult(k)
    res2 = KnnResultShift(k)
    st = initialstate(res2)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        push!(res, i, V[i])
        st = push!(res2, st, i, V[i])
    end
    
    testsorted(res, copy(Vsorted))
    testsorted2(res2, st, Vsorted)
end
