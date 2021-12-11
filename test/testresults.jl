# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test

function testsorted(res, st, Vsorted)
    @info "========================", (typeof(res), length(res, st), maxlength(res, st), st)
    @show res Vsorted

    @test minimum(res, st) == first(Vsorted)[end]
    @test maximum(res, st) == last(Vsorted)[end]
    @test argmin(res, st) == first(Vsorted)[1]
    @test argmax(res, st) == last(Vsorted)[1]

    @show res
    @show st
    @test collect(idview(res, st)) == first.(Vsorted)
    @test collect(distview(res, st)) == last.(Vsorted)
    
    pop!(Vsorted)
    _, st = pop!(res, st)
    @test collect(idview(res, st)) == first.(Vsorted)
    @test collect(distview(res, st)) == last.(Vsorted)

    popfirst!(Vsorted)
    _, st = popfirst!(res, st)
    @test collect(idview(res, st)) == first.(Vsorted)
    @test collect(distview(res, st)) == last.(Vsorted)
end

function create_random_array(n, k)
    V = rand(Float32, n)
    Vsorted = sort!([i => v for (i, v) in enumerate(V)], by=x->x[2])[1:k]
    V, Vsorted
end

@testset "Matrix-based result set" begin
    k = 10
    res = KnnResultMatrix(k)
    st = initialstate(res)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        st = push!(res, st, i, V[i])
    end
    
    testsorted(res, st, Vsorted)
end

@testset "shifted vector-based result set" begin
    k = 10
    res = KnnResult(k)
    st = initialstate(res)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        st = push!(res, st, i, V[i])
    end
    
    testsorted(res, st, Vsorted)
end

