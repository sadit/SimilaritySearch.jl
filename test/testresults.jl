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
    res = KnnResult(k)
    st = initialstate(res)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        st = push!(res, st, i, V[i])
    end
    
    testsorted(res, st, Vsorted)
end

@testset "shifted vector-based result set" begin
    k = 10
    res = KnnResultShifted(k)
    st = initialstate(res)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        st = push!(res, st, i, V[i])
    end
    
    testsorted(res, st, Vsorted)
end

exit(0)

@testset "Vector-based result set" begin
    k = 4
    V = rand(Float32, 50)
    Vsorted = sort!([i => v for (i, v) in enumerate(V)], by=x->x[2])[1:k]
    res = KnnResultVector(k)

    n = 0
    for i in eachindex(V)
        n = push!(res, n, i, V[i])
    end

    @show res Vsorted
    @test getdist(res, 1) == first(Vsorted)[end]
    @test getdist(res, n) == last(Vsorted)[end]
    @test getid(res, 1) == first(Vsorted)[1]
    @test getid(res, n) == last(Vsorted)[1]

    @test collect(getidlist(res, n)) == first.(Vsorted)
    @test collect(getdistlist(res, n)) == last.(Vsorted)
end


@testset "Shifted result set" begin
    k = 4
    V = rand(Float32, 50)
    Vsorted = sort!([i => v for (i, v) in enumerate(V)], by=x->x[2])[1:k]
    res = KnnResultShifted(k)

    n = 0
    for i in eachindex(V)
        n = push!(res, n, i, V[i])
    end

    @show res Vsorted
    @test getdist(res, 1) == first(Vsorted)[end]
    @test getdist(res, n) == last(Vsorted)[end]
    @test getid(res, 1) == first(Vsorted)[1]
    @test getid(res, n) == last(Vsorted)[1]

    @test collect(getidlist(res, n)) == first.(Vsorted)
    @test collect(getdistlist(res, n)) == last.(Vsorted)
end
