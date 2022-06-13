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

@testset "vector-based result set" begin
    k = 10
    res = KnnResult(k)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        push!(res, i, V[i])
    end
    
    testsorted(res, copy(Vsorted))
end

@testset "set of knn results" begin
    k = 10
    S = KnnResultSet(k, 1)
    res = KnnResult(S, 1)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        push!(res, i, V[i])
    end
    
    testsorted(res, copy(Vsorted))
end

@testset "equality" begin
    k = 10
    m = 1000
    n = 10_000
    S = KnnResultSet(k, m)
    R = KnnResult(k)
    for i in 1:m
        test = rand(n)
        V = KnnResult(S, i)
        R = reuse!(R)
        for (j, d) in enumerate(test)
            push!(R, j, d)
            push!(V, j, d)
        end
        
        @test idview(V) == idview(R)
    end
end
