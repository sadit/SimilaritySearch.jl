# This file is a part of SimilaritySearch.jl

using SimilaritySearch
using Test

function testsorted(res, st, Vsorted)
    @info "========================", (typeof(res), length(res, st), length(res), maxlength(res), st), length(Vsorted)
    @show res
    @show Vsorted
    @info "======================== starting ============="
    @test length(res, st) == length(res)
    @test minimum(res, st) == minimum(res)
    @test maximum(res, st) == maximum(res)
    @test argmin(res, st) == argmin(res)
    @test argmax(res, st) == argmax(res)

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
    @show res.id
    @show res.dist
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
    st = initialstate(res)
    V, Vsorted = create_random_array(50, k)
    for i in eachindex(V)
        st = push!(res, st, i, V[i])
    end
    
    testsorted(res, st, Vsorted)
end
