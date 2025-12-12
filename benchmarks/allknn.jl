# This file is a part of SimilaritySearch.jl

using SimilaritySearch, LinearAlgebra

function create_database(n)
    X = rand(Float32, 8, n)
    for c in eachcol(X)
        normalize!(c)
    end
    MatrixDatabase(X)
end

function main(n)
    @info "this benchmark is intended to work with multithreading enabled julia sessions"
    db = create_database(n)
    dist = NormalizedCosineDistance()
    k = 8
    @info "----- computing gold standard"
    ctx = SearchGraphContext(
        hyperparameters_callback=OptimizeParameters(MinRecall(0.9)),
        parallel_block=128
    )
    goldsearchtime = @elapsed gold_knns = allknn(ExhaustiveSearch(; db, dist), ctx, k)
    @info "----- computing search graph with k=$k"
    H = SearchGraph(; db, dist)
    index!(H, ctx)
    optimize_index!(H, ctx, ksearch=k)
    GC.enable(false)
    searchtime = @elapsed knns = allknn(H, ctx, k; sort=false, progress=nothing)
    GC.enable(true)
    n = length(db)
    @info "gold:" (; n, goldsearchtime, qps=n / goldsearchtime)
    @info "searchgraph:" (; n, searchtime, qps=n / searchtime)
    @info "recall:" macrorecall(gold_knns, knns)
end

main(10^3)
main(10^5)
