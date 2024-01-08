# This file is a part of SimilaritySearch.jl

using SimilaritySearch, LinearAlgebra

function create_database()
    X = rand(Float32, 8, 100_000)
    for c in eachcol(X) normalize!(c) end
    MatrixDatabase(X)
end

function main()
    @info "this benchmark is intended to work with multithreading enabled julia sessions"
    db = create_database()
    dist = NormalizedCosineDistance()
    k = 32
    @info "----- computing gold standard"
    ctx = SearchGraphContext(
                             hyperparameters_callback=OptimizeParameters(MinRecall(0.9)),
                             logger=nothing,
                             parallel_block=128
                            )
    goldsearchtime = @elapsed gI, gD = allknn(ExhaustiveSearch(; db, dist), ctx, k)
    @info "----- computing search graph"
    H = SearchGraph(; db, dist)
    index!(H, ctx)
    optimize_index!(H, ctx, ksearch=k)
    searchtime = @elapsed hI, hD = allknn(H, ctx, k)
    n = length(db)
    @info "gold:" (; n, goldsearchtime, qps=n/goldsearchtime)
    @info "searchgraph:" (; n, searchtime, qps=n / searchtime)
    @info "recall:" macrorecall(gI, hI)
end

main()
