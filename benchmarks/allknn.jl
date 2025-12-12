# This file is a part of SimilaritySearch.jl

#using SimilaritySearch, FixedSizeArrays, LinearAlgebra, Random
using SimilaritySearch, LinearAlgebra, Random


function create_database(dim, n)
    rng = Xoshiro(n)
    X = Matrix{Float32}(undef, dim, n)
    #X = FixedSizeMatrix{Float32}(undef, dim, n)
    rand!(rng, X)
    for c in eachcol(X)
        normalize!(c)
    end

    MatrixDatabase(X)
end

function run(dim, n, k)
    @info "=========================== dim=$dim n=$n k=$k ============================"
    db = create_database(dim, n)
    dist = NormalizedCosineDistance()
    @info "----- computing gold standard"
    ctx = SearchGraphContext(
        hyperparameters_callback=OptimizeParameters(MinRecall(0.99)),
        parallel_block=1024
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

#function @main(ARGS)
begin
    run(8, 10^3, 8)
    run(8, 10^5, 8)
    #    return 0
end
