using SimilaritySearch, Random, LoopVectorization, StaticArrays

struct SqL2Turbo <: SemiMetric end

function SimilaritySearch.evaluate(::SqL2Turbo, u::T, v::T) where T
    d = zero(Float32)
    @turbo for i in eachindex(u, v)
        d += (u[i] - v[i])^2
    end

    d
end

generate_dataset(dim, n) = [rand(Float32, dim) for i in 1:n]

function mysearchbenchmark(seq, Q, R)
    # warming
    search(seq, Q[1], R[1])

    @elapsed for i in eachindex(Q)
        empty!(R[i])
        search(seq, Q[i], R[i])
    end
end

function main_exhaustive(n, nqueries, k, dim)
    S = generate_dataset(dim, n)
    Q = generate_dataset(dim, nqueries)
    R = [KnnResult(k) for i in eachindex(Q)]
    
    #@btime mysearch($seq, $Q, $R)
    seq = ExhaustiveSearch(SqL2Distance(), S)
    simdtime = mysearchbenchmark(seq, Q, R)

    seq = ExhaustiveSearch(SqL2Turbo(), S)
    turbotime = mysearchbenchmark(seq, Q, R)


    S = SVector{dim}.(S)
    Q = SVector{dim}.(Q)

    seq = ExhaustiveSearch(SqL2Turbo(), S)
    staticsimdtime = mysearchbenchmark(seq, Q, R)

    seq = ExhaustiveSearch(SqL2Turbo(), S)
    staticturbotime = mysearchbenchmark(seq, Q, R)
    
    (simd=(vector=simdtime, svector=staticsimdtime), turbo=(vector=turbotime, svector=staticturbotime))
end

function main()
    k = 1
    n = 100_000
    nqueries = 1000
    for dim in [4, 8, 16, 32, 64, 128, 256, 512]
        #for dim in [10, 30, 100, 200, 300]
        @info "====================================================="
        @info "# objects: $n, # queries: $nqueries, knn: $k, dim=$dim"
        @info main_exhaustive(n, nqueries, k, dim)
    end
end

main()