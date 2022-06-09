using SimilaritySearch

function main(n, m, dim)
    S = MatrixDatabase(rand(Float32, dim, n))
    Q = MatrixDatabase(rand(Float32, dim, m))

    seq = ExhaustiveSearch(SqL2Distance(), S)
    k = 10
    @info "running KnnResultSet"
    @time searchbatch(seq, Q, k)
    @info "running KnnResult array"
    @time begin
        knnlist = [KnnResult(k) for _ in 1:m]
        searchbatch(seq, Q, knnlist)
    end

    @info "running KnnResult array + copying to matrices"
    begin
        knnlist = [KnnResult(k) for _ in 1:m]
        D = Matrix{Float32}(undef, k, m)
        I = Matrix{Int32}(undef, k, m)
        @time begin
            searchbatch(seq, Q, knnlist)
            @inbounds for i in eachindex(knnlist)
                k_ = length(knnlist[i])
                I[1:k_, i] .= knnlist[i].id
                D[1:k_, i] .= knnlist[i].dist
            end
        end
        D
    end
end

@info "warming"
@info size(main(100, 10, 3))
@info "============= benchmark ================="
@info size(main(1000_000, 1000, 3))
