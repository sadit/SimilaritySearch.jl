using SimilaritySearch, Random

function main(k, n=30_000_000)
    Random.seed!(0)

    #res = KnnResultShifted(k)
    res = KnnResult(k)
    @show typeof(res), k, n
    @info @timed push!(res, 1 => rand())
    @info @timed push!(res, 2 => rand())

    @info @timed for i in 3:n
        rand() < 0.001 && pop!(res)
        rand() < 0.001 && popfirst!(res)
        push!(res, i => rand())
    end

    res
end

main(10)