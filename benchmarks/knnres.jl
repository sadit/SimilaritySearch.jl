using SimilaritySearch, Random

function main(k, n=30_000_000)
    Random.seed!(0)

    res = KnnResult(k)
    push!(res, 0 => rand())
    @time for i in 1:n
        push!(res, i => rand())
    end
end

main(10)