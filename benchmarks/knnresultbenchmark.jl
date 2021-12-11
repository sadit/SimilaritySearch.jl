using SimilaritySearch, Random

function main(res, n=30_000_000)
    Random.seed!(0)
    @info "===========", typeof(res), n
    st = initialstate(res)
    st = push!(res, st, 1, rand())
    st = push!(res, st, 2, rand())


    for i in 3:n
        if length(res, st) > 0 && rand() < 0.001
            _, st = pop!(res, st)
        end

        if length(res, st) > 0 && rand() < 0.001
            _, st = popfirst!(res, st)
        end
        st = push!(res, st, i, rand())
    end

    res
end

ksearch = 10
n = 1000
main(KnnResult(ksearch), n)
main(KnnResultShifted(ksearch), n)

n = 30_000_000
@timev main(KnnResult(ksearch), n)
@timev main(KnnResultShifted(ksearch), n)
