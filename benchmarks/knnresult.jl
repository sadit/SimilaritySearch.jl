using SimilaritySearch, Random

function benchmark(res, n)
    # @info "===========", typeof(res), n
    push_item!(res, 1, rand())

    popfirst_ = pop_ = 0
    for i in 3:n
        if length(res) > 0 && rand() < 0.001
            pop!(res)
            pop_ += 1
        end

        if length(res) > 0 && rand() < 0.001
            popfirst!(res)
            popfirst_ += 1
        end

        # push!(res, i, rand())
        push_item!(res, i, 3 * maximum(res) * rand())
    end

    #@info "finished", pop_, popfirst_
    res
end

function main()
    ksearch = 10
    n = 1000
    res = KnnResult(ksearch)

    @timev benchmark(res, n)

    n = 30_000_000
    res = KnnResult(ksearch)
    @timev benchmark(res, n)
end

main()
