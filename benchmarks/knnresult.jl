using SimilaritySearch, Random

function benchmark(res, n)
    # @info "===========", typeof(res), n
    push!(res, 1, rand())

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

<<<<<<< HEAD
=======
        # push!(res, i, rand())
>>>>>>> copying-knnresult-again
        push!(res, i, 3 * maximum(res) * rand())
    end

    #@info "finished", pop_, popfirst_
    res
end

<<<<<<< HEAD
function main(k, n)
    res = KnnResult(k)
    @info k n typeof(res)
    @time benchmark(res, n)

    S = KnnResultSet(k, 1)
    res = KnnResult(S, 1)
    @info k n typeof(res)
    @time benchmark(res, n)
end

@info "==== warming"
main(10, 1000)

@info "==== running benchmark"
main(10, 30_000_000)

=======
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
>>>>>>> copying-knnresult-again
