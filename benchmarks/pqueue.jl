using SimilaritySearch, Random

function benchmark(res, n)
    # @info "===========", typeof(res), n
    push_item!(res, 1, rand())

    popfirst_ = pop_ = 0
    for i in 3:n
        if length(res) > 0 && rand() < 0.001
            pop_max!(res)
            pop_ += 1
        end

        #=if length(res) > 0 && rand() < 0.001
            pop_min!(res)
            popfirst_ += 1
        end=#

        # push!(res, i, rand())
        push_item!(res, i, rand())
    end

    #@info "finished", pop_, popfirst_
    res
end

function main(pqueue_, k)
    n = 1000
    res = pqueue_(k)

    # @timev "warming k=$k pqueue=$pqueue_ n=$n" benchmark(res, n)

    n = 30_000_000
    res = pqueue_(k)
    @timev "!!!!!!! k=$k pqueue=$pqueue_ n=$n" benchmark(res, n)
end

for k in [10, 100, 1000]
    main(knn, k)
    main(xknn, k)
end
