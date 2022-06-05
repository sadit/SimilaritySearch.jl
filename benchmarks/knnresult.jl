using SimilaritySearch, Random

function main(res, n)
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

        push!(res, i, rand())
    end

    #@info "finished", pop_, popfirst_
    res
end


ksearch = 10
n = 1000
res = KnnResultView(KnnResultSet(ksearch, 1), 1)

@timev main(res, n)

n = 30_000_000
res = KnnResultView(KnnResultSet(ksearch, 1), 1)
@timev main(res, n)
