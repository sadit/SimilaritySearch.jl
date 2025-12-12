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

function main(pqtype, n, k)
    res = knnqueue(pqtype, k)
    @time benchmark(res, n)
end

for pqtype in [KnnSorted, KnnHeap]
    @info "================= type=$pqtype ================="
    for n in [100, 10^6]
        for k in [10, 100, 1000, 10000]
            @info "=== type=$pqtype n=$n k=$k ==="
            main(pqtype, n, k)
        end
    end
end
