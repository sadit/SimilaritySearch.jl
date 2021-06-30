# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export allknn

"""
    allknn(index::Index, dist::PreMetric; k::Int=1)

Finds k-nearest-neighbors for all items in the index's dataset; removes the object itself from its knn 
"""
function allknn(index::Index, dist::PreMetric; k::Int=1)
    n = length(index)
    A = [KnnResult(k+1) for i in 1:n]

    for i in 1:n
        x = index[i]
        res = A[i]
        search(index, dist, x, res)
        if first(res).id == i
            popfirst!(res)
        end

        if (i % 1000) == 1
            println(stderr, "computing all-knn for index $(typeof(index)); k=$(k); advance $i of n=$n")
        end
    end

    A
end

"""
    allknn(index::Index, dist::PreMetric, X::AbstractVector; k::Int=1)

Finds k-nearest-neighbors for all items in the given X
"""
function allknn(index::Index, dist::PreMetric, X::AbstractVector; k::Int=1)
    n = length(X)
    A = [KnnResult(k) for i in 1:n]

    for i in 1:n
        x = X[i]
        res = A[i]
        search(index, dist, x, res)

        if (i % 1000) == 1
            println(stderr, "computing all-knn for index $(typeof(index)); k=$(k); advance $i of n=$n")
        end
    end

    A
end


"""
    allknn(index::SearchGraph{T}, dist::PreMetric; k::Int=1)

Finds k-nearest-neighbors for all items in the index's dataset; removes the object itself from its knn.
This function is optimized for the SearchGraph similarity-index
"""
function allknn(index::SearchGraph, dist::PreMetric; k::Int=1)
    n = length(index)
    A = [KnnResult(k+1) for i in 1:n]
    # TODO: Test caching the distance function
    for i in 1:n
        q = index[i]
        res = A[i]
        hints = [p.id for p in res]
        append!(hints, index.links[i])
        empty!(res)
        search(index, dist, q, res, hints=hints)
        if first(res).id == i
            popfirst!(res)
        end
        
        for p in res
            if p.id > i
                push!(A[p.id], i, p.dist)
            end
        end

        if (i % 1000) == 1
            println(stderr, "computing all-knn for $(index.search_algo); k=$(k); advance $i of n=$n")
        end
    end

    A
end

## """
## Finds k-nearest-neighbors for all items in the given dataset (represented as a SearchGraph object).
## 
## This function is optimized for the SearchGraph similarity-index
## """
## function allknn(index::SearchGraph{T}, dist::Function, X::SearchGraph{T}; k::Int=1, bsize::Int=4) where T
##     n = length(X.db)
##     A = Vector{KnnResult}(undef, n)
##     # TODO: Test caching the distance function
## 
##     L = [(id=n, hints=rand(1:n, bsize))]
##     while length(L) > 0
##         t = pop!(L)
##         search(index, dist, X.db[i], )
##     end
##     hints = Int32[]
##     for i in 1:n
##         q = X.db[i]
## 
##         
##         if A[i] == undef
##             A[i] = KnnResult(k)
##         else
##             continue
##         end
## 
##         # hints = rand(1:n, bsize)
##         res = search(index, dist, q, A[i], hints=hints)
##         push!(L, (id=i, res))
##         clear!(hints)
##         append!(index.links[first(res).id])
##         if (i % 10000) == 1
##             println(stderr, "computing all-knn for $(index.search_algo); k=$(k); advance $i of n=$n")
##         end
##     end
## 
##     A
## end
