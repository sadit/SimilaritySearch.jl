# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export allknn, pairwise, simetricpairwise

"""
    pairwise(dist::Function, X::AbstractVector{PointType}, Y::AbstractVector{PointType})::Matrix{Float64} where PointType
    pairwise(dist::Function, X::AbstractVector{PointType})::Matrix{Float64} where PointType

Computes the distance matrix among all pairs in ``X \\times Y``
"""
function pairwise(dist::Function, X::AbstractVector{PointType}, Y::AbstractVector{PointType})::Matrix{Float64} where PointType
    A = Matrix{Float64}(undef, length(X), length(Y))

    @inbounds for i in eachindex(X), j in eachindex(Y)
        A[i, j] = dist(X[i], Y[j])
    end

    A
end

pairwise(dist::Function, X::AbstractVector) = pairwise(dist, X, X)

"""
    simetricpairwise(dist::Function, X::AbstractVector{PointType}, Y::AbstractVector{PointType})::Matrix{Float64} where PointType
    simetricpairwise(dist::Function, X::AbstractVector{PointType})::Matrix{Float64} where PointType

Computes the superior triangular metrix of of the distance matrix among items in `X` and `Y`
"""
function simetricpairwise(dist::Function, X::AbstractVector{PointType}, Y::AbstractVector{PointType})::Matrix{Float64} where PointType
    n = length(X)
    m = length(Y)

    A = Matrix{Float64}(undef, n, m)
    @inbounds for i in 1:n
        u = X[i]
        A[i, i] = dist(u, Y[i])
        for j in i+1:m
            d = dist(u, Y[j])
            A[i, j] = d
            A[j, i] = d
        end
    end

    A
end

simetricpairwise(dist::Function, X::AbstractVector) = simetricpairwise(dist, X, X)

"""
    allknn(index::Index, dist::Function; k::Int=1)

Finds k-nearest-neighbors for all items in the index's dataset; removes the object itself from its knn 
"""
function allknn(index::Index, dist::Function; k::Int=1)
    n = length(index.db)
    A = [KnnResult(k+1) for i in 1:n]

    for i in 1:n
        x = index.db[i]
        res = A[i]
        search(index, dist, x, res)
        if first(res).objID == i
            popfirst!(res)
        end

        if (i % 1000) == 1
            println(stderr, "computing all-knn for index $(typeof(index)); k=$(k); advance $i of n=$n")
        end
    end

    A
end

"""
    allknn(index::Index, dist::Function, X::AbstractVector; k::Int=1)

Finds k-nearest-neighbors for all items in the given X
"""
function allknn(index::Index, dist::Function, X::AbstractVector; k::Int=1)
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
    allknn(index::SearchGraph{T}, dist::Function; k::Int=1)

Finds k-nearest-neighbors for all items in the index's dataset; removes the object itself from its knn.
This function is optimized for the SearchGraph similarity-index
"""
function allknn(index::SearchGraph, dist::Function; k::Int=1)
    n = length(index.db)
    A = [KnnResult(k+1) for i in 1:n]
    # TODO: Test caching the distance function
    for i in 1:n
        q = index.db[i]
        res = A[i]
        hints = [p.objID for p in res]
        append!(hints, index.links[i])
        empty!(res)
        search(index, dist, q, res, hints=hints)
        if first(res).objID == i
            popfirst!(res)
        end
        
        for p in res
            if p.objID > i
                push!(A[p.objID], i, p.dist)
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
##         append!(index.links[first(res).objID])
##         if (i % 10000) == 1
##             println(stderr, "computing all-knn for $(index.search_algo); k=$(k); advance $i of n=$n")
##         end
##     end
## 
##     A
## end
