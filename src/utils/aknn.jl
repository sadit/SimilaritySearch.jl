#  Copyright 2019 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

export allknn

"""
Finds k-nearest-neighbors for all items in the index's dataset; removes the object itself from its knn 
"""
function allknn(index::Index, dist::Function; k::Int=1) where T
    n = length(index.db)
    A = [KnnResult(k+1) for i in 1:n]

    for i in 1:n
        x = index.db[i]
        res = A[i]
        search(index, dist, x, res)
        if first(res).objID == i
            popfirst!(res)
        end

        if (i % 10000) == 1
            println(stderr, "computing all-knn for index $(typeof(index)); k=$(k); advance $i of n=$n")
        end
    end

    A
end

"""
Finds k-nearest-neighbors for all items in the given X
"""
function allknn(index::Index, dist::Function, X::AbstractVector{T}; k::Int=1) where T
    n = length(X)
    A = [KnnResult(k) for i in 1:n]

    for i in 1:n
        x = X[i]
        res = A[i]
        search(index, dist, x, res)

        if (i % 10000) == 1
            println(stderr, "computing all-knn for index $(typeof(index)); k=$(k); advance $i of n=$n")
        end
    end

    A
end


"""
Finds k-nearest-neighbors for all items in the index's dataset; removes the object itself from its knn.

This function is optimized for the SearchGraph similarity-index
"""
function allknn(index::SearchGraph{T}, dist::Function; k::Int=1) where T
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

        if (i % 10000) == 1
            println(stderr, "computing all-knn for $(index.search_algo); k=$(k); advance $i of n=$n")
        end
    end

    A
end

"""
Finds k-nearest-neighbors for all items in the given dataset (represented as a SearchGraph object).

This function is optimized for the SearchGraph similarity-index
"""
function allknn(index::SearchGraph{T}, dist::Function, X::SearchGraph{T}; k::Int=1, bsize::Int=4) where T
    n = length(X.db)
    A = Vector{KnnResult}(undef, n)
    # TODO: Test caching the distance function
    L = [(id=1, hints=rand(1:n, bsize))]
    hints = Int32[]
    for i in 1:n
        q = X.db[i]

        if A[i] == undef
            A[i] = KnnResult(k)
        else
            continue
        end

        # hints = rand(1:n, bsize)
        res = search(index, dist, q, A[i], hints=hints)
        push!(L, (id=i, res))
        clear!(hints)
        append!(index.links[first(res).objID])
        if (i % 10000) == 1
            println(stderr, "computing all-knn for $(index.search_algo); k=$(k); advance $i of n=$n")
        end
    end

    A
end