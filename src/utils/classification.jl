# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import StatsBase: fit, predict
export KNN, fit, predict

"""
    predict(index::Index, y::AbstractVector, nclasses::Int, dist::Function, Q::AbstractVector, k::Int)

Classify each ``u \\in Q`` according to the most frequent class among its ``k`` nearest
neighbors. Please note that it expects ``index.db \\subset U``, ``Q \\subset U``, and `` dist(u, v) \\forall u, v \\in U``.

- `index` the index used to solve knn queries
- `y` vector of classes, integer values in `1:nclasses`, aligned to the dataset `index.db`
- `nclasses` number of classes
- `dist` distance function
- `Q` vector of items to be classified 
- `k` the number of neighbors to be used in the classification

"""
function predict(index::Index, y::AbstractVector, nclasses::Int, dist::Function, Q::AbstractVector, k::Int)
    m = length(Q)
    ypred = Vector{Int}(undef, m)
    freqs = Vector{Int}(undef, nclasses)
    res = KnnResult(k)
    
    for i in eachindex(Q)
        empty!(res)
        fill!(freqs, 0)
        search(index, dist, Q[i], res)
        if k == 1
            ypred[i] = y[first(res).objID]
        else
            for p in res
                freqs[y[p.objID]] += 1
            end

            l = findmax(freqs)[end]
            ypred[i] = l
        end
    end

    ypred
end

struct KNN
    index::Index
    y::Vector{Int}
    nclasses::Int
    k::Int
end

function fit(::Type{KNN}, index::Index, y::Vector{Int}, k::Integer=1)
    KNN(index, y, length(unique(y)), k)
end

function predict(knn::KNN, dist::Function, Q::AbstractVector, k::Integer=0)
    predict(knn.index, knn.y, knn.nclasses, dist, Q, k == 0 ? knn.k : k)
end
