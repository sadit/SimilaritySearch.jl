# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import Base: push!
import StatsBase: fit

export Sequential, search, push!, fit, optimize!

"""
    Sequential{T}

A simple exhaustive search index
"""
struct Sequential{T} <: Index
    db::Vector{T}
end

"""
    fit(::Type{Sequential}, db::Vector)

Creates a sequential-exhaustive index
"""
function fit(::Type{Sequential}, db::Vector)
    Sequential(db)
end

"""
    search(index::Sequential, dist, q, res::KnnResult)

Solves the query evaluating ``dist(q,u) \\forall u \\in index`` against 
"""
function search(index::Sequential, dist, q, res::KnnResult)
    i::Int32 = 1
    d::Float64 = 0.0
    for obj in index.db
        d = dist(q, obj)
        push!(res, i, convert(Float32, d))
        i += 1
    end

    res
end

"""
   push!(index::Sequential, dist, item)

Interts an item into the index
"""
function push!(index::Sequential, dist, item)
    push!(index.db, item)
    length(index.db)
end

"""
    optimize!(index::Sequential{T}, dist, recall::Float64)

Optimizes the index for the required quality
"""
function optimize!(index::Sequential{T}, dist, recall::Float64) where T
    # do nothing for sequential
end