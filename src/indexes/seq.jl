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
function search(index::Sequential, dist::Fun, q, res::KnnResult) where Fun
    db = index.db
    
    for i in eachindex(db)
        @inbounds d = dist(q, db[i])
        push!(res, i, d)
    end
    q
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