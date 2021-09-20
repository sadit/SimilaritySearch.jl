# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

"""
    mutable struct RandomHintsCallback

Indicates that hints are a random sample of the dataset
"""
@with_kw mutable struct RandomHintsCallback <: Callback
    logbase::Float32 = 1.5
end

"""
    callback(opt::RandomHintsCallback, index)

SearchGraph's callback for selecting hints at random
"""
function callback(opt::RandomHintsCallback, index)
    n = length(index)
    m = ceil(Int, log(opt.logbase, length(index)))
    sample = unique(rand(1:n, m))
    empty!(index.search_algo.hints)
    append!(index.search_algo.hints, sample)
end

"""
    struct DisjointNeighborhoodHints

Indicates that hints are selected to have a disjoint neighborhood
"""
@with_kw struct DisjointNeighborhoodHints <: Callback
    logbase::Float32 = 1.5
    expansion::Int32 = 3
end

"""
    callback(opt::DisjointNeighborhoodHints, index)

SearchGraph's callback for selecting hints at random
"""
function callback(opt::DisjointNeighborhoodHints, index)
    n = length(index)
    m = ceil(Int, log(opt.logbase, length(index)))
    visited = Set{Int}()
    sample = Int32[]
    for i in 1:m
        p = rand(1:n)
        p in visited && continue
        push!(sample, p)
        push!(visited, p)
        for child in keys(index.links[i])
            push!(visited, child)
        end
    end
        
    empty!(index.search_algo.hints)
    append!(index.search_algo.hints, sample)
end