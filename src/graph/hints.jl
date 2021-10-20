# This file is a part of SimilaritySearch.jl

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
    expansion::Int32 = 2
end

"""
    callback(opt::DisjointNeighborhoodHints, index)

SearchGraph's callback for selecting hints at random
"""
function callback(opt::DisjointNeighborhoodHints, index)
    n = length(index)
    m = ceil(Int, log(opt.logbase, length(index)))
    visited = Set{Int}()
    empty!(index.search_algo.hints)
    E = Pair{Int32,Int32}[]
    for i in 1:m
        p = rand(1:n)
        p in visited && continue
        push!(index.search_algo.hints, p)
        push!(visited, p)
        # visit the neighborhood with some expansion factor
        push!(E, p => 0)
        while length(E) > 0
            parent, e = pop!(E)
            for child in keys(index.links[parent])
                if !(child in visited)
                    push!(visited, child)
                    e + 1 <= opt.expansion && push!(E, child => e + 1)
                end
            end
        end
    end
    
    @info "disjoint hints $m --> $(length(index.search_algo.hints))"
end