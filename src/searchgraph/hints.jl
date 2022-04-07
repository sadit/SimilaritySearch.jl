# This file is a part of SimilaritySearch.jl

"""
    mutable struct RandomHints

Indicates that hints are a random sample of the dataset
"""
@with_kw mutable struct RandomHints <: Callback
    logbase::Float32 = 1.1
end

"""
    executed_callback(opt::RandomHints, index)

SearchGraph's callback for selecting hints at random
"""
function execute_callback(opt::RandomHints, index)
    n = length(index)
    m = ceil(Int, log(opt.logbase, n))
    empty!(index.hints)
    V = Set{Int}()

    for i in 1:m
        p = rand(1:n)
        if !(p in V)
            push!(V, p)
            push!(index.hints, p)
        end
    end
end

function callback__(opt::RandomHints, index)
    empty!(index.hints)
    push!(index.hints, 1)
end

"""
    mutable struct DisjointHints

Indicates that hints are a small disjoint (untouched neighbors) subsample 
"""
@with_kw mutable struct DisjointHints <: Callback
    logbase::Float32 = 1.1
end

function execute_callback(opt::DisjointHints, index)
    n = length(index)
    m = ceil(Int, log(opt.logbase, n))
    empty!(index.hints)
    meansize = mean(length(index.links[i]) for i in 1:n)
    res = KnnResult(m)
    for i in 1:n
        push!(res, i, abs(length(index.links[i]) - meansize))
    end

    V = Set{Int}()
    for i in res.id
        i in V && continue
        push!(index.hints, i)
        push!(V, i)
        union!(V, index.links[i])
    end
end
"""
    struct KDisjointHints

Indicates that hints are selected to have a disjoint neighborhood
"""
@with_kw struct KDisjointHints <: Callback
    logbase::Float32 = 1.1
    disjoint::Int32 = 3
    expansion::Int32 = 4
end

"""
    execute_callback(opt::KDisjointHints, index)

SearchGraph's callback for selecting hints at random
"""
function execute_callback(opt::KDisjointHints, index)
    n = length(index)
    m = ceil(Int, log(opt.logbase, length(index)))
    sample = unique(rand(1:n, opt.expansion * m))
    m = min(length(sample), m)
    sort!(sample, by=i->length(index.links[i]), rev=true)

    visited = Set{Int32}()
    empty!(index.hints)
    E = Pair{Int32,Int32}[]
    i = 1
    while length(index.hints) < m && i < length(sample)
        # p = rand(1:n)
        p = sample[i]
        i += 1
        p in visited && continue
        push!(index.hints, p)
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
    
    # @info "disjoint hints $m --> $(length(index.hints))"
end