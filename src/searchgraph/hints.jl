# This file is a part of SimilaritySearch.jl

"""
    mutable struct RandomHints

Indicates that hints are a random sample of the dataset
"""
@with_kw mutable struct RandomHints <: Callback
    logbase::Float32 = 1.1
end

"""
    executed_callback(index::SearchGraph, ctx::SearchGraphContext, opt::RandomHints)

SearchGraph's callback for selecting hints at random
"""
function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::RandomHints)
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

function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::DisjointHints)
    n = length(index)
    m = ceil(Int, log(opt.logbase, n))
    empty!(index.hints)
    meansize = mean(length(neighbors(index.adj, i)) for i in 1:n)
    res = KnnResult(m)
    for i in 1:n
        push_item!(res, i, abs(length(neighbors(index.adj, i)) - meansize))
    end

    V = Set{Int}()
    for item in res
        i = item.id
        i in V && continue
        push!(index.hints, i)
        push!(V, i)
        union!(V, neighbors(index.adj, i))
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
    execute_callback(index, ctx, opt::KDisjointHints)

SearchGraph's callback for selecting hints at random
"""
function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::KDisjointHints)
    n = length(index)
    m = ceil(Int, log(opt.logbase, length(index)))
    sample = unique(rand(UInt32(1):UInt32(n), opt.expansion * m))
    m = min(length(sample), m)
    sort!(sample, by=i->length(neighbors(index.adj, i)), rev=true)
    IType = eltype(index.hints)
    visited = Set{IType}()
    empty!(index.hints)
    E = Pair{IType,Int32}[]
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
            for child in keys(neighbors(index.adj, parent))
                if !(child in visited)
                    push!(visited, child)
                    e + 1 <= opt.expansion && push!(E, child => e + 1)
                end
            end
        end
    end
end

"""
    mutable struct EpsilonCentersHints

Indicates that hints are a small set of objects having a minimal distance between them 
"""
mutable struct EpsilonCentersHints <: Callback
    epsilon::Float32
    samplesize::Function

    EpsilonCentersHints(epsilon::Number, samplesize=sqrt) = new(convert(Float32, epsilon), samplesize)
end

function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::EpsilonCentersHints)
    n = length(index)
    m = min(n, ceil(Int, opt.samplesize(n)))
    s = rand(1:n, m) |> unique! |> sort!

    sample = VectorDatabase(s)
    out = VectorDatabase(UInt32[])
    dist = DistanceWithIdentifiers(distance(index), database(index))
    E = ExhaustiveSearch(; dist, db=out)
    ϵ = opt.epsilon > 0.0 ? opt.epsilon : let
        D = distsample(dist, sample; m)
        quantile(D, abs(ϵ))
    end

    neardup(E, getcontext(E), sample, ϵ)
    resize!(index.hints, length(out))
    index.hints .= out.vecs
end

