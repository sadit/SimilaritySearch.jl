# This file is a part of SimilaritySearch.jl
#
"""
    approx_by_hints!(index::SearchGraph, q, hints, res, vstate)

Approximate the result using a set of hints (the set of identifiers (integers)) behints  `hints`
"""
function approx_by_hints!(index::SearchGraph, q, hints::T, res, vstate) where {T<:Union{AbstractVector,Tuple,Integer,Set}}
    for objID in hints
        enqueue_item!(index, q, database(index, objID), res, objID, vstate)
    end

    res
end

struct AdjacentStoredHints{DB<:AbstractDatabase}
    hints::DB
    map::Vector{Int32}
end

Base.length(A::AdjacentStoredHints) = length(A.hints)

function matrixhints(index::SearchGraph, ::Type{DBType}=MatrixDatabase) where {DBType<:AbstractDatabase}
    h = Vector{Int32}(index.hints)
    s = SubDatabase(database(index), h)
    @set index.hints = AdjacentStoredHints(DBType(s), h)
end

function approx_by_hints!(index::SearchGraph, q, h::AdjacentStoredHints, res, vstate)
    for (i, objID) in enumerate(h.map)
        enqueue_item!(index, q, h.hints[i], res, objID, vstate)
    end

    res
end

"""
    mutable struct RandomHints

Indicates that hints are a random sample of the dataset
"""
@kwdef mutable struct RandomHints <: Callback
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

"""
    mutable struct DisjointHints

Indicates that hints are a small disjoint (untouched neighbors) subsample 
"""
@kwdef mutable struct DisjointHints <: Callback
    logbase::Float32 = 1.1
end

function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::DisjointHints)
    n = length(index)
    m = ceil(Int, log(opt.logbase, n))
    empty!(index.hints)
    meansize = mean(length(neighbors(index.adj, i)) for i in 1:n)
    res = knnqueue(ctx, m)
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
@kwdef struct KDisjointHints <: Callback
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
    sort!(sample, by=i -> length(neighbors(index.adj, i)), rev=true)
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
    mutable struct EpsilonHints

Indicates that hints are a small set of objects having a minimal distance between them 
"""
mutable struct EpsilonHints <: Callback
    epsilon::Float32
    minepsilon::Float32
    quantile::Float32
    samplesize::Function
    maxsize::Function
end

EpsilonHints(; quantile=0.01, epsilon=0.0f0, minepsilon=1e-5, samplesize=sqrt, maxsize=x -> log(1.1, x)) =
    EpsilonHints(convert(Float32, epsilon),
        convert(Float32, minepsilon),
        convert(Float32, quantile),
        samplesize,
        maxsize)

function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::EpsilonHints)
    n = length(index)
    m = min(n, ceil(Int, opt.samplesize(n)))
    s = rand(1:n, m) |> unique! |> sort!

    sample = VectorDatabase(s)
    out = VectorDatabase(UInt32[])
    dist = DistanceWithIdentifiers(distance(index), database(index))
    E = ExhaustiveSearch(; dist, db=out)
    ϵ = opt.quantile <= 0.0 ? opt.epsilon : let
        D = distsample(dist, sample; samplesize=m)
        max(opt.minepsilon, quantile(D, opt.quantile))
    end

    neardup(E, getcontext(E), sample, ϵ)
    v = out.vecs # internals of VectorDatabase
    max_ = ceil(Int, opt.maxsize(n))
    if length(v) > max_
        shuffle!(v)
        resize!(v, max_)
    end

    resize!(index.hints, length(v))
    index.hints .= v
end

"""
    mutable struct KCentersHints

Indicates that hints are a small set of objects having a minimal distance between them 
"""
mutable struct KCentersHints <: Callback
    logbase::Float32
    powsample::Float32
    qdiscard::Float32
end

KCentersHints(; logbase=1.1, powsample=1.5, qdiscard=0.1) = KCentersHints(logbase, powsample, qdiscard)

function execute_callback(index::SearchGraph, ctx::SearchGraphContext, opt::KCentersHints)
    n = length(index)
    k = min(n ÷ 2, ceil(Int, log(opt.logbase, n))) + 1
    @assert n > k
    m = min(n, ceil(Int, k^opt.powsample))
    #m / n
    D = let s = rand(1:n, m) |> unique! #|> sort!
        degrees = neighbors_length.(Ref(index.adj), s)
        min_, max_ = quantile(degrees, [0.25, 0.95])
        s = [j for (i, j) in enumerate(s) if min_ <= degrees[i] <= max_]
        sort!(s)
        SubDatabase(database(index), s)
    end
    A = fft(distance(index), D, k; ctx.verbose)
    # @show A unique(A.nn)
    M = Dict(c => i for (i, c) in enumerate(A.centers))
    count = zeros(Int, length(M))
    for nn in A.nn
        count[M[nn]] += 1
    end
    x = quantile(count, opt.qdiscard)
    C = A.centers[count.>=x]

    verbose(ctx) && @info "KCentersHints: n=$n, m=$m, k=$k, numcenters=$(length(A.centers)), C=$(length(C))"
    resize!(index.hints, length(C))
    index.hints .= D.map[C]
end

