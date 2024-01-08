# This file is a part of SimilaritySearch.jl

export closestpair

"""
    closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; minbatch=0)

Finds the closest pair among all elements in `idx`. If the index `idx` is approximate then pair of points could be also an approximation.

# Arguments:
- `idx`: the search structure that indexes the set of points
- `ctx`: the search context (caches, hyperparameters, etc)

# Keyword Arguments:
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
"""
function closestpair(idx::AbstractSearchIndex, ctx::AbstractContext; minbatch=0)
    if Threads.nthreads() == 1 || minbatch < 0 
        sequential_closestpair(idx, ctx)
    else
        parallel_closestpair(idx, ctx, minbatch)
    end
end

function search_hint(idx::AbstractSearchIndex, ctx::AbstractContext, i::Integer)
    res = getknnresult(2, ctx)
    search(idx, ctx, database(idx, i), res)
    argmin(res) == i ? res[2] : res[1]
end

function search_hint(G::SearchGraph, ctx::SearchGraphContext, i::Integer)
    res = getknnresult(8, ctx)
    vstate = getvstate(length(G), ctx)
    visit!(vstate, convert(UInt64, i))
    search(G.search_algo, G, ctx, database(G, i), res, rand(neighbors(G.adj, i)))
    argmin(res) == i ? res[2] : res[1]
end

function parallel_closestpair(idx::AbstractSearchIndex, ctx, minbatch)::Tuple{Int32,Int32,Float32}
    n = length(idx)
    B = [(zero(Int32), zero(Int32), typemax(Float32)) for _ in 1:Threads.nthreads()]

    minbatch = getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for objID in 1:n
        p = search_hint(idx, ctx, objID)
        tID = Threads.threadid()
        @inbounds if p.weight < last(B[tID])
            B[tID] = (Int32(objID), p.id, p.weight)
        end
    end

    _, i = findmin(last, B)
    B[i]
end

function sequential_closestpair(idx::AbstractSearchIndex, ctx)::Tuple{Int32,Int32,Float32}
    mindist = typemax(Float32)
    I = J = zero(Int32)

    for i in eachindex(idx)
        p = search_hint(idx, ctx, i)
        if p.weight < mindist
            I, J, mindist = Int32(i), p.id, p.weight
        end
    end

    (I, J, mindist)
end
