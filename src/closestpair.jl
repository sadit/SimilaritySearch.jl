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

function search_hint(idx::AbstractSearchIndex, ctx::AbstractContext, i::Integer, res)
    res = reuse!(res, 2)
    search(idx, ctx, database(idx, i), res)
    argmin(res) == i && pop_min!(res)
    nearest(res)
end

function search_hint(G::SearchGraph, ctx::SearchGraphContext, i::Integer, res)
    res = reuse!(res)
    vstate = getvstate(length(G), ctx)
    visit!(vstate, convert(UInt64, i))
    search(G.algo, G, ctx, database(G, i), res, rand(neighbors(G.adj, i)))
    argmin(res) == i && pop_min!(res)
    nearest(res)
end

function parallel_closestpair(idx::AbstractSearchIndex, ctx, minbatch; blocksize=Threads.nthreads())::Tuple{Int32,Int32,Float32}
    n = length(idx)
    minbatch = getminbatch(minbatch, n)
    B = [(zero(Int32), zero(Int32), typemax(Float32)) for _ in 1:Threads.nthreads()]
    R = xknnset(8, blocksize)

    @batch minbatch=minbatch per=thread for objID in 1:n
        tID = Threads.threadid()
        p = search_hint(idx, ctx, objID, R.knns[tID])
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
    res = xknn(8)
    for i in eachindex(idx)
        reuse!(res)
        p = search_hint(idx, ctx, i, res)
        if p.weight < mindist
            I, J, mindist = Int32(i), p.id, p.weight
        end
    end

    (I, J, mindist)
end
