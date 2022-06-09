# This file is a part of SimilaritySearch.jl

export closestpair

"""
    closestpair(idx::AbstractSearchContext; minbatch=0, pools=getpools(idx))

Finds the closest pair among all elements in `idx`. If the index `idx` is approximate then pair of points could be also an approximation.

# Arguments:
- `idx`: the search structure that indexes the set of points

# Keyword Arguments:
- `minbatch`: controls how multithreading is used for evaluating configurations, see [`getminbatch`](@ref)
- `pools`: The pools needed for the index. Only used for special cases, default values should work in most cases. See [`getpools`](@ref) for more information.
"""
function closestpair(idx::AbstractSearchContext; minbatch=0, pools=getpools(idx))
    if Threads.nthreads() == 1 || minbatch < 0 
        sequential_closestpair(idx, pools)
    else
        parallel_closestpair(idx, pools, minbatch)
    end
end

function search_hint(idx::AbstractSearchContext, i::Integer, pools)
    res = getknnresult(2, pools)
    search(idx, idx[i], res; pools)
    argmin(res) == i ? (argmax(res), maximum(res)) : (argmin(res), minimum(res))
end

function search_hint(G::SearchGraph, i::Integer, pools)
    res = getknnresult(3, pools)
    vstate = getvstate(length(G), pools)
    visit!(vstate, i)
    search(G.search_algo, G, G[i], res, rand(G.links[i]), pools; vstate)
    argmin(res), minimum(res)
end

function parallel_closestpair(idx::AbstractSearchContext, pools, minbatch)::Tuple{Int32,Int32,Float32}
    n = length(idx)
    B = [(zero(Int32), zero(Int32), typemax(Float32)) for _ in 1:Threads.nthreads()]

    minbatch = getminibatch(minbatch, n)

    @batch minbatch=minbatch per=thread for objID in 1:n
        id_, d_ = search_hint(idx, objID, pools)
        tID = Threads.threadid()
        @inbounds if d_ < last(B[tID])
            B[tID] = (Int32(objID), id_, d_)
        end
    end

    _, i = findmin(last, B)
    B[i]
end

function sequential_closestpair(idx::AbstractSearchContext, pools)::Tuple{Int32,Int32,Float32}
    mindist = typemax(Float32)
    I = J = zero(Int32)

    for i in eachindex(idx)
        id_, d_ = search_hint(idx, i, pools)
        if d_ < mindist
            I, J, mindist = Int32(i), id_, d_
        end
    end

    (I, J, mindist)
end