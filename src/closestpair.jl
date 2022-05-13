# This file is a part of SimilaritySearch.jl

export closestpair

"""
    closestpair(idx::AbstractSearchContext; parallel=false, pools=getpools(idx))

Finds the closest pair among all elements in `idx`. If the index `idx` is approximate then pair of points could be also an approximation.

# Arguments:
- `idx`: the search structure that indexes the set of points

# Keyword Arguments:
- `parallel`: If true then the algorithm uses all available threads to compute the closest pair
- `pools`: The pools needed for the index. Only used for special cases, default values should work in most cases. See [`getpools`](@ref) for more information.
"""
function closestpair(idx::AbstractSearchContext; parallel=false, pools=getpools(idx))
    parallel ? parallel_closestpair(idx, Threads.nthreads(), pools) : sequential_closestpair(idx, pools)
end

function search_hint(idx::AbstractSearchContext, i::Integer, pools)
    res = getknnresult(2)
    search(idx, idx[i], res; pools)
    argmin(res) == i ? (argmax(res), maximum(res)) : (argmin(res), minimum(res))
end

function search_hint(G::SearchGraph, i::Integer, pools)
    res = getknnresult(2)
    search(G.search_algo, G, G[i], res, first(G.links[i]), pools)
    argmin(res) == i ? (argmax(res), maximum(res)) : (argmin(res), minimum(res))
end

function parallel_closestpair(idx::AbstractSearchContext, parallel_block, pools)
    n = length(idx)
    parallel_block = min(n, parallel_block)
    B = [(zero(Int32), zero(Int32), typemax(Float32)) for _ in 1:parallel_block]

    for block in Iterators.partition(1:n, parallel_block)
        Threads.@threads for i in 1:length(block)
            @inbounds objID = block[i]
            id_, d_ = search_hint(idx, objID, pools)
            @inbounds if d_ < last(B[i])
                B[i] = (objID, id_, d_)
            end
        end
    end

    _, i = findmin(last, B)
    B[i]
end

function sequential_closestpair(idx::AbstractSearchContext, pools)
    mindist = typemax(Float32)
    I = J = zero(Int32)

    for i in eachindex(idx)
        id_, d_ = search_hint(idx, i, pools)
        if d_ < mindist
            I, J, mindist = i, id_, d_
        end
    end

    (I, J, mindist)
end