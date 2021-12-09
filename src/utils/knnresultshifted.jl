# This file is a part of SimilaritySearch.jl

export KnnResultShifted

"""
    KnnResultShifted(ksearch::Integer)

Creates a priority queue with fixed capacity (`ksearch`) representing a knn result set.
It starts with zero items and grows with [`push!(res, id, dist)`](@ref) calls until `ksearch`
size is reached. After this only the smallest items based on distance are preserved.
"""
mutable struct KnnResultShifted{IdType<:Integer,DistType<:Real} <: AbstractVector{Tuple{IdType,DistType}}
    id::Vector{IdType}
    dist::Vector{DistType}
    k::Int32  # number of neighbors
    shift::Int32 # shift position of the first element (to support popfirst! efficiently)
    
    function KnnResultShifted(id::I, dist::D, k::Integer) where {I,D}
        @assert k > 0
        new{eltype(I), eltype(D)}(id, dist, Int32(k), 0)
    end
end

function KnnResultShifted(k::Integer, F=Float32)
    res = KnnResultShifted(Int32[], F[], k)
    sizehint!(res.id, k)
    sizehint!(res.dist, k)
    res
end

function Base.copy(res::KnnResultShifted)
    compact!(res)
    KnnResultShifted(copy(res.id), copy(res.dist), res.k)
end


"""
    push!(res::KnnResultShifted, item::Pair)
    push!(res::KnnResultShifted, id::Integer, dist::Real)

Appends an item into the result set
"""
@inline function Base.push!(res::KnnResultShifted, id::Integer, dist::Real)
    if length(res) < maxlength(res)
        k = res.k
        if length(res.id) >= 2k-1
            compact!(res, 1)
            @inbounds res.id[end], res.dist[end] = id, dist
        else
            push!(res.id, id)
            push!(res.dist, dist)
        end
    
        fixorder!(res.shift, res.id, res.dist)
        return true
    end

    dist >= last(res.dist) && return false

    @inbounds res.id[end], res.dist[end] = id, dist
    fixorder!(res.shift, res.id, res.dist)
    true
end

function compact!(res::KnnResultShifted, resize_extra=0)
    if res.shift > 0
        n = length(res)
        j = res.shift
        @inbounds for i in 1:n
            j += 1
            res.id[i] = res.id[j]
            res.dist[i] = res.dist[j]
        end
        res.shift = 0
        resize!(res.id, n+resize_extra)
        resize!(res.dist, n+resize_extra)
    end

    res
end

@inline Base.push!(res::KnnResultShifted, p::Pair) = push!(res, p.first, p.second)


"""
    popfirst!(p::KnnResultShifted)

Removes and returns the nearest neeighboor pair from the pool, an O(length(p.pool)) operation
"""
@inline function Base.popfirst!(res::KnnResultShifted)
    res.shift += 1
    @inbounds res.id[res.shift] => res.dist[res.shift]
    #popfirst!(res.id), popfirst!(res.dist)
end

"""
    pop!(p)

Removes and returns the last item in the pool, it is an O(1) operation
"""
@inline function Base.pop!(res::KnnResultShifted)
    pop!(res.id) => pop!(res.dist)
end

@inline Base.keys(res::KnnResultShifted) = @view res.id[eachindex(res)]
@inline Base.values(res::KnnResultShifted) = @view res.dist[eachindex(res)]
@inline Base.maximum(res::KnnResultShifted) = last(res.dist)
@inline Base.minimum(res::KnnResultShifted) = @inbounds res.dist[firstindex(res)]
@inline Base.firstindex(res::KnnResultShifted) = 1+res.shift
@inline Base.lastindex(res::KnnResultShifted) = lastindex(res.id)
@inline Base.first(res::KnnResultShifted) = @inbounds res.id[firstindex(res)] => res.dist[firstindex(res)]
@inline Base.last(res::KnnResultShifted) = @inbounds res.id[lastindex(res)] => res.dist[lastindex(res)]
@inline Base.argmin(res::KnnResultShifted) = @inbounds res.id[firstindex(res)]
@inline Base.argmax(res::KnnResultShifted) = @inbounds res.id[lastindex(res)]
@inline Base.length(res::KnnResultShifted) = length(res.id) - res.shift
@inline Base.size(res::KnnResultShifted) = (length(res),)

"""
    maxlength(res::KnnResultShifted)

The maximum allowed cardinality (the k of knn)
"""
@inline maxlength(res::KnnResultShifted) = res.k

"""
    covrad(p::KnnResultShifted)

Returns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned
"""
@inline covrad(res::KnnResultShifted)::Float32 = length(res) < maxlength(res) ? typemax(Float32) : res.dist[end]

"""
    empty!(res::KnnResultShifted)
    empty!(res::KnnResultShifted, k::Integer)

Clears the content of the result pool. If k is given then the size of the pool is changed; the internal buffer is adjusted
as needed (only grows).
"""
@inline function Base.empty!(res::KnnResultShifted, k::Integer=res.k)
    @assert k > 0
    empty!(res.id)
    empty!(res.dist)
    res.k = k
    res.shift = 0
end

"""
    getindex(res::KnnResultShifted, i)

Access the i-th item in `res`
"""
@inline function Base.getindex(res::KnnResultShifted, i)
    i += res.shift
    @inbounds res.id[i] => res.dist[i]
end


"""
    eachindex(res::KnnResultShifted)

Iterator of valid item indexes in `res`
"""
@inline Base.eachindex(res::KnnResultShifted) = firstindex(res):lastindex(res)

##### iterator interface
### KnnResultShifted
"""
    Base.iterate(res::KnnResultShifted, state::Int=1)

Support for iteration
"""
function Base.iterate(res::KnnResultShifted, state::Int=1)
    n = length(res)
    if n == 0 || state > n
        nothing
    else
        @inbounds res[state], state + 1
    end
end

Base.eltype(res::KnnResultShifted{I,F}) where {I,F} = Pair{I,F}
Base.IndexStyle(::Type{<:KnnResultShifted}) = IndexLinear()