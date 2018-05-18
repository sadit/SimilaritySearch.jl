export AngleDistance, CosineDistance,  DenseCosine, sim_cos

mutable struct AngleDistance
    calls::Int
    AngleDistance() = new(0)
end

struct DenseCosine{T <: Real}
    vec::Vector{T}
    invnorm::T
end

function DenseCosine(vec::Vector{T}) where T
    xnorm = zero(T)
    @fastmath @inbounds @simd for i in eachindex(vec)
        xnorm += vec[i]^2
    end
    
    (xnorm <= eps(T)) && error("A valid DenseCosine object cannot have a zero norm $xnorm -- vec: $vec")
    DenseCosine(vec, 1/sqrt(xnorm))
end

function (o::AngleDistance)(a::DenseCosine{T}, b::DenseCosine{T})::Float64 where {T <: Real}
    o.calls += 1
    m = max(-1.0, sim_cos(a, b))
    acos(min(1.0, m))
end

function (o::AngleDistance)(a::AbstractVector{T}, b::DenseCosine{T})::Float64 where {T <: Real}
    o.calls += 1
    m = max(-1.0, sim_cos(DenseCosine(a), b))
    acos(min(1.0, m))
end

function (o::AngleDistance)(a::DenseCosine{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    o.calls += 1
    m = max(-1.0, sim_cos(a, DenseCosine(b)))
    acos(min(1.0, m))
end

mutable struct CosineDistance
    calls::Int
    CosineDistance() = new(0)
end

function (o::CosineDistance)(a::DenseCosine{T}, b::DenseCosine{T})::Float64 where {T <: Real}
    o.calls += 1
    return -sim_cos(a, b) + 1
end

function (o::CosineDistance)(a::AbstractVector{T}, b::DenseCosine{T})::Float64 where {T <: Real}
    o.calls += 1
    return -sim_cos(DenseCosine(a), b) + 1
end

function (o::CosineDistance)(a::DenseCosine{T}, b::AbstractVector{T})::Float64 where {T <: Real}
    o.calls += 1
    return -sim_cos(a, DenseCosine(b)) + 1
end

function sim_cos(a::DenseCosine{T}, b::DenseCosine{T})::Float64 where {T <: Real}
    sum::T = zero(T)
    avec = a.vec
    bvec = b.vec

    @fastmath @inbounds @simd for i in eachindex(avec)
        sum += avec[i] * bvec[i]
    end

    sum * a.invnorm * b.invnorm
end
