export AngleDistance, CosineDistance,  DenseCosine, sim_cos

mutable struct AngleDistance
    calls::Int
    AngleDistance() = new(0)
end

immutable DenseCosine{T <: Real}
    vec::Vector{T}
    invnorm::T
end

function DenseCosine{T}(vec::Vector{T})
    xnorm = zero(T)
    @fastmath @inbounds @simd for i in eachindex(vec)
        xnorm += vec[i]^2
    end
    DenseCosine(vec, 1/sqrt(xnorm))
end

function (o::AngleDistance){T <: Real}(a::DenseCosine{T}, b::DenseCosine{T})::Float64
    o.calls += 1
    m = max(-1, sim_cos(a, b))
    return acos(min(1, m))
end

mutable struct CosineDistance
    calls::Int
    CosineDistance() = new(0)
end

function (o::CosineDistance){T <: Real}(a::DenseCosine{T}, b::DenseCosine{T})::Float64
    o.calls += 1
    return -sim_cos(a, b) + 1
end

function sim_cos{T <: Real}(a::DenseCosine{T}, b::DenseCosine{T})::Float64
    sum::T = zero(T)
    avec = a.vec
    bvec = b.vec

    @fastmath @inbounds @simd for i in eachindex(avec)
        sum += avec[i] * bvec[i]
    end

    sum * a.invnorm * b.invnorm
end

function save(ostream, item::DenseCosine)
    write(ostream, length(item.vec) |> Int32)
    for x in item.vec
        write(ostream, x)
    end
    write(ostream, item.invnorm)
end

function load{T}(istream, ::Type{DenseCosine{T}})::DenseCosine{T}
    vec = Vector{T}(read(istream, Int32))
    @inbounds for i in 1:length(vec)
        vec[i] = read(istream, T)
    end
    invnorm = read(istream, T)
    DenseCosine(vec, invnorm)
end

function saves(item::DenseCosine)
    join([string(x) for x in item.vec], ' ')
end

function loads{T}(line::String, ::Type{DenseCosine{T}})::DenseCosine{T}
    vec = [parse(T, x) for x in split(line, ' ')]
    DenseCosine(vec)
end
