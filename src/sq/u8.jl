export SQu8, SQu8Vec

### note we need to avoid overflows in high dimensional vectors (i.e., accumulated squared differences like 127^2)

function quant_u8!(vout, v, min::Float32, c::Float32)
    # c = 255f0 / (max - min)
    for j in eachindex(v)
        x = round((v[j] - min) * c; digits=0)
        vout[j] = clamp(x, 0, 255)
    end

    vout
end

function quant_u8!(vout, v::AbstractVector; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    c = (max - min + eps) / 255f0
    quant_u8!(vout, v, min, 1f0/c)
    SQMinC(min, c)
end

struct SQu8Vec{VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
end

function SQu8Vec(v::AbstractVector)
    vout = Vector{UInt8}(undef, length(v))
    minc = quant_u8!(vout, v)
    SQu8Vec(minc, vout)
end

Base.@propagate_inbounds function Base.getindex(qvec::SQu8Vec, i::Integer)::Float32
    Float32(qvec.V[i]) * qvec.E.c + qvec.E.min
end

Base.eachindex(a::SQu8Vec) = eachindex(a.V)
Base.eachindex(a::SQu8Vec, b::SQu8Vec) = eachindex(a.V, b.V)
Base.eltype(::SQu8Vec) = Float32
Base.eltype(::Type{T}) where {T<:SQu8Vec} = Float32

struct SQu8 <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}

    function SQu8(X::AbstractMatrix)
        m, n = size(X)
        Q = Matrix{UInt8}(undef, m, n)
        E = Vector{SQMinC}(undef, n)
        for i in 1:n
            E[i] = quant_u8!(view(Q, :, i), view(X, :, i))
        end

        new(E, Q)
    end
end

Base.eltype(Q::SQu8) = typeof(Q[1])
Base.length(Q::SQu8) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu8, i::Integer) 
   SQu8Vec(Q.E[i], view(Q.Q, :, i))
end
