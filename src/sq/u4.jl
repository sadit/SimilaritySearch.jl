export SQu4, SQu4Vec, SQu4L1, SQu4L2, SQu4SqL2


function quant_u4!(vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    m = length(v)
    k = 1
    j = 1
    @inbounds while j <= m
        a = round((v[j] - min) * c; digits=0)
        a = UInt8(clamp(a, 0, 15))
        b = zero(UInt8)
        if j+1 <= m
            b = let b = round((v[j+1] - min) * c; digits=0)
                UInt8(clamp(b, 0, 15))
            end
        end
    
        vout[k] = a | (b << 4)
        j += 2
        k += 1
    end

    vout    
end

function quant_u4!(vout::AbstractVector{UInt8}, v::AbstractVector; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    c = (max - min + eps) / 15f0
    quant_u4!(vout, v, min, 1f0/c)
    SQMinC(min, c)
end

struct SQu4Vec{VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
end

function SQu4Vec(v::AbstractVector)
    vout = Vector{UInt8}(undef, ceil(Int, length(v) / 2))
    minc = quant_u4!(vout, v)
    SQu4Vec(minc, vout)
end

Base.@propagate_inbounds function Base.getindex(qvec::SQu4Vec, i::Integer)::Float32
    if isodd(i)
        i = (i + 1) >> 1
        val = qvec.V[i] & UInt8(0x0f)
    else
        i >>= 1
        val = qvec.V[i] >> 4
    end

    Float32(val) * qvec.E.c + qvec.E.min
end

Base.length(a::SQu4Vec) = 2length(a.V)
Base.eachindex(a::SQu4Vec) = 1:2length(a.V)

function Base.eachindex(a::SQu4Vec, b::SQu4Vec)
    @assert length(a) === length(b)
    eachindex(a.V)
end

Base.eltype(::SQu4Vec) = Float32
Base.eltype(::Type{T}) where {T<:SQu4Vec} = Float32

struct SQu4 <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}

    function SQu4(X::AbstractMatrix)
        m, n = size(X)
        Q = Matrix{UInt8}(undef, ceil(Int, m / 2), n)
        E = Vector{SQMinC}(undef, n)
        @batch per=thread minbatch=4 for i in 1:n
            E[i] = quant_u4!(view(Q, :, i), view(X, :, i))
        end

        new(E, Q)
    end
end

Base.eltype(Q::SQu4) = typeof(Q[1])
Base.length(Q::SQu4) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu4, i::Integer) 
   SQu4Vec(Q.E[i], view(Q.Q, :, i))
end


### distances

"""
    SQu4L1()

"""
struct SQu4L1 <: Metric end

@inline function evaluate(::SQu4L1, A::SQu4Vec, B::SQu4Vec)::Float32
    d = zero(Float32)
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        af = Float32(a & 0x0f) * A.E.c + A.E.min
        bf = Float32(b & 0x0f) * B.E.c + B.E.min 
        m = abs(af - bf)
        a >>= 4; b >>= 4
        af = Float32(a) * A.E.c + A.E.min
        bf = Float32(b) * B.E.c + B.E.min
        m += abs(af - bf)
        d += m
    end

    d
end

function squared_euclidean(A::SQu4Vec, B::SQu4Vec)::Float32
    d = zero(Float32)    
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        af = Float32(a & 0x0f) * A.E.c + A.E.min
        bf = Float32(b & 0x0f) * B.E.c + B.E.min 
        m = (af - bf)^2
        a >>= 4; b >>= 4
        af = Float32(a) * A.E.c + A.E.min
        bf = Float32(b) * B.E.c + B.E.min
        m += (af - bf)^2
        d += m
    end

    d
end

function squared_euclidean(A::SQu4Vec, B)::Float32
    d = zero(Float32)
    n = length(A.V)
    odd = isodd(length(A))
    if odd
        n -= 1
    end

    @inbounds @simd for i in 1:n
        a = A.V[i]
        j = (i+1)>>1
        af = Float32(a & 0x0f) * A.E.c + A.E.min
        bf = B[j]
        m = (af - bf)^2
        a >>= 4
        af = Float32(a) * A.E.c + A.E.min
        bf = B[j+1]
        m += (af - bf)^2
        d += m
    end

    if odd
        i = n + 1
        a = A.V[i]
        j = (i+1)>>1
        af = Float32(a & 0x0f) * A.E.c + A.E.min
        bf = B[j]
        m = (af - bf)^2
        d += m
    end

    d
end

squared_euclidean(a, b::SQu4Vec) = squared_euclidean(b, a)

"""
    SQu4L2()

"""
struct SQu4L2 <: Metric end

@inline evaluate(::SQu4L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SQu4SqL2()

"""
struct SQu4SqL2 <: Metric end

@inline evaluate(::SQu4SqL2, a, b)::Float32 = squared_euclidean(a, b)
