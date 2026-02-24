export SQu2, SQu2Vec, SQu2L1, SQu2L2, SQu2SqL2

function quant_u2!(vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    n = length(v)
    m, r = divrem(n, 4)

    @inbounds @simd for k in 1:m 
        j = ((k-1) << 2) + 1
        x = zero(UInt8)
        for i in 0:3
            a = round((Float32(v[j+i]) - min) * c; digits=0)
            a = UInt8(clamp(a, 0, 3))
            x = x | (a << 2i)
        end
        
        vout[k] = x
    end

    if r > 0
        x = zero(UInt8)
        j = n - r + 1
        k = m + 1
        
        for i in 0:r-1
            a = round((Float32(v[j+i]) - min) * c; digits=0)
            a = UInt8(clamp(a, 0, 3))
            x = x | (a << 2i)
        end
        
        vout[k] = x
    end

    vout    
end

function quant_u2!(vout::AbstractVector{UInt8}, v::AbstractVector; eps::Float32=1f-6)
    min, max = extrema(v)
    min, max = Float32(min), Float32(max)
    c = (max - min + eps) / 3f0
    quant_u2!(vout, v, min, 1f0/c)    
    SQMinC(min, c)
end

struct SQu2Vec{VEC<:AbstractVector{UInt8}}
    E::SQMinC
    V::VEC
end

function SQu2Vec(v::AbstractVector)
    vout = Vector{UInt8}(undef, ceil(Int, length(v) / 4))
    minc = quant_u2!(vout, v)
    SQu2Vec(minc, vout)
end

Base.@propagate_inbounds function Base.getindex(qvec::SQu2Vec, i::Integer)::Float32
    i = UInt32(i-1)
    b = (i >> 2) + 1
    p = i & 0x3
    val = (qvec.V[b] >> 2p) & 0x3
    Float32(val) * qvec.E.c + qvec.E.min
end

Base.length(a::SQu2Vec) = 4length(a.V)
Base.eachindex(a::SQu2Vec) = 1:4length(a.V)

function Base.eachindex(a::SQu2Vec, b::SQu2Vec)
    @assert length(a) === length(b)
    eachindex(a.V)
end

Base.eltype(::SQu2Vec) = Float32
Base.eltype(::Type{T}) where {T<:SQu2Vec} = Float32

struct SQu2 <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}

    function SQu2(X::AbstractMatrix)
        m, n = size(X)
        Q = Matrix{UInt8}(undef, ceil(Int, m / 4), n)
        E = Vector{SQMinC}(undef, n)
        @batch per=thread minbatch=4 for i in 1:n
            E[i] = quant_u2!(view(Q, :, i), view(X, :, i))
        end

        new(E, Q)
    end
end

Base.eltype(Q::SQu2) = typeof(Q[1])
Base.length(Q::SQu2) = size(Q.Q, 2)

Base.@propagate_inbounds function Base.getindex(Q::SQu2, i::Integer) 
   SQu2Vec(Q.E[i], view(Q.Q, :, i))
end


### distances

"""
    SQu2L1()

"""
struct SQu2L1 <: Metric end

@inline function evaluate(::SQu2L1, A::SQu2Vec, B::SQu2Vec)::Float32
    d = zero(Float32)    
    n = length(A.V)

    @inbounds @simd for i in 1:n
        a, b = A.V[i], B.V[i]
        m = zero(Float32)
        for p in 0:2:6
            af = Float32((a >> p) & 0x03) * A.E.c + A.E.min
            bf = Float32((b >> p) & 0x03) * B.E.c + B.E.min
            m += (af - bf)
        end

        d += m
    end

    d
end

function squared_euclidean(A::SQu2Vec, B::SQu2Vec)::Float32
    d = zero(Float32)    
    n = length(A.V)

    @inbounds @simd for i in 1:n
    #ii = 0
    #for i in 1:n
        a, b = A.V[i], B.V[i]
        m = zero(Float32)
        for p in 0:2:6
            #ii += 1
            af = Float32((a >> p) & 0x03) * A.E.c + A.E.min
            #a_ = A[ii]
            #@assert af ≈ a_ "-- ii: $ii, af: $af, a_: $a_, i=$i, p=$p"
            bf = Float32((b >> p) & 0x03) * B.E.c + B.E.min
            m += (af - bf)^2
        end

        d += m
    end

    d
end

function squared_euclidean(A::SQu2Vec, B)::Float32
    d = zero(Float32)    
    n = length(B)
    m, r = divrem(n, 4)

    @inbounds @simd for k in 1:m  # A index 
        j = ((k - 1) << 2) + 1    # B index (each 4)
        a = A.V[k]
        m = zero(Float32)
        for p in 0:3
            af = Float32((a >> 2p) & 0x03) * A.E.c + A.E.min
            bf = B[j+p]
            m += (af - bf)^2
        end

        d += m
    end
    
    if r > 0
        j = n - r + 1
        k = m + 1

        a = A.V[k]
        m = zero(Float32)
        for p in 0:3
            af = Float32((a >> 2p) & 0x03) * A.E.c + A.E.min
            bf = B[j+p]
            m += (af - bf)^2
        end

        d += m
    end

    d
end

squared_euclidean(a, b::SQu2Vec) = squared_euclidean(b, a)

"""
    SQu2L2()

"""
struct SQu2L2 <: Metric end

@inline evaluate(::SQu2L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SQu2SqL2()

"""
struct SQu2SqL2 <: Metric end

@inline evaluate(::SQu2SqL2, a, b)::Float32 = squared_euclidean(a, b)
