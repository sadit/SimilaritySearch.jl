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

"""
    SQu2Vec(v::AbstractVector)

A single vector quantized to 2 bits per coordinate. It stores the packed codes (four
2-bit codes per `UInt8`, `V`) along with the linear dequantization parameters (`E::SQMinC`)
computed from the extrema of `v`. Indexing a `SQu2Vec` (`qvec[i]`) unpacks and dequantizes
the `i`-th coordinate back to a `Float32` approximation of the original value.

This type is the element produced by indexing a [`SQu2`](@ref) database; it is normally
not created directly by users.

# Arguments
- `v`: the input vector to quantize
"""
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
    i = Int32(i-1)
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

"""
    SQu2(X::AbstractMatrix)

Scalar-quantizes each column (vector) of `X` to 2 bits per coordinate, packing four
codes into each `UInt8`. This reduces the memory footprint of a database of vectors by
roughly a factor of 16 with respect to `Float32` at the cost of precision. Each column
is quantized independently using its own minimum and scale factor, computed from the
extrema of the column so that the whole range `[min, max]` is mapped to the `\\{0,1,2,3\\}`
codes.

`SQu2` implements the `AbstractDatabase` interface, i.e., `length(db)` gives the number
of vectors and `db[i]` returns the `i`-th vector as a [`SQu2Vec`](@ref) that can be
indexed to retrieve dequantized `Float32` coordinates.

# Arguments
- `X`: a matrix whose columns are the vectors to be quantized

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> db = ScalarQuant.SQu2(X);

julia> db[1][1]  # dequantized approximation of X[1, 1]
```
"""
struct SQu2 <: AbstractDatabase
    E::Vector{SQMinC}
    Q::Matrix{UInt8}

    function SQu2(X::AbstractMatrix)
        m, n = size(X)
        Q = Matrix{UInt8}(undef, ceil(Int, m / 4), n)
        E = Vector{SQMinC}(undef, n)
        minbatch = getminbatch(n)
        @batch per=thread minbatch=minbatch for i in 1:n
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

A Manhattan-like (``L_1``) distance for [`SQu2Vec`](@ref) (2-bit quantized) vectors.
`evaluate` dequantizes both codes coordinate by coordinate and accumulates their
difference `af - bf`.

Note: unlike the general [`L1`](@ref) distance, this implementation does not take the
absolute value of the per-coordinate difference before accumulating, so the result is
not guaranteed to be non-negative; it should be understood as an approximation intended
for relative ranking of 2-bit quantized vectors rather than a true metric.
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

The Euclidean (``L_2``) distance between two 2-bit quantized vectors ([`SQu2Vec`](@ref)),
or between a [`SQu2Vec`](@ref) and a plain vector. `evaluate` dequantizes coordinate by
coordinate, accumulates the squared differences (see [`SQu2SqL2`](@ref)), and returns its
square root.
"""
struct SQu2L2 <: Metric end

@inline evaluate(::SQu2L2, a, b) = sqrt(squared_euclidean(a, b))

"""
    SQu2SqL2()

The squared Euclidean distance between two 2-bit quantized vectors ([`SQu2Vec`](@ref)),
or between a [`SQu2Vec`](@ref) and a plain vector. `evaluate` dequantizes coordinate by
coordinate and accumulates the squared differences `(af - bf)^2`, avoiding the
square root computed by [`SQu2L2`](@ref).
"""
struct SQu2SqL2 <: Metric end

@inline evaluate(::SQu2SqL2, a, b)::Float32 = squared_euclidean(a, b)
