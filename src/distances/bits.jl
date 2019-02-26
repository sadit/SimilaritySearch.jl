export hamming_distance, setbit, resetbit

# we export a few bit-handling primitives
# BitArray contains too much extra functionality at the cost of O(1) extra words,
# however this could be an issue, since we represent our database as n vectors


function hamming_distance(a::T, b::T)::Float64 where {T <: Unsigned}
    count_ones(a ⊻ b)
end

function hamming_distance(a::AbstractVector{T}, b::AbstractVector{T})::Float64 where {T <: Unsigned}
    d = 0
    @inbounds @simd for i in eachindex(a)
        d += count_ones(a[i] ⊻ b[i])
    end

    return d
end

const ONE8 = UInt8(1)
const ONE16 = UInt16(1)
const ONE32 = UInt32(1)
const ONE64 = UInt64(1)
const ONE128 = UInt128(1)

for (itype, one) in ((:UInt8, ONE8),
                     (:UInt16, ONE16),
                     (:UInt32, ONE32),
                     (:UInt64, ONE64),
                     (:UInt128, ONE128))
    @eval begin
        function setbit(a::$itype, i::Int)
            return a | ($one << i)
        end

        function resetbit(a::$itype, i::Int)
            return a & ~($one << i)
        end
    end
end
