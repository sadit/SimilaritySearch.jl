# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export CommonPrefixDissimilarity, GenericLevenshteinDistance, StringHammingDistance, LevenshteinDistance, LcsDistance

"""
    CommonPrefixDissimilarity()

Uses the common prefix as a measure of dissimilarity between two strings
"""
struct CommonPrefixDissimilarity <: PreMetric end

"""
    StringHammingDistance()

The hamming distance counts the differences between two equally sized strings
"""
struct StringHammingDistance <: PreMetric end

"""
    GenericLevenshteinDistance(icost, dcost, rcost)

The levenshtein distance measures the minimum number of edit operations to convert one string into another.
The costs insertion `icost`, deletion cost `dcost`, and replace cost `rcost`.
"""
struct GenericLevenshteinDistance <: PreMetric
    icost::Int # insertion cost
    dcost::Int # deletion cost
    rcost::Int # replace cost
end

StructTypes.StructType(::Type{CommonPrefixDissimilarity}) = StructTypes.Struct()
StructTypes.StructType(::Type{StringHammingDistance}) = StructTypes.Struct()
StructTypes.StructType(::Type{GenericLevenshteinDistance}) = StructTypes.Struct()


"""
    LevenshteinDistance(a, b)

Instantiates a GenericLevenshteinDistance object to perform traditional levenshtein distance
"""
LevenshteinDistance() = GenericLevenshteinDistance(1, 1, 1)

"""
    LcsDistance(a, b)
 
Instantiates a GenericLevenshteinDistance object to perform LCS distance
"""
LcsDistance() = GenericLevenshteinDistance(1, 1, 2)

"""
    common_prefix(a, b)

Computes the length of the common prefix among two strings represented as arrays
"""
function common_prefix(a, b)
    len_a::Int = length(a)
    len_b::Int = length(b)
    i::Int = 1
    min_len::Int = min(len_a, len_b)
    @inbounds while i <= min_len && a[i] == b[i]
    	i += 1
    end

    i - 1
end


raw"""
    evaluate(::CommonPrefixDissimilarity, a, b)

Computes a dissimilarity based on the common prefix between two strings

$$ 1 - common\_prefix(a-b) / \min\{|a|, |b|\} $$
"""
function evaluate(::CommonPrefixDissimilarity, a, b)
    p = min(length(a), length(b))
	1.0 - common_prefix(a, b) / p
end

"""
    evaluate(GenericLevenshteinDistance, a, b)::Int

Computes the edit distance between two strings, this is a low level function
"""
function evaluate(lev::GenericLevenshteinDistance, a, b)::Int
    if length(a) < length(b)
        a, b = b, a
    end

    alen::Int = length(a)
    blen::Int = length(b)

    alen == 0 && return blen
    blen == 0 && return alen

    C::Vector{Int} = Array(0:blen)

    prevA::Int = 0
    @inbounds for i in 1:alen
        prevA = i
        prevC::Int = C[1]
        j::Int = 1

        while j <= blen
            cost = lev.rcost
            if a[i] == b[j]
               cost = 0
            end
            C[j] = prevA
            j += 1
            prevA = min(C[j]+lev.dcost, prevA+lev.icost, prevC+cost)
            prevC = C[j]
        end

        C[j] = prevA
    end

    prevA
end



"""
     evaluate(::StringHammingDistance, a, b)
     
Computes the hamming distance between two sequences of the same length
"""
function evaluate(::StringHammingDistance, a, b)
    d::Int32 = 0

    @inbounds for i = 1:length(a)
        d += Int(a[i] != b[i])
    end

    d
end


# function kerrormatch(a::T1, b::T2, errors::Int)::Bool where {T1 <: Any,T2 <: Any}
#     # if length(a) < length(b)
#     #     a, b = b, a
#     # end

#     alen::Int = length(a)
#     blen::Int = length(b)

#     alen == 0 && return alen == blen
#     blen == 0 && return true

#     C::Vector{Int} = Vector{Int}(0:blen)

#     @inbounds for i in 1:alen
#         prevA::Int = 0
#         prevC::Int = C[1]
#         j::Int = 1

#         while j <= blen
#             cost::Int = 1
#             if a[i] == b[j]
#                 cost = 0
#             end
#             C[j] = prevA
#             j += 1
#             prevA = min(C[j]+1, prevA+1, prevC+cost)
#             prevC = C[j]
# 	    end

#         C[j] = prevA
#         if prevA <= errors
#             return true
#         end
#     end

#     return false
# end

# function best_match_levenshtein(a::T1, b::T2)::Int where {T1 <: Any,T2 <: Any}
#     # if length(a) < length(b)
#     #     a, b = b, a
#     # end

#     alen::Int = length(a)
#     blen::Int = length(b)

#     alen == 0 && return blen
#     blen == 0 && return alen

#     C::Vector{Int} = 1:blen |> collect

#     mindist = alen
#     @inbounds for i in 1:alen
#         prevA::Int = 0
#         prevC::Int = C[1]
#         j::Int = 1

#         while j <= blen
#             cost::Int = 1
#             if a[i] == b[j]
#                 cost = 0
#             end
#             C[j] = prevA
#             j += 1
#             prevA = min(C[j]+1, prevA+1, prevC+cost)
#             prevC = C[j]
#         end

#         C[j] = prevA
#         if prevA < mindist
#             mindist = prevA
#         end
#     end

#     return mindist
# end
