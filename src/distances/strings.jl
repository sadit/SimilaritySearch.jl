# This file is a part of SimilaritySearch.jl

export CommonPrefixDissimilarity, GenericLevenshteinDistance, StringHammingDistance, LevenshteinDistance, LcsDistance

"""
    CommonPrefixDissimilarity()

Uses the common prefix as a measure of dissimilarity between two strings
"""
struct CommonPrefixDissimilarity <: SemiMetric
end

"""
    StringHammingDistance()

The hamming distance counts the differences between two equally sized strings
"""
struct StringHammingDistance <: SemiMetric
end

"""
    GenericLevenshteinDistance(;icost, dcost, rcost)

The levenshtein distance measures the minimum number of edit operations to convert one string into another.
The costs insertion `icost`, deletion cost `dcost`, and replace cost `rcost`. Not thread safe, use a copy of for each thread.
"""
@with_kw struct GenericLevenshteinDistance <: SemiMetric
    icost::Int32 = 1 # insertion cost
    dcost::Int32 = 1 # deletion cost
    rcost::Int32 = 1 # replace cost
    C::Vector{Int16} = Vector{Int16}(undef, 64)
end

"""
    LevenshteinDistance()

Instantiates a GenericLevenshteinDistance object to perform traditional levenshtein distance
"""
LevenshteinDistance() = GenericLevenshteinDistance()

"""
    LcsDistance()
 
Instantiates a GenericLevenshteinDistance object to perform LCS distance
"""
LcsDistance() = GenericLevenshteinDistance(rcost=2)

"""
    common_prefix(a, b)

Computes the length of the common prefix among two strings represented as arrays
"""
function common_prefix(a, b)
    len_a = length(a)
    len_b = length(b)
    i = 1
    min_len = min(len_a, len_b)
    @inbounds while i <= min_len && a[i] == b[i]
    	i += 1
    end

    i - 1
end


"""
    evaluate(::CommonPrefixDissimilarity, a, b)

Computes a dissimilarity based on the common prefix between two strings
"""
evaluate(::CommonPrefixDissimilarity, a, b) = 1.0 - common_prefix(a, b) / min(length(a), length(b))


"""
    evaluate(::GenericLevenshteinDistance, a, b)

Computes the edit distance between two strings, this is a low level function
"""
function evaluate(lev::GenericLevenshteinDistance, a, b)
    if length(a) < length(b)
        a, b = b, a
    end

    alen = length(a)
    blen = length(b)

    alen == 0 && return blen
    blen == 0 && return alen

    C = lev.C
    resize!(C, blen+1)
    @inbounds for i in 0:blen
        C[i+1] = i
    end

    prevA = 0
    @inbounds for i in 1:alen
        prevA = i
        prevC = C[1]
        j = 1

        while j <= blen
            cost = a[i] == b[j] ? 0 : lev.rcost
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
    d = 0

    @inbounds for i = 1:length(a)
        d += Int(a[i] != b[i])
    end

    d
end


# function kerrormatch(a::T1, b::T2, errors::Integer)::Bool where {T1 <: Any,T2 <: Any}
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
