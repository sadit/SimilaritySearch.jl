# This file is a part of SimilaritySearch.jl

export CommonPrefix, Levenshtein, Hamming, LCS

"""
    CommonPrefix()

Uses the common prefix as a measure of dissimilarity between two strings
"""
struct CommonPrefix <: SemiMetric
end

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
    evaluate(::CommonPrefix, a, b)

Computes a dissimilarity based on the common prefix between two strings
"""
evaluate(::CommonPrefix, a, b) = 1.0 - common_prefix(a, b) / min(length(a), length(b))


"""
    Levenshtein(;icost, dcost, rcost)

The levenshtein distance measures the minimum number of edit operations to convert one string into another.
The costs insertion `icost`, deletion cost `dcost`, and replace cost `rcost`. Not thread safe, use a copy of for each thread.
"""
struct Levenshtein <: Metric
    icost::Int32 # insertion cost
    dcost::Int32 # deletion cost
    rcost::Int32 # replace cost

    Cpool::Vector{Vector{Int16}}
end

Levenshtein(; icost=1, dcost=1, rcost=1) =
    Levenshtein(icost, dcost, rcost, [Vector{Int16}(undef, 64) for i in 1:Threads.maxthreadid()])

"""
    evaluate(::Levenshtein, a, b)

Computes the edit distance between two strings, this is a low level function
"""
function evaluate(lev::Levenshtein, a, b)
    if length(a) < length(b)
        a, b = b, a
    end

    alen = length(a)
    blen = length(b)

    alen == 0 && return blen
    blen == 0 && return alen

    C = lev.Cpool[Threads.threadid()]
    resize!(C, blen + 1)
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
            prevA = min(C[j] + lev.dcost, prevA + lev.icost, prevC + cost)
            prevC = C[j]
        end

        C[j] = prevA
    end

    prevA
end


"""
    Hamming()

The hamming distance counts the differences between two equally sized strings
"""
struct Hamming <: Metric
end

"""
     evaluate(::Hamming, a, b)
     
Computes the hamming distance between two sequences of the same length
"""
function evaluate(::Hamming, a, b)
    d = 0

    @inbounds for i in 1:length(a)
        d += Int(a[i] != b[i])
    end

    d
end


"""
    LCS()
 
Instantiates a Levenshtein object to perform LCS distance
"""
struct LCS <: Metric
    lev::Levenshtein
    LCS() = new(Levenshtein(rcost=2))
end

@inline evaluate(lcs::LCS, a, b) = evaluate(lcs.lev, a, b)

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
