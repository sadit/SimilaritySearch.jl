# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export common_prefix_distance, generic_levenshtein, hamming_distance, levenshtein_distance, lcs_distance

"""
    common_prefix(a, b)

common_prefix computes the length of the common prefix among
two strings represented as arrays
"""
function common_prefix(a, b)::Int
    len_a::Int = length(a)
    len_b::Int = length(b)
    i::Int = 1
    min_len::Int = min(len_a, len_b)
    @inbounds while i <= min_len && a[i] == b[i]
    	i += 1
    end

    i - 1
end


function common_prefix_distance(a, b)
    p = min(length(a), length(b))
	1.0 - common_prefix(a, b) / p
end

"""
    generic_levenshtein(a, b, icost::Int, dcost::Int, rcost::Int)::Int

Computes the edit distance between two strings, this is a low level function. Please use dist_lev
"""
function generic_levenshtein(a, b, icost::Int, dcost::Int, rcost::Int)::Int
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
            cost::Int = rcost
            if a[i] == b[j]
               cost = 0
            end
            C[j] = prevA
            j += 1
            prevA = min(C[j]+dcost, prevA+icost, prevC+cost)
            prevC = C[j]
        end

        C[j] = prevA
    end

    prevA
end


"""
    levenshtein_distance(a, b)

computes edit distance over two strings
"""
function levenshtein_distance(a, b)
    generic_levenshtein(a, b, 1, 1, 1)
end

"""
    lcs_distance(a, b)
 
computes longest common subsequence distance
"""
function lcs_distance(a, b)
    generic_levenshtein(a, b, 1, 1, 2)
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

"""
     hamming_distance(a, b)
     
Computes the hamming distance between two sequences of the same length
"""

function hamming_distance(a, b)
    d::Int = 0

    @inbounds for i = 1:length(a)
        #if a[i] != b[i]
        #    d += 1
        #end
        d += Int(a[i] != b[i])
    end

    d
end
