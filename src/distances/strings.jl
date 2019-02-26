export common_prefix_distance, generic_levenshtein, hamming_distance, levenshtein_distance, lcs_distance

"""
sim_common_prefix computes the length of the common prefix among
two strings represented as arrays
"""
function sim_common_prefix(a::T, b::T)::Int where {T <: Any}
    len_a::Int = length(a)
    len_b::Int = length(b)
    i::Int = 1
    min_len::Int = min(len_a, len_b)
    @inbounds while i <= min_len && a[i] == b[i]
    	i += 1
    end

    return i - 1
end

mutable struct CommonPrefixDistance
	calls::Int
	CommonPrefixDistance() = new(0)
end

function common_prefix_distance(a::T, b::T) where T
	return sim_common_prefix(a, b) / max(length(a), length(b))
end

"""
generic_levenshtein computes the edit distance between two strings,
this is a low level function. Please use dist_lev
"""
function generic_levenshtein(a::T, b::T, icost::Int, dcost::Int, rcost::Int)::Int where T
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

    return prevA
end


"""
levenstein_distance 
computes edit distance over two strings or arrays of comparable objects
"""
function levenshtein_distance(a::T, b::T)::Float64 where T
    generic_levenshtein(a, b, 1, 1, 1)
end

"""
lcs_distance 
computes lcs distance over associated to the longest common subsequence; it accepts two strings or arrays of comparable objects
"""
function lcs_distance(a::T, b::T)::Float64 where T
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
hamming_distance computes the hamming distance between two slices of integers
"""

function hamming_distance(a::T, b::T)::Float64 where T
    d::Int = 0

    @inbounds for i = 1:length(a)
        #if a[i] != b[i]
        #    d += 1
        #end
        d += Int(a[i] != b[i])
    end

    return d
end
