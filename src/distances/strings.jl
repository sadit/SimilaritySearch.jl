export sim_common_prefix, best_match_levenshtein, kerrormatch
export CommonPrefixDistance, HammingDistance, GenericLevenshtein, LevDistance, LcsDistance

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

function (o::CommonPrefixDistance)(a::T, b::T) where {T}
	o.calls += 1
	return sim_common_prefix(a, b) / max(length(a), length(b))
end

"""
dist_levenshtein computes the edit distance between two strings,
this is a low level function. Please use dist_lev
"""
mutable struct GenericLevenshtein
    calls::Int
    icost::Int
    dcost::Int
    rcost::Int
end

"""LevDistance computes the edit distance"""
LevDistance() = GenericLevenshtein(0, 1, 1, 1)

"""LCS computes the distance associated to the longest common subsequence"""
LcsDistance() = GenericLevenshtein(0, 1, 1, 2)

function (o::GenericLevenshtein)(a::T, b::T)::Float64 where {T <: Any}
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
            cost::Int = o.rcost
            if a[i] == b[j]
               cost = 0
            end
            C[j] = prevA
            j += 1
            prevA = min(C[j]+o.dcost, prevA+o.icost, prevC+cost)
            prevC = C[j]
        end
	
        C[j] = prevA
    end
    
    return prevA
end

function kerrormatch(a::T1, b::T2, errors::Int)::Bool where {T1 <: Any,T2 <: Any}
    # if length(a) < length(b)
    #     a, b = b, a
    # end

    alen::Int = length(a)
    blen::Int = length(b)

    alen == 0 && return alen == blen
    blen == 0 && return true

    C::Vector{Int} = Vector{Int}(0:blen)

    @inbounds for i in 1:alen
        prevA::Int = 0
        prevC::Int = C[1]
        j::Int = 1
            
        while j <= blen
            cost::Int = 1
            if a[i] == b[j]
                cost = 0
            end
            C[j] = prevA
            j += 1
            prevA = min(C[j]+1, prevA+1, prevC+cost)
            prevC = C[j]
	    end
	
        C[j] = prevA
        if prevA <= errors
            return true
        end
    end

    return false
end

function best_match_levenshtein(a::T1, b::T2)::Int where {T1 <: Any,T2 <: Any}
    # if length(a) < length(b)
    #     a, b = b, a
    # end

    alen::Int = length(a)
    blen::Int = length(b)

    alen == 0 && return blen
    blen == 0 && return alen

    C::Vector{Int} = 1:blen |> collect

    mindist = alen
    @inbounds for i in 1:alen
        prevA::Int = 0
        prevC::Int = C[1]
        j::Int = 1
            
        while j <= blen
            cost::Int = 1
            if a[i] == b[j]
                cost = 0
            end
            C[j] = prevA
            j += 1
            prevA = min(C[j]+1, prevA+1, prevC+cost)
            prevC = C[j]
        end
	
        C[j] = prevA
        if prevA < mindist
            mindist = prevA
        end
    end
    
    return mindist
end

"""
dist_hamming computes the hamming distance between two slices of integers
"""
mutable struct HammingDistance
    calls::Int
    HammingDistance() = new(0)
end

function (o::HammingDistance)(a::T, b::T)::Float64 where {T <: Any}
    o.calls += 1
    d::Int = 0

    @inbounds for i = 1:length(a)
        #if a[i] != b[i]
        #    d += 1
        #end
        d += Int(a[i] != b[i])
    end

    return d
end
