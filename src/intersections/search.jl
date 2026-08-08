# This file is part of Intersections.jl

export binarysearch, doublingsearch, doublingsearchrev, seqsearch, seqsearchrev

"""
 	binarysearch(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A` in the range `sp:ep`
"""
@inline function binarysearch(A, x, sp::Integer = 1, ep::Integer = length(A))::Int
    sp_i = Int(sp)
    hi = Int(ep) + 1
    @inbounds while sp_i < hi
        mid = (sp_i + hi) >>> 1
        if A[mid] < x
            sp_i = mid + 1
        else
            hi = mid
        end
    end

    sp_i
end

"""
 	doublingsearch(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A`, starting at `sp`
"""
@inline function doublingsearch(A, x, sp::Integer = 1, ep::Integer = length(A))::Int
    sp_i = Int(sp)
    ep_i = Int(ep)
    (sp_i > ep_i || @inbounds A[sp_i] >= x) && return sp_i

    step = 1
    hi = sp_i + step
    @inbounds while hi <= ep_i && A[hi] < x
        sp_i = hi
        step += step
        hi = sp_i + step
    end

    binarysearch(A, x, sp_i + 1, min(ep_i, hi))
end


"""
 	doublingsearchrev(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A`, starting at the end
"""
@inline function doublingsearchrev(A, x, sp::Integer = 1, ep::Integer = length(A))::Int
    sp_i = Int(sp)
    ep_i = Int(ep)
    (sp_i > ep_i || @inbounds x > A[ep_i]) && return ep_i + 1

    step = 1
    lo = ep_i - step
    @inbounds while lo >= sp_i && x <= A[lo]
        ep_i = lo
        step += step
        lo = ep_i - step
    end

    binarysearch(A, x, max(sp_i, lo + 1), ep_i)
end

"""
 	seqsearchrev(A, x, sp=1, ep=length(A))

Reverse sequential search, i.e., it starts from `ep` to `sp`
"""
@inline function seqsearchrev(A, x, sp::Integer = 1, ep::Integer = length(A))::Int
    sp_i = Int(sp)
    ep_i = Int(ep)
    pos = ep_i + 1
    @inbounds while pos > sp_i && x <= A[pos - 1]
        pos -= 1
    end

    pos
end

"""
 	seqsearch(A, x, sp=1, ep=length(A))

Sequential search, i.e., it starts from `sp` to `ep`
"""
@inline function seqsearch(A, x, sp::Integer = 1, ep::Integer = length(A))::Int
    sp_i = Int(sp)
    ep_i = Int(ep)
    @inbounds while sp_i <= ep_i && A[sp_i] < x
        sp_i += 1
    end

    sp_i
end

