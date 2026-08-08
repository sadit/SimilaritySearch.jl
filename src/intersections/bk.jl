# This file is part of Intersections.jl

export bk!

"""
    bk!(output, L, P, findpos::Function=doublingsearch) -> int. size

Computes the intersection of a list of posting lists using the Barbay and Kenyon algorithm.

See [`umerge!`](@ref) if you need to change how matches are captured (`onmatch!` function).

"""
function bk!(output, L::AbstractVector, findpos::Function=doublingsearch)
    P = ones(Int, length(L))
    bk!(output, L, P, findpos)
end
 
function bk!(output, L, P, findpos::Function=doublingsearch)
    n = length(L)
    _max = L[1][1]
    c = 0
    isize = 0  # number of onmatch calls

    @inbounds while true
        for i in eachindex(P)
            P[i] = findpos(L[i], _max, P[i])
            P[i] > length(L[i]) && return isize
            pval = L[i][P[i]]
            if pval == _max
                c += 1
                if c == n
                    onmatch!(output, L, P, c)
                    isize += 1
                    c = 0
                    P[i] += 1
                    P[i] > length(L[i]) && return isize
                    _max = L[i][P[i]]
                end
            else
                c = 0
                _max = pval
            end
        end
    end

    isize
end

