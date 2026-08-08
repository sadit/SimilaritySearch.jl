# This file is part of Intersections.jl

export bkt!

"""
    bkt!(output, L, [P,], [findpos::Function=doublingsearch]; t::Int=length(L)) -> num. matches

Barybay & Kenyon t-thresholds 

See [`umerge!`](@ref) if you need to change how matches are captured (`onmatch!` function).

"""
function bkt!(output, L::AbstractVector, findpos::Function=doublingsearch; t::Int=length(L))
    P = ones(Int, length(L))
    bkt!(output, L, P, findpos; t)
end

function bkt!(output, L::AbstractVector, P::AbstractVector, findpos::Function=doublingsearch; t::Int=length(L))
    usize = 0  # number of onmatch calls

    @inbounds while length(L) > 0
        _remove_empty!(L, P)
        n = length(L)
        (n == 0 || t > n) && break
        _sort!(L, P)
        s = L[1][P[1]]
        if t > 1
            e = L[t][P[t]]
            if s != e
                for i in 1:t-1
                    P[i] = findpos(L[i], e, P[i])
                end 

                continue 
            end
        end

        m = t
        while m < n 
            s == L[m+1][P[m+1]] || break
            m += 1
        end
       
        onmatch!(output, L, P, m)
        usize += 1

        for i in 1:m
            P[i] += 1
        end
        
        # show_list_state(L, P, "END t=$t")
    end

    usize
end

