# This file is part of Intersections.jl
export imerge2!

"""
    imerge2!(output, A, B) -> num. matches

Computes the intersection of `A` and `B` (sorted arrays), using a merge-like algorithm, and stores it in `output` or calls `onmatch(A, i, B, j)` on every match.
See [`bk!`](@ref) for how change the behaviour of matches.
"""
function imerge2!(output, A, B)
    i = j = 1
    n, m = length(A), length(B)
    count = 0
    @inbounds while i <= n && j <= m
        a = A[i]
        b = B[j]
        c = cmp(a, b)
        if c == 0
            onmatch2!(output, A, i, B, j)
            count += 1
            i += 1
            j += 1
        elseif c < 0
            i += 1
        else
            j += 1
        end
    end

    count
end

