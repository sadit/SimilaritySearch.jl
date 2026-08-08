# This file is part of Intersections.jl
export baezayates!

"""
    baezayates!(output, A, B, findpos=binarysearch) -> num. matches
    baezayates!(output, A, a_sp, a_ep, B, b_sp, b_ep, findpos)

Computes the intersection between first and second ordered lists using the Baeza-Yates algorithm [cite].
The intersection is stored in `output`.

Please specialize `onmatch2!(output, A, i, B, j)` function if you need a particular behaviour on matches beyond push items into `output`.

"""

function baezayates!(output, A, B, findpos::Function = binarysearch)
    baezayates!(output, A, 1, length(A), B, 1, length(B), findpos)
end

function baezayates!(output, A, a_sp::Int, a_ep::Int, B, b_sp::Int, b_ep::Int, findpos::Function)
    (a_ep < a_sp || b_ep < b_sp) && return 0
    imedian = ceil(Int, (a_ep + a_sp) / 2)
    median = A[imedian]
    medpos = min(findpos(B, median, b_sp), b_ep) ## our findpos returns n + 1 when median is larger than B[end]
    matches = median === B[medpos]

    count = baezayates!(output, A, a_sp, imedian - 1, B, b_sp, medpos - matches, findpos)
    if matches
        count += 1
        onmatch2!(output, A, imedian, B, medpos)
    end

    count + baezayates!(output, A, imedian + 1, a_ep, B, medpos + matches, b_ep, findpos)
end

