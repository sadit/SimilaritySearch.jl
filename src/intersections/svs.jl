# This file is part of Intersections.jl
export svs

"""
    svs(postinglists, intersect2=baezayates!) -> output

Computes the intersection of the ordered lists in `postinglists` using a
small vs small strategy. Accepts an intersection algorithm of two sets.

This method does not give explicit support for `onmatch2!`

"""
function svs_!(curr, prev, postinglists, intersect2::Function=baezayates!)
    sort!(postinglists, by=length, rev=true)
    isize = intersect2(curr, pop!(postinglists), pop!(postinglists))
    isize == 0 && return curr
    sizehint!(prev, length(curr)) 

    while length(postinglists) > 0
        empty!(prev)
        isize = intersect2(prev, curr, pop!(postinglists))
        isize == 0 && return prev
        prev, curr = curr, prev
    end

    curr
end

function svs(postinglists::Vector{Vector{T}}, intersect2::Function=baezayates!) where T
    curr = T[]
    prev = T[]

    svs_!(curr, prev, postinglists, intersect2)
end
