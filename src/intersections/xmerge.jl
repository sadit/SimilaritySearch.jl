# This file is part of Intersections.jl
export xmerge!

"""
    xmerge!(output, L, P=ones(Int32, length(L_)); t::Int=1) -> num. matches

Solves t-threshold set operation using other algorithms choosing among them by given `t`

# Arguments:
- `output`: vector like to store the t-threshold set
- `L`: the list of posting lists to be merged. The posting lists are left untouched but the container is modified.
- `P`: indices of the current merging-state (idem to `L`)
- `t`: the threshold, i.e., `t=1` (union) ... `t=|L|` performs intersection)

Simple wrapper around other specific operations depending on `t` value

See [`umerge!`](@ref) if you need to modify the output behaviour.
"""
function xmerge!(output, L::AbstractVector, P=ones(Int32, length(L)); t::Int=1)
    n = length(L)
    t = convert(Int, t == 0 ? n : (t < 0 ? n - t : t))
    t = min(t, n)

    if t == n  # intersection
        bk!(output, L, P)
    elseif t <= 2
        umerge!(output, L, P; t)
    else
        bkt!(output, L, P; t)
    end
end

