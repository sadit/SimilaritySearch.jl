# This file is part of Intersections.jl
export umerge!

function show_list_state(L, P, nick)
    println(stderr, "====== $nick ===")
    println(stderr, "L: $L")
    println(stderr, "P: $P")
    for i in eachindex(L)
        if P[i] > length(L[i])
            println(stderr, "@ $i> beyond list frontier P[i] = $(P[i]), |L[i]| = $(length(L[i]))")
        else
            println(stderr, "@ $i> L[] = $(L[i][P[i]]) -- P[] = $(P[i])")
        end
    end
end

"""
    _remove_empty!(L, P)

Inplace removal of empty lists
"""
function _remove_empty!(L, P)
    j = 1
    n = length(P)
    @inbounds for i in 1:n
        # @show (i, j, P[i] <= length(L[i]), P[i], length(L[i]))
        if P[i] <= length(L[i])
            if i != j
                P[j] = P[i]
                L[j] = L[i]
            end
            j += 1
        end
    end
    # @show j n P L

    while length(P) >= j
        pop!(P); pop!(L)
    end
end

"""
    _sort!(L, P)

Adaptive bubble sort, efficient than other approaches because we expect a few sets and almost sorted
"""
function _sort!(L, P::Vector)
    s = 1
    n = length(P)

    while s > 0
        s = 0
        for i in 1:n-1
            @inbounds if L[i][P[i]] > L[i+1][P[i+1]]
                swapitems!(P, i, i+1)
                swapitems!(L, i, i+1)
                s += 1
            end
        end
    end
end

"""
     umerge!(output, L, P=ones(Int32, length(L_)); t::Int=1) -> num. matches

Merges posting lists in `L` and saves the union in `output`.
The merge result is stored into output array. You can customize how to do this
specializing the `onmatch!(output, L, P, t::Int)` function.

# Arguments:
- `L`: The array of posting lists, the array can be destroyed in the process.
- `P`: The array of current positions in posting lists, i.e., initial state as an array of ones of size ``|L|``.

- `t`: Computes t-thresholds, i.e., t from 1 (union) to |L| (intersection) of posting lists in `L` using `findpos` storing the result set in `output`.

# About the callback function
`output`, `L` and `P` are the arguments same than the input, while `t` is the actual number of lists having the match.
Note 1: you should access `L[i][P[i]]` to get the entry of the `i`th list, i.e., ``1 \\leq t \\leq |P|``.
Note 2: `L` and `P` as container lists will be also modified, the contained lists remain untouched.

"""
function umerge!(output, L, P = ones(Int32, length(L)); t::Int=1)
    _sort!(L, P)  # sort!(L, by=first)
    usize = 0

    @inbounds while true
        _remove_empty!(L, P)
        n = length(L)
        (n == 0 || t > n) && break
        _sort!(L, P)
        s = L[1][P[1]]

        if t > 1
            e = L[t][P[t]]

            if s != e
                for i in 1:t-1
                    P[i] += 1
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
        for i in 1:m
            P[i] += 1
        end

        usize += 1
    end

    usize
end

