# This file is a part of SimilaritySearch.jl

export neardup

"""
    neardup(idx::AbstractSearchIndex, X::AbstractVector, ϵ)

Find nearest duplicates in database `X` using the empty index `idx`. The algorithm iteratively try to index elements in `X`,
and items being near than `ϵ` to some element in `idx` will be ignored.

The function returns a named tuple `(idx, map, nn, dist)` where:
- `idx`: it is the index of the non duplicated elements
- `map`: a mapping from `1-|idx|` to its positions in `X`
- `nn`: an array where each element in ``x \\in X`` points to its covering element (previously indexed element `u` such that ``d(u, x_i) \\leq ϵ``)
- `dist`: an array of distance values to each covering element (correspond to each element in `nn`)

# Arguments
- `idx`: An empty index (i.e., a `SearchGraph`)
- `X`: The input dataset
- `ϵ`: Real value to cut

# Notes
- The index `idx` must support incremental construction with `push!`
- You can access the set of elements being 'ϵ'-non duplicates (the ``ϵ-net``) using `idx.db` or where `nn[i] == i`
"""
function neardup(idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real)
    res = KnnResult(3)  # should be 1, but index's setups work better on larger `k` values
    L = zeros(Int32, length(X))
    D = zeros(Float32, length(X))
    M = UInt32[1]
    L[1] = 1
    D[1] = 0.0
    push_item!(idx, X[1])
    @inbounds for i in 2:length(X)
        res = reuse!(res)
        x = X[i]
        search(idx, x, res)
        if length(res) == 0 || minimum(res) > ϵ
            push_item!(idx, x)
            L[i] = i
            D[i] = 0.0
            push!(M, i)
        else
            L[i] = M[argmin(res)]
            D[i] = minimum(res)

        end
    end

    (idx=idx, map=M, nn=L, dist=D)
end
