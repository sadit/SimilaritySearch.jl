# This file is a part of SimilaritySearch.jl

export neardup


"""
    neardup(push_fun::Function, idx::AbstractSearchIndex, X::AbstractVector, ϵ)
    neardup(idx::AbstractSearchIndex, X::AbstractVector, ϵ)

Find nearest duplicates in database `X` using the empty index `idx`. The algorithm iteratively try to index elements in `X`,
and items being near than `ϵ` to some element in `idx` will be ignored.

The function returns a named tuple `(idx, map, nn, dist)` where:
- `idx`: it is the index of the non duplicated elements
- `map`: a mapping from `|idx|-1` to its positions in `X`
- `nn`: an array where each element in ``x \\in X`` points to its covering element (previously indexed element `u` such that ``d(u, x_i) \\leq ϵ``)
- `dist`: an array of distance values to each covering element (correspond to each element in `nn`)

`push_fun` argument can be used to customize object insertions (e.g., set `SearchGraphCallbacks` for `SearchGraph`)

# Arguments
- `idx`: An empty index (i.e., a `SearchGraph`)
- `X`: The input dataset
- `ϵ`: Real value to cut

# Notes
- The index `idx` must support incremental construction, e.g., with a valid `push_item!` implementation
- You can access the set of elements being 'ϵ'-non duplicates (the ``ϵ-net``) using `idx.db` or where `nn[i] == i`
"""
function neardup(idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real; kwargs...)
    neardup(push_item!, idx, X, ϵ; kwargs...)
end

"""
    neardup_block!(push_fun, idx, X, imap, L, D, M, ϵ)

# Arguments:
- `push_fun` function to push into `idx` (e.g., to pass specific arguments or catch objects as they are found)
- `idx` the output index
- `X` input database- `L` nearest neighbors of the input database to non-near dups
- `imap` list of items to test and insert
- `D` nearest neighbors distances of the input database to non-near dups
- `M` maps of `idx` to the input database
- `ϵ` radius to consider objects as near dups
"""
function neardup_block!(push_fun, idx, X, imap, tmp, L, D, M, ϵ, minbatch=0)
    empty!(tmp)
    n = length(imap)
    i = first(imap)
    push!(tmp, i)
    push!(M, i)
    push_fun(idx, X[i])
    L[i] = i
    D[i] = 0f0

    dist = distance(idx)
    R = KnnResult(1)

    for ii in 2:n
        reuse!(R)
        i = imap[ii]
        u = X[i]
        minbatch_ = getminbatch(minbatch, length(tmp))

        @batch minbatch=minbatch_ per=thread for jj in eachindex(tmp)
            j = tmp[jj]
            d = evaluate(dist, u, X[j])
            push_item!(R, j, d)
        end

        nn, d = argmin(R), minimum(R)
        if d > ϵ
            push!(tmp, i)
            push!(M, i)
            push_fun(idx, u)
            L[i] = i
            D[i] = 0f0
        else
            L[i] = nn
            D[i] = d
        end
    end
end


function neardup(push_fun::Function, idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real; k::Int=8, blocksize::Int=256)
    n = length(X)
    blocksize = min(blocksize, n) 
    res = KnnResult(k)  # should be 1, but index's setups work better on larger `k` values
    knns = Matrix{Int32}(undef, k, blocksize)
    dists = Matrix{Float32}(undef, k, blocksize)

    L = zeros(Int32, n)
    D = zeros(Float32, n)
    M = UInt32[]
    imap = UInt32[]
    tmp = UInt32[]

    for r in Iterators.partition(1:n, blocksize)
        if length(idx) == 0
            neardup_block!(push_fun, idx, X, r, tmp, L, D, M, ϵ)
        else
            empty!(imap)
            searchbatch(idx, X[r], knns, dists)
            for (i, j) in enumerate(r) # collecting non-discarded near duplicated objects
                d, nn = dists[1, i], knns[1, i]
                if d > ϵ
                    push!(imap, j)
                else
                    L[j] = M[nn]
                    D[j] = d
                end
            end

            if length(imap) > 0
                neardup_block!(push_fun, idx, X, imap, tmp, L, D, M, ϵ)
            end
        end 
    end


    (idx=idx, map=M, nn=L, dist=D)
end
