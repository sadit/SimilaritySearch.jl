# This file is a part of SimilaritySearch.jl

export neardup


"""
    neardup(push_fun::Function, idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real; k::Int=8, blocksize::Int=256, minbatch=0, filterblocks=true, verbose=true)
    neardup(idx::AbstractSearchIndex, X::AbstractVector, ϵ; kwargs...)

Find nearest duplicates in database `X` using the empty index `idx`. The algorithm iteratively try to index elements in `X`,
and items being near than `ϵ` to some element in `idx` will be ignored.

The function returns a named tuple `(idx, map, nn, dist)` where:
- `idx`: it is the index of the non duplicated elements
- `map`: a mapping from `|idx|-1` to its positions in `X`
- `nn`: an array where each element in ``x \\in X`` points to its covering element (previously indexed element `u` such that ``d(u, x_i) \\leq ϵ``)
- `dist`: an array of distance values to each covering element (correspond to each element in `nn`)

`push_fun` argument can be used to customize object insertions, e.g., set `SearchGraphCallbacks` for `SearchGraph`, it can be passed as a `do` block.

# Arguments
- `idx`: An empty index (i.e., a `SearchGraph`)
- `X`: The input dataset
- `ϵ`: Real value to cut

# Keyword arguments
- `k`: The number of nearest neighbors to retrieve (some algorithms benefit from retrieving larger `k` values)
- `blocksize`: the number of items processed at the time
- `minbatch`: argument to control `@batch` macro (see `Polyester` package multithreading)
- `filterblocks`: if true then it filters neardups inside blocks (see `blocksize` parameter), if false then it supposes that blocks are free of neardups (e.g., randomized databases)
- `verbose`: controls the verbosity of the function

# Notes
- The index `idx` must support incremental construction, e.g., with a valid `push_item!` implementation
- You can access the set of elements being 'ϵ'-non duplicates (the ``ϵ-net``) using `idx.db` or where `nn[i] == i`
"""
function neardup(idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real; kwargs...)
    neardup(push_item!, idx, X, ϵ; kwargs...)
end

"""
    neardup_block!(push_fun, idx, X, imap, tmp, L, D, M, ϵ; minbatch::Int, filterblocks::Bool)

# Arguments:
- `push_fun` function to push into `idx` (e.g., to pass specific arguments or catch objects as they are found)
- `idx` the output index
- `X` input database- `L` nearest neighbors of the input database to non-near dups
- `imap` list of items to test and insert
- `tmp` a temporary buffer to save imap elements
- `L` nearest neighbors ids of the input database to non-near dups
- `D` nearest neighbors distances of the input database to non-near dups
- `M` maps of `idx` to the input database
- `ϵ` radius to consider objects as near dups
- `minbatch` argument for the `@batch` macro (Polyester multithreading)
- `filterblocks` if true it performs neardup in blocks
"""
function neardup_block!(push_fun, idx, X, imap, tmp, L, D, M, ϵ; minbatch::Int, filterblocks::Bool)
    if !filterblocks
        for i in imap
            push!(M, i)
            push_fun(idx, X[i])
            L[i] = i
            D[i] = 0f0
        end

        return
    end

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
    push_lock = Threads.SpinLock()

    for ii in 2:n
        reuse!(R)
        i = imap[ii]
        u = X[i]
        minbatch_ = getminbatch(minbatch, length(tmp))

        @batch minbatch=minbatch_ per=thread for jj in eachindex(tmp)
            j = tmp[jj]
            d = evaluate(dist, u, X[j])
            try
                lock(push_lock)
                push_item!(R, j, d)
            finally
                unlock(push_lock)
            end
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


function neardup(push_fun::Function, idx::AbstractSearchIndex, X::AbstractDatabase, ϵ::Real;
        k::Int=8, blocksize::Int=256, filterblocks=true, minbatch::Int=0, verbose::Bool=true)
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
            if verbose
                @info "neardup> starting: $(r), current elements: $(length(idx)), n: $n, timestamp: $(Dates.now())"
            end
            neardup_block!(push_fun, idx, X, r, tmp, L, D, M, ϵ; minbatch, filterblocks)
        else
            empty!(imap)
            searchbatch(idx, X[r], knns, dists)
            if verbose
                @info "neardup> range: $(r), current elements: $(length(idx)), n: $n, timestamp: $(Dates.now())"
            end

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
                neardup_block!(push_fun, idx, X, imap, tmp, L, D, M, ϵ; minbatch, filterblocks)
            end
        end 
    end


    (idx=idx, map=M, nn=L, dist=D)
end
