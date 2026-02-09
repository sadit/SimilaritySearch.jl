# This file is a part of SimilaritySearch.jl

export neardup

"""
    neardup(idx::AbstractSearchIndex, ctx::AbstractContext, X::AbstractDatabase, ϵ::Real; k::Int=8, blocksize::Int=get_parallel_block(), filterblocks=true, verbose=true)
    neardup(dist::PreMetric, X::AbstractDatabase, ϵ::Real; kwargs...)

Find nearest duplicates in database `X` using the empty index `idx`. The algorithm iteratively try to index elements in `X`,
and items being near than `ϵ` to some element in `idx` will be ignored.

The function returns a named tuple `(idx, map, nn, dist)` where:
- `idx`: it is the index of the non duplicated elements
- `ctx`: the index's context
- `map`: a mapping from `|idx|-1` to its positions in `X`
- `nn`: an array where each element in ``x \\in X`` points to its covering element (previously indexed element `u` such that ``d(u, x_i) \\leq ϵ``)
- `dist`: an array of distance values to each covering element (correspond to each element in `nn`)


# Arguments
- `idx`: An empty index (i.e., a `SearchGraph`)
- `X`: The input dataset
- `ϵ`: Real value to cut, if negative, then ϵ will be computed using the quantile value at 'abs(ϵ)' in a small sample of nearest neighbor distances; the quantile method should be used only for applications that need some vague approximations to `ϵ`

# Keyword arguments
- `k`: The number of nearest neighbors to retrieve (some algorithms benefit from retrieving larger `k` values)
- `blocksize`: the number of items processed at the time
- `filterblocks`: if true then it filters neardups inside blocks (see `blocksize` parameter), otherwise, it supposes that blocks are free of neardups (e.g., randomized order).
- `verbose`: controls the verbosity of the function

# Notes
- The index `idx` must support incremental construction
- If you need to customize object insertions, you must wrap the index `idx` and implement your custom methods; it requires valid implementations of the following functions:
   - `searchbatch(idx::AbstractSearchIndex, ctx, queries::AbstractDatabase, knns::Matrix, dists::Matrix)`
   - `distance(idx::AbstractSearchIndex)`
   - `length(idx::AbstractSearchIndex)`
   - `append_items!(idx::AbstractSearchIndex, ctx, items::AbstractDatabase)`
- You can access the set of elements being 'ϵ-non duplicates (the ``ϵ-net``) using `database(idx)` or where `nn[i] == i`
"""
function neardup(dist::PreMetric, X::AbstractDatabase, ϵ::Real; recall=1.0, kwargs...)
    dist_ = DistanceWithIdentifiers(dist, X)
    X_ = VectorDatabase(UInt32[])
    if recall < 1.0
        idx = SearchGraph(; dist=dist_, db=X_)
        hyperparameters_callback = OptimizeParametes(MinRecall(recall))
        ctx = SearchGraphContext(getcontext(G); hyperparameters_callback)
    else
        idx = ExhaustiveSearch(; dist=dist_, db=X_)
        ctx = getcontext(idx)
    end

    R = neardup_(idx, ctx, VectorDatabase(UnitRange{UInt32}(1, length(X))), ϵ; kwargs...)
    (; R..., centers=X_.vecs)
end

function neardup(idx::AbstractSearchIndex, ctx::AbstractContext, X::AbstractDatabase, ϵ::Real; kwargs...)
    R = neardup_(idx, ctx, X, ϵ; kwargs...)
    centers = sort!(unique(R.nn))
    (; R..., centers)
end

function neardup_(idx::AbstractSearchIndex, ctx::AbstractContext, X::AbstractDatabase, ϵ::Real;
    k::Int=8, blocksize::Int=256, filterblocks=true, verbose::Bool=true)

    ϵ = convert(Float32, ϵ)
    n = length(X)
    blocksize = min(blocksize, n)
    knns = Matrix{IdWeight}(undef, k, blocksize)

    L = zeros(Int32, n)
    D = zeros(Float32, n)
    M = UInt32[]
    imap = UInt32[]
    tmp = UInt32[]

    for range in Iterators.partition(1:n, blocksize)
        if length(idx) == 0
            if verbose
                @info "neardup> starting: $(range), current elements: $(length(idx)), n: $n, ϵ: $ϵ, timestamp: $(Dates.now())"
            end
            neardup_block!(idx, ctx, X, range, tmp, L, D, M, ϵ; filterblocks)
        else
            empty!(imap)
            if size(knns, 2) != length(range)
                knns_ = view(knns, :, 1:length(range)) # the last range can change its size
                fill!(knns_, zero(IdWeight))
                searchbatch!(idx, ctx, X[range], knns_; sorted=true)
            else
                fill!(knns, zero(IdWeight))
                searchbatch!(idx, ctx, X[range], knns; sorted=true)
            end
            # @assert all(r -> length(r) > 0, view(knns, :, 1:length(range)))
            if verbose
                @info "neardup> range: $(range), current elements: $(length(idx)), n: $n, ϵ: $ϵ, timestamp: $(Dates.now())"
            end

            for (i, j) in enumerate(range) # collecting non-discarded near duplicated objects
                #d, nn = knns[1, i] #p = nearest(knns[i])
                p = knns[1, i]
                if p.weight > ϵ
                    push!(imap, j)
                else
                    D[j] = p.weight
                    L[j] = M[p.id]
                end
            end

            if length(imap) > 0
                neardup_block!(idx, ctx, X, imap, tmp, L, D, M, ϵ; filterblocks)
            end
        end
    end
    if verbose
        @info "neardup> finished current elements: $(length(idx)), n: $n, ϵ: $ϵ, timestamp: $(Dates.now())"
    end

    (idx=idx, map=M, nn=L, dist=D)
end


"""
    neardup_block!(idx, ctx, X, imap, tmp, L, D, M, ϵ; filterblocks::Bool)

# Arguments:
- `idx` the output index
- `ctx` the index's context
- `X` input database- `L` nearest neighbors of the input database to non-near dups
- `imap` list of items to test and insert
- `tmp` a temporary buffer to save imap elements
- `L` nearest neighbors ids of the input database to non-near dups
- `D` nearest neighbors distances of the input database to non-near dups
- `M` maps of `idx` to the input database
- `ϵ` radius to consider objects as near dups
- `filterblocks` if true it performs neardup in blocks
"""
function neardup_block!(idx::AbstractSearchIndex, ctx::AbstractContext, X::AbstractDatabase, imap, tmp, L, D, M, ϵ; filterblocks::Bool)
    if !filterblocks
        append_items!(idx, ctx, X[imap])
        for i in imap
            push!(M, i)
            L[i] = i
            D[i] = 0.0f0
        end

        return
    end

    empty!(tmp)
    n = length(imap)
    i = first(imap)
    push!(tmp, i)
    push!(M, i)
    L[i] = i
    D[i] = 0.0f0

    dist = distance(idx)
    res = knnqueue(ctx, 1)
    push_lock = Threads.SpinLock()

    for ii in 2:n
        reuse!(res)
        i = imap[ii]
        u = X[i]
        minbatch = getminbatch(ctx, length(tmp))

        #Threads.@threads :static for j in firstindex(tmp):minbatch:lastindex(tmp)
        @batch per=thread minbatch=4 for j in firstindex(tmp):minbatch:lastindex(tmp)
            for jj in j:min(lastindex(tmp), j + minbatch - 1)
                j = tmp[jj]
                d = evaluate(dist, u, X[j])
                try
                    lock(push_lock)
                    push_item!(res, j, d)
                finally
                    unlock(push_lock)
                end
            end
        end

        let nn = nearest(res)
            if nn.weight > ϵ
                push!(tmp, i)
                push!(M, i)
                L[i] = i
                D[i] = 0.0f0
            else
                L[i] = nn.id
                D[i] = nn.weight
            end
        end
    end

    append_items!(idx, ctx, X[tmp])
end
