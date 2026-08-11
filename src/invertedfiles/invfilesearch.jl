# This file is part of InvertedFiles.jl

export select_posting_lists, search_invfile

"""
	select_posting_lists(idx::AbstractInvertedFile, ctx::InvertedFileContext, q, tol)

Fetches and prepares the involved posting lists to solve `q`
"""
function select_posting_lists(accept::Function, idx::AbstractInvertedFile, ctx::InvertedFileContext, q)
	Q = getcontainer(idx, ctx)

	@inbounds for (tokenID, weight) in sparseiterator(idx.dist, q)
    accept((; idx, q, tokenID, weight)) || continue
		tokenID == 0 && continue
		N = neighbors(idx.adj, tokenID)
		N === nothing && continue
		if length(N) > 0
			L = PostingList(N, convert(UInt32, tokenID), convert(Float32, weight))
			push!(Q, L)
		end
	end

	Q
end

"""
	search(idx::AbstractInvertedFile, ctx::InvertedFileContext, q, res::AbstractKnn; tol=1e-6, t=1)

Searches `q` in `idx` using the cosine dissimilarity, it computes the full operation on `idx`. `res` specify the query
"""
function search(idx::AbstractInvertedFile, ctx::InvertedFileContext, q, res::AbstractKnn; tol=1e-6, t=1)
    tol = convert(Float32, tol)
    search_invfile(idx, ctx, q, res, t) do plist
        plist.weight >= tol
    end
end

function search_invfile(accept_posting_list::Function, idx::AbstractInvertedFile, ctx::InvertedFileContext, q, res::AbstractKnn, t)
    Q = select_posting_lists(accept_posting_list, idx, ctx, q)
    n = length(Q)
    n == 0 && return res
    search_invfile(idx, ctx, q, Q, res, t)
    res
end

# ── Generic fallback (no closed-form fast path) ─────────────────────────────

struct FallbackInvFileOutput{InvFileType<:InvertedFile,QType,Knn<:AbstractKnn}
    idx::InvFileType
    q::QType
    res::Knn
end

function Intersections.onmatch!(output::FallbackInvFileOutput, L, P, m::Int)
    @inbounds objID = L[1][P[1]]
    d = Dist.evaluate(output.idx.dist, output.idx.db[objID], output.q)
    push_item!(output.res, objID, d)
end

# ── Set/token adjacency (AdjType eltype == UInt32) ──────────────────────────

struct SetInvFileOutput{InvFileType<:InvertedFile,Knn<:AbstractKnn}
    idx::InvFileType
    res::Knn
    n::Int
end

function Intersections.onmatch!(output::SetInvFileOutput, L, P, isize::Int)
    @inbounds objID = L[1].list[P[1]]
    @inbounds d = set_distance_evaluate(output.idx.dist, isize, output.n, output.idx.sizes[objID])
    push_item!(output.res, objID, d)
end

"""
  search_invfile(idx::InvertedFile, ctx::InvertedFileContext, q, Q, res::AbstractKnn, t)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments

- `idx`: inverted index
- `q`: the query object, only used for distances without an exact fast path (see [`InvertedFiles.has_exact_fastpath`](@ref))
- `Q`: the set of involved posting lists, see [`select_posting_lists`](@ref)
- `t`: threshold (t=1 union, t > 1 solves the t-threshold problem); for distances without an exact
  fast path, `t` also bounds how many real `evaluate` calls happen per query — raise it to reduce cost.
"""
function search_invfile(idx::InvertedFile{<:Any,<:AbstractAdjList{UInt32}}, ctx::InvertedFileContext, q, Q::Vector{PostType}, res::AbstractKnn, t) where {PostType<:PostingList}
    n = length(Q)
    P = getpositions(n, ctx)
    cost = if has_exact_fastpath(idx.dist)
        xmerge!(SetInvFileOutput(idx, res, n), Q, P; t)
    else
        xmerge!(FallbackInvFileOutput(idx, q, res), Q, P; t)
    end
    add_block_evaluations!(ctx, length(Q))
    add_distance_evaluations!(ctx, cost)
    res
end

