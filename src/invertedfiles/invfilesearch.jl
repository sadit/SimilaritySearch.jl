# This file is part of InvertedFiles.jl

export select_posting_lists, search_invfile

"""
	select_posting_lists(idx::AbstractInvertedFile, ctx::InvertedFileContext, q, tol)

Fetches and prepares the involved posting lists to solve `q`
"""
function select_posting_lists(accept::Function, idx::AbstractInvertedFile, ctx::InvertedFileContext, q)
	Q = getcontainer(idx, ctx)
	
	@inbounds for (tokenID, weight) in sparseiterator(q)
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
    search_invfile(idx, ctx, Q, res, t)
end
