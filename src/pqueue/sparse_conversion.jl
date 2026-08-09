using SparseArrays

"""
    sparse(res::AbstractKnn, n::Integer)

Converts a k-NN queue into a `SparseVector` of length `n`. 
The `(id, distance)` pairs are extracted and sorted by `id` to satisfy the `SparseVector` constraints.
"""
@inline _lt_id(X, i, j) = @inbounds X[1][i] < X[1][j]
@inline function _swap_id_val(X, i, j)
    @inbounds X[1][i], X[1][j] = X[1][j], X[1][i]
    @inbounds X[2][i], X[2][j] = X[2][j], X[2][i]
end

function SparseArrays.sparse(res::AbstractKnn, n::Integer)
    len = length(res)
    nzind = Vector{Int}(undef, len)
    nzval = Vector{Float32}(undef, len)
    
    # Extract
    ids = IdView(res)
    dists = DistView(res)
    @inbounds @simd for i in 1:len
        nzind[i] = Int(ids[i])
    end

    @inbounds @simd for i in 1:len
        nzval[i] = dists[i]
    end
    
    # Sort by ID for SparseVector
    X = (nzind, nzval)
    heapify!(_lt_id, _swap_id_val, X, len)
    heapsort!(_lt_id, _swap_id_val, X, len)
    
    SparseVector(n, nzind, nzval)
end

"""
    sparse(ids::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}, n::Integer)

Converts a batch of search results (where columns represent query results) into a `SparseMatrixCSC` of size `(n, m)`.
The indices in each column are sorted by ID.
"""
function SparseArrays.sparse(ids::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32}, n::Integer)
    k, m = size(ids)
    nnz_max = k * m
    colptr = Vector{Int}(undef, m + 1)
    rowval = Vector{Int}(undef, nnz_max)
    nzval = Vector{Float32}(undef, nnz_max)
    
    colptr[1] = 1
    pos = 1
    
    buf_ind = Vector{Int}(undef, k)
    buf_val = Vector{Float32}(undef, k)
    
    for j in 1:m
        len = 0
        for i in 1:k
            id = ids[i, j]
            if id > 0
                len += 1
                @inbounds buf_ind[len] = Int(id)
                @inbounds buf_val[len] = dists[i, j]
            end
        end
        
        v_ind = view(buf_ind, 1:len)
        v_val = view(buf_val, 1:len)
        X = (v_ind, v_val)
        heapify!(_lt_id, _swap_id_val, X, len)
        heapsort!(_lt_id, _swap_id_val, X, len)
        
        @inbounds @simd for i in 1:len
            rowval[pos + i - 1] = v_ind[i]
            nzval[pos + i - 1] = v_val[i]
        end
        pos += len
        @inbounds colptr[j+1] = pos
    end
    
    resize!(rowval, pos - 1)
    resize!(nzval, pos - 1)
    
    SparseMatrixCSC(n, m, colptr, rowval, nzval)
end

@inline _lt_val(X, i, j) = @inbounds X[2][i] < X[2][j]

"""
    knnqueue(::Type{T}, vec::SparseVector) where {T<:AbstractKnn}

Creates a k-NN queue from a `SparseVector` by pushing all non-zero entries.
The queue will automatically reorder the elements by distance.
"""
function knnqueue(::Type{T}, vec::SparseVector) where {T<:AbstractKnn}
    k = nnz(vec)
    res = knnqueue(T, k)
    for i in 1:k
        @inbounds push_item!(res, vec.nzind[i], vec.nzval[i])
    end
    res
end

function knnqueue(::Type{KnnSorted}, vec::SparseVector)
    k = nnz(vec)
    ids = Vector{UInt32}(undef, k)
    dists = Vector{Float32}(undef, k)
    
    @inbounds @simd for i in 1:k
        ids[i] = vec.nzind[i]
        dists[i] = vec.nzval[i]
    end
    
    KnnSorted(ids, dists; is_items=true)
end

"""
    knn_matrices(mat::SparseMatrixCSC, k::Integer) -> (ids, dists)

Converts a `SparseMatrixCSC` into a batch result representation (dense matrices `ids` and `dists` of size `(k, m)`).
The elements in each column are reordered by distance.
"""
function knn_matrices(mat::SparseMatrixCSC, k::Integer)
    m = size(mat, 2)
    ids = zeros(UInt32, k, m)
    dists = fill(typemax(Float32), k, m)
    
    for j in 1:m
        start_idx = mat.colptr[j]
        end_idx = mat.colptr[j+1] - 1
        len = end_idx - start_idx + 1
        
        if len > 0
            res = knnqueue(KnnSorted, k)
            
            if len <= k
                ids_col = Vector{UInt32}(undef, len)
                dists_col = Vector{Float32}(undef, len)
                @inbounds @simd for i in 1:len
                    ids_col[i] = mat.rowval[start_idx + i - 1]
                    dists_col[i] = mat.nzval[start_idx + i - 1]
                end
                
                res = KnnSorted(ids_col, dists_col; is_items=true)
                
                v_ids = IdView(res)
                v_dists = DistView(res)
                @inbounds @simd for i in 1:len
                    ids[i, j] = v_ids[i]
                end

                @inbounds @simd for i in 1:len
                    dists[i, j] = v_dists[i]
                end
            else
                res = knnqueue(KnnSorted, k)
                @inbounds @simd for i in start_idx:end_idx
                    push_item!(res, mat.rowval[i], mat.nzval[i])
                end
                
                n_res = length(res)
                v_ids = IdView(res)
                v_dists = DistView(res)
                @inbounds @simd for i in 1:n_res
                    ids[i, j] = v_ids[i]
                end

                @inbounds @simd for i in 1:n_res
                    dists[i, j] = v_dists[i]
                end
            end
        end
    end
    
    ids, dists
end
