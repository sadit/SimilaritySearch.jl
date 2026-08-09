using Test
using SparseArrays
using SimilaritySearch

@testset "Sparse Conversions" begin
    # Test converting a single KnnResult to SparseVector and back
    k = 5
    res = knnqueue(KnnSorted, k)
    push_item!(res, 10, 0.5f0)
    push_item!(res, 3, 0.1f0)
    push_item!(res, 7, 0.9f0)

    # Convert to SparseVector
    n = 20
    sv = sparse(res, n)
    @test sv isa SparseVector
    @test length(sv) == n
    @test nnz(sv) == 3
    @test sv.nzind == [3, 7, 10]
    @test sv.nzval ≈ [0.1f0, 0.9f0, 0.5f0]

    # Convert back to KnnSorted
    res_back = knnqueue(KnnSorted, sv)
    @test res_back isa KnnSorted
    @test length(res_back) == 3
    
    # After conversion back, it should be sorted by distance
    items = collect(viewitems(res_back))
    @test items[1].id == 3
    @test items[1].dist ≈ 0.1f0
    @test items[2].id == 10
    @test items[2].dist ≈ 0.5f0
    @test items[3].id == 7
    @test items[3].dist ≈ 0.9f0

    # Test converting batch results to SparseMatrixCSC and back
    ids = UInt32[
        10 5 0;
        3  8 0;
        7  2 0;
        0  1 0
    ]
    dists = Float32[
        0.5 0.3 Inf;
        0.1 0.8 Inf;
        0.9 0.2 Inf;
        Inf 0.1 Inf
    ]
    
    # 3 columns, k=4, valid items: 3, 4, 0
    sm = sparse(ids, dists, n)
    @test sm isa SparseMatrixCSC
    @test size(sm) == (n, 3)
    @test nnz(sm) == 7
    
    # Col 1: ids (3, 7, 10), dists (0.1, 0.9, 0.5)
    c1 = sm[:, 1]
    @test c1.nzind == [3, 7, 10]
    @test c1.nzval ≈ [0.1, 0.9, 0.5]
    
    # Col 2: ids (1, 2, 5, 8), dists (0.1, 0.2, 0.3, 0.8)
    c2 = sm[:, 2]
    @test c2.nzind == [1, 2, 5, 8]
    @test c2.nzval ≈ [0.1, 0.2, 0.3, 0.8]
    
    # Col 3: empty
    c3 = sm[:, 3]
    @test nnz(c3) == 0

    # Convert back to ids and dists
    out_ids, out_dists = knn_matrices(sm, 4)
    @test size(out_ids) == (4, 3)
    @test size(out_dists) == (4, 3)
    
    # Col 1 should be sorted by distance: 3 (0.1), 10 (0.5), 7 (0.9), 0 (Inf)
    @test out_ids[:, 1] == [3, 10, 7, 0]
    @test out_dists[:, 1] ≈ [0.1, 0.5, 0.9, typemax(Float32)]
    
    # Col 2 should be sorted by distance: 1 (0.1), 2 (0.2), 5 (0.3), 8 (0.8)
    @test out_ids[:, 2] == [1, 2, 5, 8]
    @test out_dists[:, 2] ≈ [0.1, 0.2, 0.3, 0.8]
    
    # Col 3 should be empty
    @test out_ids[:, 3] == [0, 0, 0, 0]
    @test out_dists[:, 3] == fill(typemax(Float32), 4)
end
