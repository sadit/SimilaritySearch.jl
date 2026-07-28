# This file is a part of SimilaritySearch.jl
using SimilaritySearch, Test, Distances

"""
Reference (non-SIMD) implementation of the code-space squared L2 / dot product between
two nibble- or byte-packed globally-quantized vectors, used to cross-check the SIMD
kernels in `ScalarQuant.SQgu4`/`ScalarQuant.SQgu8`.
"""
function manual_packed_sql2(qa, qb; bits::Int)
    res = 0
    if bits == 4
        for i in eachindex(qa)
            xa, xb = qa[i], qb[i]
            for shift in (0, 4)
                va = Int(xa >>> shift) & 0x0f
                vb = Int(xb >>> shift) & 0x0f
                res += (va - vb)^2
            end
        end
    else
        for i in eachindex(qa)
            res += (Int(qa[i]) - Int(qb[i]))^2
        end
    end
    Float32(res)
end

function manual_packed_dot(qa, qb; bits::Int)
    res = 0
    if bits == 4
        for i in eachindex(qa)
            xa, xb = qa[i], qb[i]
            for shift in (0, 4)
                va = Int(xa >>> shift) & 0x0f
                vb = Int(xb >>> shift) & 0x0f
                res += va * vb
            end
        end
    else
        for i in eachindex(qa)
            res += Int(qa[i]) * Int(qb[i])
        end
    end
    -Float32(res)
end

@testset "ScalarQuant: per-column quantization (SQu2, SQu4, SQu8)" begin
    dim, n = 20, 30  # multiple of 4, so it satisfies SQu2's (and SQu4's) packing requirement

    for (mod, bits, has_normcosine, mixed_with_plain_works) in (
            (ScalarQuant.SQu2, 2, false, true),
            (ScalarQuant.SQu4, 4, false, true),
            (ScalarQuant.SQu8, 8, true, false),
        )
        X = rand(Float32, dim, n)
        db = mod.quantize(X)
        @test length(db) == n

        maxcode = 2^bits - 1
        for i in 1:n
            qv = db[i]
            @test length(qv) == dim
            col = view(X, :, i)
            cmin, cmax = extrema(col)
            step = (cmax - cmin + 1f-6) / maxcode
            for j in 1:dim
                @test abs(qv[j] - col[j]) <= step + 1f-4
            end
        end

        a, b = db[1], db[2]
        sql2 = evaluate(mod.SqL2(), a, b)
        l2 = evaluate(mod.L2(), a, b)
        l1 = evaluate(mod.L1(), a, b)
        @test sql2 >= 0
        @test l2 ≈ sqrt(sql2) atol=1f-3
        if bits > 2
            # SQu2's L1 skips `abs` by design (documented as a ranking-only approximation)
            @test l1 >= 0
        end

        if mixed_with_plain_works
            # SqL2/L2 must also work (both argument orders) against a plain, non-quantized
            # vector of the same (padded) dimension
            plain = X[:, 3]
            sql2_mixed = evaluate(mod.SqL2(), a, plain)
            sql2_mixed_rev = evaluate(mod.SqL2(), plain, a)
            @test sql2_mixed >= 0
            @test sql2_mixed ≈ sql2_mixed_rev atol=1f-4
            manual = sum(j -> (a[j] - plain[j])^2, 1:dim)
            @test sql2_mixed ≈ manual atol=1f-3
        else
            # NOTE: SQu8's `squared_euclidean(A::SQu8Vec, B)` (plain-vector overload)
            # currently assumes `B` also has a `.V` field (i.e., is itself packed), so it
            # errors on a genuine plain vector; not exercised here pending a fix.
        end

        if has_normcosine
            nc = evaluate(mod.NormCosine(), a, b)
            @test nc isa Float32
        end
    end
end

@testset "ScalarQuant: dimension-conformance ArgumentError (SQu2, SQu4)" begin
    # SQu2 packs 4 codes/UInt8; SQu4 packs 2 codes/UInt8 -- non-conforming dims must be
    # rejected upfront (at `quantize`/`SQuXVec` construction time) instead of silently
    # padding/truncating and mishandling the resulting tail downstream.
    @test_throws ArgumentError ScalarQuant.SQu2.quantize(rand(Float32, 17, 5))
    @test_throws ArgumentError ScalarQuant.SQu2.SQu2Vec(rand(Float32, 17))
    @test_throws ArgumentError ScalarQuant.SQu4.quantize(rand(Float32, 17, 5))
    @test_throws ArgumentError ScalarQuant.SQu4.SQu4Vec(rand(Float32, 17))

    # conforming dims work
    @test ScalarQuant.SQu2.quantize(rand(Float32, 16, 5)) isa ScalarQuant.SQu2.SQu2Database
    @test ScalarQuant.SQu4.quantize(rand(Float32, 16, 5)) isa ScalarQuant.SQu4.SQu4Database
end

@testset "ScalarQuant: global quantization (SQgu4, SQgu8)" begin
    dim, n = 200, 8  # dim > 128 exercises the unrolled + single + scalar-tail SIMD phases

    for (mod, bits) in ((ScalarQuant.SQgu4, 4), (ScalarQuant.SQgu8, 8))
        X = rand(Float32, dim, n)
        Q = mod.quantize(X; minmax=(0f0, 1f0))
        expected_rows = bits == 4 ? cld(dim, 2) : dim
        @test size(Q) == (expected_rows, n)
        @test eltype(Q) == UInt8

        # default (quantile-estimated) minmax path should also run without error
        Q2 = mod.quantize(X)
        @test size(Q2) == size(Q)

        maxcode = 2^bits - 1
        c = Float32(maxcode / (1f0 - 0f0 + 1f-6))
        for pair in ((1, 2), (3, 4), (5, 6))
            a, b = view(Q, :, pair[1]), view(Q, :, pair[2])

            sql2 = evaluate(mod.SqL2(), a, b)
            @test sql2 == manual_packed_sql2(a, b; bits)

            nc = evaluate(mod.NormCosine(), a, b)
            @test nc == manual_packed_dot(a, b; bits)
            @test nc <= 0  # it's a negated dot product of non-negative codes
        end

        # loose round-trip sanity: quantized SqL2 should correlate with true squared L2
        # (same global scale for every column, so ranking is preserved up to code rounding);
        # a generous relative tolerance absorbs the accumulated per-coordinate rounding noise
        true_sql2 = sum(abs2, view(X, :, 1) .- view(X, :, 2))
        code_sql2 = evaluate(mod.SqL2(), view(Q, :, 1), view(Q, :, 2))
        @test code_sql2 / c^2 ≈ true_sql2 rtol=0.3
    end
end
