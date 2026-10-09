# This file is a part of SimilaritySearch.jl
using SimilaritySearch, Test, Distances, Random, LinearAlgebra
using SimilaritySearch.ScalarQuant: SQEncoder, sqcodes, SQQuery, codequerydot, dotquery
using SimilaritySearch: encodequery

"""
Reference (non-SIMD) implementation of the code-space squared L2 / dot product between
two 2-bit-, nibble- or byte-packed globally-quantized vectors, used to cross-check the SIMD
kernels in `ScalarQuant.SQgu2`/`ScalarQuant.SQgu4`/`ScalarQuant.SQgu8`.
"""
function manual_packed_sql2(qa, qb; bits::Int)
    res = 0
    if bits == 2
        for i in eachindex(qa)
            xa, xb = qa[i], qb[i]
            for shift in (0, 2, 4, 6)
                va = Int(xa >>> shift) & 0x03
                vb = Int(xb >>> shift) & 0x03
                res += (va - vb)^2
            end
        end
    elseif bits == 4
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
    if bits == 2
        for i in eachindex(qa)
            xa, xb = qa[i], qb[i]
            for shift in (0, 2, 4, 6)
                va = Int(xa >>> shift) & 0x03
                vb = Int(xb >>> shift) & 0x03
                res += va * vb
            end
        end
    elseif bits == 4
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

    for (mod, bits) in ((ScalarQuant.SQu2, 2), (ScalarQuant.SQu4, 4), (ScalarQuant.SQu8, 8))
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
        @test l1 >= 0
        # L1 is a true metric at every width (SQu2's used to skip `abs`)
        @test l1 ≈ sum(j -> abs(a[j] - b[j]), 1:dim) atol=1f-3

        # SqL2/L2 must also work (both argument orders) against a plain, non-quantized
        # vector of the same (padded) dimension
        plain = X[:, 3]
        sql2_mixed = evaluate(mod.SqL2(), a, plain)
        sql2_mixed_rev = evaluate(mod.SqL2(), plain, a)
        @test sql2_mixed >= 0
        @test sql2_mixed ≈ sql2_mixed_rev atol=1f-4
        manual = sum(j -> (a[j] - plain[j])^2, 1:dim)
        @test sql2_mixed ≈ manual atol=1f-3

    end
end

@testset "ScalarQuant: per-column SqL2 is exact on equal scales and accurate otherwise" begin
    # SqL2 between two per-column vectors no longer dequantizes coordinate by coordinate: it
    # expands (a*cA + mA - b*cB - mB)^2, which leaves one integer dot product over the codes
    # plus per-vector sums known since quantization. Two properties that buys, and one it
    # would have cost without the equal-scale branch:
    #
    #   * more accurate than the loop it replaces, because the sums are integers rather than
    #     Float32 accumulations -- checked here against a Float64 evaluation of the same codes;
    #   * exactly 0f0 for a vector against itself. The general expansion cancels only up to
    #     rounding, so equal scales (self-comparisons, duplicates) take an exact integer path.
    #     `neardup` at radius 0 depends on this.
    for (mod, cpb) in ((ScalarQuant.SQu2, 4), (ScalarQuant.SQu4, 2), (ScalarQuant.SQu8, 1))
        for dim in (8, 16, 64, 260)
            X = randn(Float32, dim, 24)
            db = mod.quantize(X)

            for i in 1:24
                @test evaluate(mod.SqL2(), db[i], db[i]) == 0f0
                @test evaluate(mod.L2(), db[i], db[i]) == 0f0
            end

            for (i, j) in ((1, 2), (3, 11), (5, 24))
                a, b = db[i], db[j]
                truth = sum((Float64(a[t]) - Float64(b[t]))^2 for t in 1:dim)
                got = evaluate(mod.SqL2(), a, b)
                @test abs(got - truth) <= 1f-5 * max(1.0, truth)
                @test got >= 0f0
                @test evaluate(mod.L2(), a, b) ≈ sqrt(got)
            end
        end
    end

    # two vectors that share a scale but differ in codes still take the exact path and agree
    # with the general one to Float32 tolerance
    v = randn(Float32, 64)
    a = ScalarQuant.SQu8.SQu8Vec(v)
    b = ScalarQuant.SQu8.SQu8Vec(ScalarQuant.SQMinC(a.E.min, a.E.c), reverse(a.V))
    truth = sum((Float64(a[t]) - Float64(b[t]))^2 for t in 1:64)
    @test abs(evaluate(ScalarQuant.SQu8.SqL2(), a, b) - truth) <= 1f-4 * truth

    # NormCosine goes through the same expansion, at every width, and against a plain vector
    X = randn(Float32, 64, 8)
    for mod in (ScalarQuant.SQu2, ScalarQuant.SQu4, ScalarQuant.SQu8)
        db = mod.quantize(X)
        for (i, j) in ((1, 2), (3, 8))
            dot64 = sum(Float64(db[i][t]) * Float64(db[j][t]) for t in 1:64)
            @test abs(evaluate(mod.NormCosine(), db[i], db[j]) - (1.0 - dot64)) <= 1f-4 * max(1.0, abs(1.0 - dot64))
            dotq = sum(Float64(db[i][t]) * Float64(X[t, j]) for t in 1:64)
            @test abs(evaluate(mod.NormCosine(), db[i], X[:, j]) - (1.0 - dotq)) <= 1f-4 * max(1.0, abs(1.0 - dotq))
            @test evaluate(mod.NormCosine(), X[:, j], db[i]) == evaluate(mod.NormCosine(), db[i], X[:, j])
        end
    end
end

@testset "ScalarQuant: mixed quantized-vs-Float32 distances agree across container types" begin
    # The mixed path (a quantized vector against a plain Float32 one, e.g. an unquantized query)
    # has an explicitly vectorized method for contiguous Float32 storage and a generic fallback
    # for everything else. Both must agree with each other and with a Float64 evaluation; the
    # SIMD one is also the only place where a packed vector's codes are interleaved back into
    # coordinate order by hand, so an off-by-one there would be silent.
    for (mod, cpb) in ((ScalarQuant.SQu2, 4), (ScalarQuant.SQu4, 2), (ScalarQuant.SQu8, 1))
        for dim in (8, 16, 64, 100, 128, 260)
            dim % cpb == 0 || continue
            X = randn(Float32, dim, 8)
            db = mod.quantize(X)
            q = randn(Float32, dim)
            qview = view(hcat(q, q), :, 1)          # what a MatrixDatabase column looks like
            qgeneric = Float64.(q)                  # not Float32: takes the fallback

            for i in 1:8
                truth = sum((Float64(db[i][t]) - Float64(q[t]))^2 for t in 1:dim)
                got = evaluate(mod.SqL2(), db[i], q)
                @test abs(got - truth) <= 1f-5 * max(1.0, truth)
                @test evaluate(mod.SqL2(), db[i], qview) == got          # same path, same answer
                @test abs(evaluate(mod.SqL2(), db[i], qgeneric) - truth) <= 1f-5 * max(1.0, truth)
                @test evaluate(mod.L2(), db[i], q) ≈ sqrt(got)
            end

            # a quantized vector against its own dequantization is (nearly) zero distance
            deq = Float32[db[1][t] for t in 1:dim]
            @test evaluate(mod.SqL2(), db[1], deq) <= 1f-6 * max(1f0, sum(abs2, deq))

            # L1 and NormCosine go through the same block unpacking, mixed and between codes,
            # so they are checked at the same dimensions (whole blocks, partial blocks, tails)
            for i in 1:8
                l1truth = sum(abs(Float64(db[i][t]) - Float64(q[t])) for t in 1:dim)
                @test abs(evaluate(mod.L1(), db[i], q) - l1truth) <= 1f-4 * max(1.0, l1truth)
                @test evaluate(mod.L1(), q, db[i]) == evaluate(mod.L1(), db[i], q)
                @test abs(evaluate(mod.L1(), db[i], qgeneric) - l1truth) <= 1f-4 * max(1.0, l1truth)
                l1pair = sum(abs(Float64(db[i][t]) - Float64(db[3][t])) for t in 1:dim)
                @test abs(evaluate(mod.L1(), db[i], db[3]) - l1pair) <= 1f-4 * max(1.0, l1pair)
                dotq = sum(Float64(db[i][t]) * Float64(q[t]) for t in 1:dim)
                @test abs(evaluate(mod.NormCosine(), db[i], q) - (1.0 - dotq)) <= 1f-4 * max(1.0, abs(1.0 - dotq))
                @test abs(evaluate(mod.NormCosine(), db[i], qgeneric) - (1.0 - dotq)) <= 1f-4 * max(1.0, abs(1.0 - dotq))
            end
            @test evaluate(mod.L1(), db[1], db[1]) == 0f0
        end
    end
end

@testset "ScalarQuant: per-column databases rebuild from their own fields (#69)" begin
    # The persistence shape: a caller stores `E` and `Q`, and on the way back in has exactly
    # those two and not the Float32 matrix they came from. Rebuilding that matrix to re-quantize
    # costs several times the memory the quantization was chosen to avoid, to recompute codes
    # already in hand -- so every per-column database must accept its own fields. SQu2 always
    # could (it has Julia's default field constructor); SQu4/SQu8 fused quantization into their
    # only constructor and could not.
    dim, n = 8, 64
    X = rand(Float32, dim, n)

    for (mod, T) in ((ScalarQuant.SQu2, ScalarQuant.SQu2.SQu2Database),
                     (ScalarQuant.SQu4, ScalarQuant.SQu4.SQu4Database),
                     (ScalarQuant.SQu8, ScalarQuant.SQu8.SQu8Database))
        db = mod.quantize(X)
        rebuilt = T(db.E, db.Q)              # no matrix, no re-quantization

        @test length(rebuilt) == length(db)
        @test rebuilt.E == db.E
        @test rebuilt.Q == db.Q
        # identical as a database: same codes in, same distances out (L1/L2/SqL2 are defined by
        # all three modules; NormCosine only by SQu8)
        for (i, j) in ((1, 2), (3, 17), (n - 1, n))
            for dist in (mod.L1(), mod.L2(), mod.SqL2())
                @test evaluate(dist, rebuilt[i], rebuilt[j]) == evaluate(dist, db[i], db[j])
            end
            mod === ScalarQuant.SQu8 &&
                @test evaluate(mod.NormCosine(), rebuilt[i], rebuilt[j]) == evaluate(mod.NormCosine(), db[i], db[j])
        end
        # and usable as an index's database without dequantizing anything
        seq = ExhaustiveSearch(mod.SqL2(), rebuilt)
        res = search(seq, GenericContext(), rebuilt[1], knnqueue(KnnSorted, 3))
        @test nearest(res).id == 1
        @test nearest(res).dist == 0f0
    end

    # the one invariant the fields must satisfy: exactly one SQMinC per stored vector
    for (mod, T) in ((ScalarQuant.SQu4, ScalarQuant.SQu4.SQu4Database),
                     (ScalarQuant.SQu8, ScalarQuant.SQu8.SQu8Database))
        db = mod.quantize(X)
        @test_throws ArgumentError T(db.E[1:(n ÷ 2)], db.Q)
        @test_throws ArgumentError T(db.E, db.Q.matrix[:, 1:(n ÷ 2)])
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

    # the per-width names are aliases of the one vector type and the one set of distances
    @test ScalarQuant.SQu2.SQu2Vec === ScalarQuant.SQVec{2}
    @test ScalarQuant.SQu4.SQu4Vec === ScalarQuant.SQVec{4}
    @test ScalarQuant.SQu8.SQu8Vec === ScalarQuant.SQVec{8}
    @test ScalarQuant.SQu8.SqL2 === ScalarQuant.SQu4.SqL2 === ScalarQuant.SqL2
    @test ScalarQuant.SQu2.L1 === ScalarQuant.L1 && ScalarQuant.SQu2.NormCosine === ScalarQuant.NormCosine
    q = ScalarQuant.SQVec{4}(rand(Float32, 16))
    @test ScalarQuant.codewidth(q) == 4 && length(q) == 16 && eltype(q) == Float32
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

        end

        # loose round-trip sanity: quantized SqL2 should correlate with true squared L2
        # (same global scale for every column, so ranking is preserved up to code rounding);
        # a generous relative tolerance absorbs the accumulated per-coordinate rounding noise
        true_sql2 = sum(abs2, view(X, :, 1) .- view(X, :, 2))
        code_sql2 = evaluate(mod.SqL2(), view(Q, :, 1), view(Q, :, 2))
        @test code_sql2 / c^2 ≈ true_sql2 rtol=0.3
    end
end

@testset "ScalarQuant: every SIMD phase boundary (SQgu2, SQgu4, SQgu8)" begin
    # These kernels run a blocked/unrolled pass, then a half-width pass, then a scalar
    # tail, and which of them runs is decided purely by `length % N`. A remainder of 16..31
    # used to reach neither SIMD pass in the 32-byte-wide kernels (SQgu2, SQgu8) and fell
    # to the scalar loop instead; the 2-bit one skipped SIMD outright below 32 bytes, which
    # is exactly a 64-hyperplane sketch. The lengths below put the remainder in each class
    # (0, 1..15, 16, 17..31) so no phase can be silently skipped or double-counted. The
    # 16-byte-wide kernel (SQgu4) splits the same way one notch down, at 8, so the lengths
    # also cover a remainder below and above that.
    for nbytes in (0, 1, 8, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 63, 64, 65, 79, 80, 104, 127, 128, 129, 200, 208)
        a, b = rand(UInt8, nbytes), rand(UInt8, nbytes)
        for (mod, bits) in ((ScalarQuant.SQgu2, 2), (ScalarQuant.SQgu4, 4), (ScalarQuant.SQgu8, 8))
            @test evaluate(mod.SqL2(), a, b) == manual_packed_sql2(a, b; bits)
            @test evaluate(mod.SqL2(), a, a) == 0f0
        end
    end
end

"The `k`-th unpacked code of `a`, as the integer it is."
codes_at(a::SimilaritySearch.ScalarQuant.SQVec{B}, k::Integer) where {B} =
    B == 8 ? Int(a.V[k]) : B == 4 ? Int((a.V[(k-1) ÷ 2 + 1] >> (4 * ((k-1) % 2))) & 0x0f) :
    Int((a.V[(k-1) ÷ 4 + 1] >> (2 * ((k-1) % 4))) & 0x03)

@testset "ScalarQuant: prepared queries evaluate through the integer image (#110)" begin
    # `SQQuery` carries the query's sums and an integer image of it on its own range; the mixed
    # distances are ScalarQuant's generic ones, what `SQEncoder` evaluates with (the per-module
    # `SQu4.SqL2` and relatives are the raw-code distances and keep their own paths); the mixed
    # distances against it are the expansion over the stored code sums plus one integer dot
    # product, never a dequantized coordinate. Checked against a Float64 evaluation of the
    # same expansion over the image (tight), against the Float32-query path (loose: the image
    # is 15 bits against 8-bit codes, 8 bits against 4- and 2-bit codes), at whole blocks,
    # partial blocks and tails, in both families.
    rng = Xoshiro(11)
    for (mod, B, cpb) in ((ScalarQuant.SQu2, 2, 4), (ScalarQuant.SQu4, 4, 2), (ScalarQuant.SQu8, 8, 1),
                          (ScalarQuant.SQgu2, 2, 4), (ScalarQuant.SQgu4, 4, 2), (ScalarQuant.SQgu8, 8, 1))
        for dim in (8, 16, 64, 100, 128, 260)
            dim % cpb == 0 || continue
            X = randn(rng, Float32, dim, 8)
            enc = SQEncoder(mod, X)
            db = sqcodes(enc, X)
            q = randn(rng, Float32, dim)
            Q = encodequery(enc, q)
            @test Q isa SQQuery{B}
            @test Q isa AbstractVector{Float32} && length(Q) == dim && Q[3] == q[3]   # the rotated query, as a vector
            @test length(Q.planes) == cpb
            P = cpb
            qimg = [Float64(Q.mq) + Float64(Q.sq) * (Float64(Q.planes[(i-1) % P + 1][(i-1) ÷ P + 1]) + Q.half) for i in 1:dim]
            @test maximum(abs.(qimg .- q)) <= (B == 8 ? 2e-4 : 1e-2) * (maximum(q) - minimum(q))   # the image's resolution
            @test Q.sumq ≈ sum(Float64.(q)) && Q.sumqq ≈ sum(abs2, Float64.(q))
            for i in 1:8
                a = db[i]
                # the dot product of the codes with the image is exact
                @test codequerydot(a, Q) ≈ sum(Float64(codes_at(a, k)) * qimg[k] for k in 1:dim) atol=1e-6 * dim
                # SqL2 is the expansion with the image in the cross term and the query itself in the
                # rest: (c²Saa + 2cmSa + nm²) − 2(c·Σaᵢqimgᵢ + m·Σqᵢ) + Σqᵢ², the sums exact
                codes = [Float64(codes_at(a, k)) for k in 1:dim]
                c, m = Float64(a.E.c), Float64(a.E.min)
                expected = c * c * sum(codes .^ 2) + 2 * c * m * sum(codes) + dim * m * m -
                           2 * (c * sum(codes .* qimg) + m * Q.sumq) + Q.sumqq
                got = evaluate(ScalarQuant.SqL2(), a, Q)
                @test abs(got - expected) <= 1e-5 * max(1.0, expected)
                @test evaluate(ScalarQuant.SqL2(), Q, a) == got
                # against the Float32 query the image's resolution is the only difference; the
                # tolerances scale with the vectors, since these are randn and not unit vectors
                ref = evaluate(ScalarQuant.SqL2(), a, q)
                @test abs(got - ref) <= (B == 8 ? 1e-3 : 2e-2) * max(1.0, ref)
                @test evaluate(ScalarQuant.L2(), a, Q) ≈ sqrt(got)
                scale = sqrt(c * c * sum(codes .^ 2) + 2 * c * m * sum(codes) + dim * m * m) * sqrt(Q.sumqq)   # ‖x̂‖·‖q‖
                @test abs(evaluate(ScalarQuant.NormCosine(), a, Q) - evaluate(ScalarQuant.NormCosine(), a, q)) <= (B == 8 ? 1e-3 : 2e-2) * max(1.0, scale)
                @test abs(evaluate(ScalarQuant.Cosine(), a, Q) - evaluate(ScalarQuant.Cosine(), a, q)) <= (B == 8 ? 1e-3 : 2e-2)
                @test evaluate(ScalarQuant.L1(), a, Q) == evaluate(ScalarQuant.L1(), a, q)
                # the encoder dispatches both orders to the same thing
                @test evaluate(enc, Q, a) == got && evaluate(enc, a, Q) == got
            end
            # a prepared query of another width falls back to the query's coordinates
            other = SQQuery{B == 8 ? 4 : 8}(q)
            @test evaluate(ScalarQuant.SqL2(), db[1], other) == evaluate(ScalarQuant.SqL2(), db[1], q)
        end
    end

    # the integer kernels on their own, against a scalar loop, across SIMD phases
    for n in (1, 15, 16, 17, 31, 32, 33, 64, 100, 128)
        x = rand(rng, UInt8, n)
        d16 = rand(rng, Int16(-16384):Int16(16383), n)
        @test dotquery(Val(8), x, d16) == sum(Int64(x[i]) * Int64(d16[i]) for i in 1:n)
        dlo = rand(rng, Int8, n); dhi = rand(rng, Int8, n)
        @test dotquery(Val(4), x, dlo, dhi) == sum(Int64(x[i] & 0x0f) * dlo[i] + Int64(x[i] >> 4) * dhi[i] for i in 1:n)
        ds = ntuple(_ -> rand(rng, Int8, n), 4)
        @test dotquery(Val(2), x, ds...) == sum(Int64((x[i] >> (2p)) & 0x03) * ds[p+1][i] for i in 1:n, p in 0:3)
    end

    # the asymmetric graph navigates with prepared queries, and its recall is what it was
    dim = 32
    X = randn(rng, Float32, dim, 2000); Qm = randn(rng, Float32, dim, 30)
    enc = SQEncoder(ScalarQuant.SQgu8, X)
    G = AsymmetricSearchGraph(enc, sqcodes(enc)); ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))
    ids, _ = searchbatch(G, ctx, MatrixDatabase(Qm), 10)
    gold, _ = searchbatch(ExhaustiveSearch(Dist.SqL2(), MatrixDatabase(X)), GenericContext(), MatrixDatabase(Qm), 10)
    exh, _ = searchbatch(ExhaustiveSearch(enc, sqcodes(enc, X)), GenericContext(), VectorDatabase([encodequery(enc, q) for q in eachcol(Qm)]), 10)
    r(a) = sum(length(intersect(Set(a[:, j]), Set(gold[:, j]))) for j in 1:30) / (30 * 10)
    @test r(exh) >= 0.95                       # the codes' own ceiling, with prepared queries
    @test r(ids) >= r(exh) - 0.15
    # length must match the packing, like SQVec
    @test_throws ArgumentError SQQuery{4}(randn(Float32, 7))
end

@testset "ScalarQuant: GlobalQuantDatabase keeps what raw-data comparisons need (#77)" begin
    # A globally quantized vector is a per-column one whose scale happens to be shared, so this
    # database yields the same SQu*Vec types and inherits their kernels. What it adds is the
    # pair of parameters `quantize` used to throw away -- without them stored codes cannot be
    # dequantized at all -- and the per-vector code sums, which is what an order-preserving
    # cosine needs.
    dim, n = 64, 40
    X = randn(Float32, dim, n)
    mm = extrema(X)

    for (bits, VT) in ((8, ScalarQuant.SQu8.SQu8Vec), (4, ScalarQuant.SQu4.SQu4Vec), (2, ScalarQuant.SQu2.SQu2Vec))
        db = ScalarQuant.GlobalQuantDatabase(bits, X; minmax=mm)
        @test length(db) == n
        @test db[1] isa VT
        @test db isa SimilaritySearch.AbstractDatabase

        # every vector shares one scale, and dequantization is code * c + min
        @test db[1].E === db[n].E
        @test db[3][1] ≈ Float32(db.Q.matrix[1, 3] & (bits == 8 ? 0xff : bits == 4 ? 0x0f : 0x03)) * db.E.c + db.E.min

        # rebuilt from the stored codes and the pair they were made with
        reb = ScalarQuant.GlobalQuantDatabase(bits, db.Q, mm)
        @test reb.E == db.E
        @test reb.Sa == db.Sa && reb.Saa == db.Saa

        sql2 = bits == 8 ? ScalarQuant.SQu8.SqL2() : bits == 4 ? ScalarQuant.SQu4.SqL2() : ScalarQuant.SQu2.SqL2()
        for i in (1, 7, n)
            # equal scales: the exact integer path, so a vector against itself is exactly zero
            @test evaluate(sql2, db[i], db[i]) == 0f0
            # against a plain Float32 vector: the mixed kernels, no quantization of the query
            raw = Float32[db[i][t] for t in 1:dim]
            @test evaluate(sql2, db[i], raw) <= 1f-6 * max(1f0, sum(abs2, raw))
        end

        # Cosine against a Float64 evaluation of the same dequantized vectors
        for (i, j) in ((1, 2), (5, 31))
            a = Float64[db[i][t] for t in 1:dim]
            b = Float64[db[j][t] for t in 1:dim]
            truth = 1.0 - dot(a, b) / (norm(a) * norm(b))
            @test abs(evaluate(ScalarQuant.Cosine(), db[i], db[j]) - truth) <= 1f-4
        end
        @test evaluate(ScalarQuant.Cosine(), db[1], db[1]) <= 1f-6

        # a query quantized with the database's own parameters is comparable with it
        q = ScalarQuant.quantize(db, view(X, :, 5))
        @test q isa VT
        @test evaluate(sql2, db[5], q) == 0f0
    end

    @test_throws ArgumentError ScalarQuant.GlobalQuantDatabase(3, X)
    @test_throws ArgumentError ScalarQuant.GlobalQuantDatabase(16, X)
    # a dimension that does not fill its last byte used to be padded silently, and the mixed
    # kernels then read the plain query past its end; it is rejected upfront now, as the
    # per-column family always did
    @test_throws ArgumentError ScalarQuant.GlobalQuantDatabase(4, rand(Float32, 7, 5))
    @test_throws ArgumentError ScalarQuant.GlobalQuantDatabase(2, rand(Float32, 6, 5))
    @test ScalarQuant.GlobalQuantDatabase(2, rand(Float32, 8, 5)) isa ScalarQuant.GlobalQuantDatabase{2}
    @test_throws ArgumentError ScalarQuant.quantize(ScalarQuant.GlobalQuantDatabase(8, X), rand(Float32, dim + 1))

    # usable directly as an index's database
    db = ScalarQuant.GlobalQuantDatabase(8, X; minmax=mm)
    seq = ExhaustiveSearch(ScalarQuant.Cosine(), db)
    res = search(seq, GenericContext(), db[7], knnqueue(KnnSorted, 3))
    @test nearest(res).id == 7

    # the ordering the raw-code dot product could not give: on centered data, ranking by
    # Cosine must agree with exact cosine far better than chance (issue #77 measured 0.005)
    Y = randn(Float32, 32, 500); foreach(j -> normalize!(view(Y, :, j)), 1:500)
    dby = ScalarQuant.GlobalQuantDatabase(8, Y; minmax=extrema(Y))
    hits = 0
    for q in 1:20
        gold = partialsortperm([-dot(view(Y, :, i), view(Y, :, q)) for i in 1:500], 1:5)
        got = partialsortperm([evaluate(ScalarQuant.Cosine(), dby[i], dby[q]) for i in 1:500], 1:5)
        hits += length(intersect(Set(gold), Set(got)))
    end
    @test hits >= 0.7 * 20 * 5
end

@testset "ScalarQuant: the code storage is pluggable, and every backend yields the same codes (#87)" begin
    # A quantized database is its parameters plus *some* database of code vectors. The same
    # sequence of insertions through each backend must produce byte-identical codes, the same
    # per-vector sums and the same distances, which makes the one-shot MatrixDatabase build the
    # reference and every other backend a conformance check.
    dim, n = 32, 48
    X = randn(Float32, dim, n)
    mm = extrema(X)
    q = randn(Float32, dim)

    function backends(codes::Matrix{UInt8}, dir)
        nb = size(codes, 1)
        Dict(
            "matrix" => MatrixDatabase(codes),
            "block" => BlockMatrixDatabase(nb, UInt8, 3),                     # 8 items per block: crosses boundaries
            "vector" => VectorDatabase(type=Vector{UInt8}),
            "mmap" => MMapMatrixDatabase(joinpath(dir, "codes-$nb-$(rand(UInt32)).mmapdb"), nb, UInt8; capacity_bits=3),
        )
    end

    mktempdir() do dir
        for bits in (2, 4, 8), family in (:global, :pervector)
            ref = family == :global ? ScalarQuant.GlobalQuantDatabase(bits, X; minmax=mm) :
                  bits == 8 ? ScalarQuant.SQu8.quantize(X) : bits == 4 ? ScalarQuant.SQu4.quantize(X) : ScalarQuant.SQu2.quantize(X)
            @test ref isa ScalarQuant.QuantDatabase{bits}
            @test ScalarQuant.isglobal(ref) == (family == :global)
            @test ScalarQuant.codewidth(ref) == bits
            @test ref.dim == dim
            @test length(ref) == n

            empty = Matrix{UInt8}(undef, size(ref.Q.matrix, 1), 0)
            for (name, Q) in backends(empty, dir)
                # start empty, in each backend, and grow one vector at a time
                db = family == :global ? ScalarQuant.GlobalQuantDatabase(bits, Q, mm; dim) :
                     ScalarQuant.QuantDatabase{bits}(ScalarQuant.SQMinC[], Q; dim)
                @test length(db) == 0
                @test eltype(db) == ScalarQuant.SQVec{bits}
                if name == "matrix"
                    @test_throws ErrorException push_item!(db, X[:, 1])       # a MatrixDatabase does not grow
                    continue
                end
                for i in 1:(n ÷ 2)
                    push_item!(db, X[:, i])
                end
                append_items!(db, X[:, (n ÷ 2 + 1):n])                       # the rest, as a matrix
                @test length(db) == n
                @test db == ref                                               # codes and parameters, byte for byte
                @test db.Sa == ref.Sa && db.Saa == ref.Saa
                for i in (1, n ÷ 2, n)
                    @test collect(db[i].V) == collect(ref[i].V)
                    @test db[i].E == ref[i].E
                    for dist in (ScalarQuant.SqL2(), ScalarQuant.L1(), ScalarQuant.Cosine(), ScalarQuant.NormCosine())
                        @test evaluate(dist, db[i], db[1]) == evaluate(dist, ref[i], ref[1])
                    end
                    for dist in (ScalarQuant.SqL2(), ScalarQuant.L1(), ScalarQuant.NormCosine())
                        @test evaluate(dist, db[i], q) == evaluate(dist, ref[i], q)
                    end
                end
                # an already quantized vector goes in as it is
                push_item!(db, ref[3])
                @test length(db) == n + 1 && collect(db[n + 1].V) == collect(ref[3].V)
                @test_throws ArgumentError push_item!(db, randn(Float32, dim + 4))
                # and a database is searched through the ordinary interface, whatever backs it
                res = search(ExhaustiveSearch(ScalarQuant.SqL2(), db), GenericContext(), q, knnqueue(KnnSorted, 5))
                resref = search(ExhaustiveSearch(ScalarQuant.SqL2(), ref), GenericContext(), q, knnqueue(KnnSorted, 5))
                @test collect(IdView(res))[1:5] == collect(IdView(resref))[1:5]
                Q isa MMapMatrixDatabase && close(Q)
            end
        end

        # the global family refuses a vector quantized under other parameters
        db = ScalarQuant.GlobalQuantDatabase(8, BlockMatrixDatabase(dim, UInt8), mm; dim)
        other = ScalarQuant.GlobalQuantDatabase(8, X; minmax=(mm[1] - 1, mm[2] + 1))
        @test_throws ArgumentError push_item!(db, other[1])
        # an empty VectorDatabase cannot tell the dimension: `dim` is required
        @test_throws ArgumentError ScalarQuant.GlobalQuantDatabase(8, VectorDatabase(type=Vector{UInt8}), mm)
        # and the growable constructors validate the dimension the same way the batch ones do
        @test_throws ArgumentError ScalarQuant.GlobalQuantDatabase(4, BlockMatrixDatabase(3, UInt8), mm; dim=7)
    end
end

@testset "ScalarQuant: an mmap-backed quantized database survives the process (#87)" begin
    dim, n = 16, 40
    X = randn(Float32, dim, n)
    mm = extrema(X)
    mktempdir() do dir
        path = joinpath(dir, "quant.mmapdb")
        db = ScalarQuant.GlobalQuantDatabase(8, MMapMatrixDatabase(path, dim, UInt8; capacity_bits=3), mm; dim)
        append_items!(db, MatrixDatabase(X))
        @test length(db) == n
        ref = ScalarQuant.GlobalQuantDatabase(8, X; minmax=mm)
        @test db == ref
        Sa, Saa = copy(db.Sa), copy(db.Saa)
        close(db.Q)

        # reopened with only the codes and the range: the sums are recomputed
        re = ScalarQuant.GlobalQuantDatabase(8, MMapMatrixDatabase(path; read_only=true), mm)
        @test length(re) == n && re.dim == dim
        @test re == ref
        @test re.Sa == Sa && re.Saa == Saa
        res = search(ExhaustiveSearch(ScalarQuant.Cosine(), re), GenericContext(), re[7], knnqueue(KnnSorted, 3))
        @test nearest(res).id == 7
        close(re.Q)

        # reopened with the sums the caller kept: nothing is recomputed, and it keeps growing
        re2 = ScalarQuant.GlobalQuantDatabase(8, MMapMatrixDatabase(path), mm; Sa, Saa)
        @test re2 == ref
        push_item!(re2, X[:, 1])
        @test length(re2) == n + 1 && collect(re2[n + 1].V) == collect(ref[1].V)
        close(re2.Q)
    end
end

@testset "ScalarQuant: a SearchGraph grows over a quantized database (#87)" begin
    # the point of the pluggable storage: the graph inserts one item at a time, and the
    # database quantizes each on the way in with the parameters it was created with
    dim, n = 32, 600
    X = randn(Float32, dim, n)
    mm = extrema(X)
    db = ScalarQuant.GlobalQuantDatabase(4, BlockMatrixDatabase(dim ÷ 2, UInt8), mm; dim)
    G = SearchGraph(ScalarQuant.SqL2(), db)
    ctx = SearchGraphContext(; reporters=[])
    append_items!(G, ctx, MatrixDatabase(X))
    @test length(G) == n
    @test db == ScalarQuant.GlobalQuantDatabase(4, X; minmax=mm)

    # searched with a Float32 query (the mixed kernel) and with a quantized one
    seq = ExhaustiveSearch(ScalarQuant.SqL2(), db)
    hits = 0
    for j in 1:30
        q = X[:, j] .+ 0.05f0 .* randn(Float32, dim)
        gold = Set(IdView(search(seq, GenericContext(), q, knnqueue(KnnSorted, 10))))
        got = Set(IdView(search(G, ctx, q, knnqueue(KnnSorted, 10))))
        hits += length(intersect(gold, got))
        gotq = search(G, ctx, ScalarQuant.quantize(db, q), knnqueue(KnnSorted, 10))
        @test length(gotq) == 10
    end
    @test hits >= 0.8 * 300
end


@testset "prefetch_item on quantized databases" begin
    X = rand(Float32, 32, 300)
    for Q in (ScalarQuant.SQgu8, ScalarQuant.SQu4)
        enc = SQEncoder(Q, X)
        db = sqcodes(enc, X)
        @test db.Q isa BlockMatrixDatabase
        @test SimilaritySearch.prefetchable(db)
        for i in (1, 256, 257, 300)
            @test SimilaritySearch.prefetch_item(db, i) === nothing
        end
    end
end
