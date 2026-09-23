# This file is a part of SimilaritySearch.jl

using Test, SimilaritySearch, SimilaritySearch.Projections, Random, Statistics

const SQ = SimilaritySearch.ScalarQuant


"Unpacks every `nbits`-wide code held by the packed byte vector `p` (low bits first)."
function unpackcodes(p::AbstractVector{UInt8}, nbits::Int)
    percell = 8 ÷ nbits
    mask = UInt8((1 << nbits) - 1)
    [(p[cld(i, percell)] >> (nbits * ((i - 1) % percell))) & mask for i in 1:percell*length(p)]
end

@testset "SQgu2" begin
    Random.seed!(7)

    for m in (4, 8, 100, 128, 129, 260)
        X = rand(Float32, m, 5)
        minmax = (0f0, 1f0)
        Q = SQ.SQgu2.quantize(X; minmax)
        @test size(Q) == (cld(m, 4), 5)
        @test eltype(Q) == UInt8

        c = SQ.sqglobalscale(3, minmax...)
        for j in 1:5
            want = UInt8[UInt8(clamp(round(X[i, j] * c; digits=0), 0, 3)) for i in 1:m]
            @test unpackcodes(view(Q, :, j), 2)[1:m] == want
            # the single-vector method agrees with the matrix one
            @test SQ.SQgu2.quantize(view(X, :, j); minmax) == Q[:, j]
        end

        # distances read the packed codes directly: check them against a naive unpack.
        # they cover *every* slot, padding included -- which is why both operands must be
        # padded consistently (they are: the kernel zero-fills the tail of the last byte)
        a, b = view(Q, :, 1), view(Q, :, 2)
        fa, fb = Int32.(unpackcodes(a, 2)), Int32.(unpackcodes(b, 2))
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.SqL2(), a, b) ≈ Float32(sum((fa .- fb) .^ 2))
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.NormCosine(), a, b) ≈ -Float32(sum(fa .* fb))
    end

    # a coarse quantizer still has to be monotone: the code never decreases with the value
    codes = [first(unpackcodes(SQ.SQgu2.quantize([x]; minmax=(0f0, 1f0)), 2)) for x in 0f0:0.05f0:1f0]
    @test issorted(codes)
    @test extrema(codes) == (0x00, 0x03)

    # the range is clamped, not wrapped around
    @test unpackcodes(SQ.SQgu2.quantize([-10f0, 10f0, 0f0, 1f0]; minmax=(0f0, 1f0)), 2) == UInt8[0, 3, 0, 3]

    # the SIMD kernels run in `Int16` lanes over 32-byte groups, so the byte length must
    # be exercised right around that group size and past it -- these lengths land the
    # scalar tail at every offset, including none at all
    for nbytes in (1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200)
        m = 4nbytes
        A = SQ.SQgu2.quantize(rand(Float32, m); minmax=(0f0, 1f0))
        B = SQ.SQgu2.quantize(rand(Float32, m); minmax=(0f0, 1f0))
        @test length(A) == nbytes
        fa, fb = Int32.(unpackcodes(A, 2)), Int32.(unpackcodes(B, 2))
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.SqL2(), A, B) ≈ Float32(sum((fa .- fb) .^ 2))
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.NormCosine(), A, B) ≈ -Float32(sum(fa .* fb))
    end

    # adversarial, far past one accumulator block: every lane takes the maximum a byte can
    # contribute (4 fields * 3^2 == 36) on every iteration, which overflows an Int16 lane
    # unless the kernel really does widen per block
    let nbytes = 70_000            # > _U2_BLOCK (65536), so at least two blocks
        A = fill(0x00, nbytes)
        B = fill(0xff, nbytes)
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.SqL2(), A, B) == Float32(nbytes * 4 * 9)
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.NormCosine(), A, B) == 0f0
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.NormCosine(), B, B) == -Float32(nbytes * 4 * 9)
        @test SimilaritySearch.Dist.evaluate(SQ.SQgu2.SqL2(), B, B) == 0f0
    end

    # `quantize!` is the allocating `quantize` writing into a caller's buffer
    v = rand(Float32, 64)
    minmax = (0f0, 1f0)
    for (mod, cl) in ((SQ.SQgu2, cld(64, 4)), (SQ.SQgu4, cld(64, 2)), (SQ.SQgu8, 64))
        out = Vector{UInt8}(undef, cl)
        @test mod.quantize!(out, v, minmax) === out
        @test out == mod.quantize(v; minmax)
    end
end

@testset "QuantSketch and SketchedSearch" begin
    Random.seed!(11)
    dist = SimilaritySearch.Dist.L2()
    dim = 32
    n = 2^12
    X = MatrixDatabase(randn(Float32, dim, n))
    Q = MatrixDatabase(randn(Float32, dim, 30))
    ncomp = 128

    models = (
        "DistantHyperplanes" => DistantHyperplanes(dist, X, ncomp; henc=2^9, verbose=false),
        "AnchoredDistantHyperplanes" => AnchoredDistantHyperplanes(dist, X, ncomp; henc=2^9, verbose=false),
        "RandomHyperplanes" => RandomHyperplanes(dist, SubDatabase(X, rand(1:n, 2ncomp)), ncomp),
        "RandomProjections" => Projections.gaussian(dim, ncomp),
        "HadamardProjection" => HadamardProjection(dim),
    )

    @test_throws ArgumentError QuantSketch(last(models[1]), 3, X)
    @test_throws ArgumentError QuantSketch(last(models[1]), 16, X)

    for (name, model) in models
        @testset "$name" begin
            m = outdim(model)

            for nbits in (1, 2, 4, 8)
                qs = QuantSketch(model, nbits, X)
                @test sketchbits(qs) == nbits
                @test outdim(qs) == m
                @test sketchsize(qs) == (nbits == 1 ? cld(m, 64) : cld(m, 8 ÷ nbits))

                s = quantsketch(qs, X[1])
                @test length(s) == sketchsize(qs)
                @test eltype(s) == (nbits == 1 ? UInt64 : UInt8)
                # a sketch costs exactly nbits per component, whatever the model
                @test 8 * sizeof(s) >= nbits * m

                B = quantsketch(qs, X)
                @test B isa MatrixDatabase
                @test size(B.matrix) == (sketchsize(qs), n)
                # the parallel collection path and the single-object path must agree
                @test all(B[i] == quantsketch(qs, X[i]) for i in 1:50)

                # the distance must actually accept what the encoder produces
                @test SimilaritySearch.Dist.evaluate(distance(qs), B[1], B[2]) isa Float32
                @test SimilaritySearch.Dist.evaluate(distance(qs), B[1], B[1]) == 0f0
            end

            # nbits == 1 *is* bitsketch: same words, same Hamming distance
            qs1 = QuantSketch(model, 1, X)
            @test distance(qs1) isa SimilaritySearch.Dist.Bits.Hamming
            for i in 1:20
                obj = X[i]
                want = model isa Projections.RandomProjections || model isa HadamardProjection ?
                       bitsketch(model, Vector{Float32}(obj)) : bitsketch(model, obj)
                @test quantsketch(qs1, obj) == want
            end

            # sketchvalues' sign convention is what makes that identity hold
            vals = sketchvalues(model, X[1])
            @test length(vals) == m
            @test all(i -> (vals[i] >= 0) == Bool((quantsketch(qs1, X[1])[cld(i, 64)] >> ((i - 1) % 64)) & 1),
                      1:m)

            # the one-step form returns codes identical to the two-step one
            B, qs = quantsketch(model, 4, X; minmax=(-1f0, 1f0))
            @test qs.minmax == (-1f0, 1f0)
            @test B.matrix == quantsketch(QuantSketch(model, 4, X; minmax=(-1f0, 1f0)), X).matrix
        end
    end

    @testset "normalize" begin
        # hyperplane models have a natural per-component width; projections do not
        hp = last(models[1])
        @test Projections.hyperplanewidths(hp) !== nothing
        @test all(>(0), Projections.hyperplanewidths(hp))
        @test Projections.hyperplanewidths(last(models[4])) === nothing

        # normalized margins live in [-1, 1] by the triangle inequality, so a global
        # range fitted on them is meaningful across hyperplanes of very different widths
        qs = QuantSketch(hp, 4, X; normalize=true)
        @test -1f0 <= qs.minmax[1] < qs.minmax[2] <= 1f0
        raw = QuantSketch(hp, 4, X; normalize=false)
        @test isempty(raw.scale)
        @test quantsketch(qs, X[1]) != quantsketch(raw, X[1])
    end

    @testset "SketchedSearch" begin
        gold, _ = searchbatch(ExhaustiveSearch(dist, X), GenericContext(), Q, 10)
        recall(ids) = mean(length(intersect(Set(view(gold, :, j)), Set(view(ids, :, j)))) / 10
                           for j in 1:length(Q))

        model = DistantHyperplanes(dist, X, 256; henc=2^9, verbose=false)
        @test_throws ArgumentError SketchedSearch(model, 4, dist, X; factor=0)

        for nbits in (1, 2, 4, 8)
            S = SketchedSearch(model, nbits, dist, X; factor=8)
            @test length(S) == n
            @test distance(S) === dist
            @test database(S) === X
            @test database(S, 3) === X[3]

            ids, dists = searchbatch(S, GenericContext(), Q, 10)
            # the pipeline returns ids into the ORIGINAL db with EXACT distances --
            # the sketch is only used to pick candidates
            for j in 1:length(Q), i in 1:10
                ids[i, j] == 0 && continue
                @test dists[i, j] ≈ SimilaritySearch.Dist.evaluate(dist, X[ids[i, j]], Q[j])
            end
            @test all(issorted(view(dists, :, j)) for j in 1:length(Q))
            # 8x widening plus exact re-scoring should recover most of the true neighbors
            @test recall(ids) > 0.7
        end

        # single-query search honors the result queue's capacity
        S = SketchedSearch(model, 4, dist, X; factor=4)
        res = search(S, GenericContext(), Q[1], knnqueue(KnnSorted, 5))
        @test length(res) == 5

        # factor=1 disables the widening but keeps the distances exact
        S1 = SketchedSearch(model, 4, dist, X; factor=1)
        ids, dists = searchbatch(S1, GenericContext(), Q, 10)
        @test all(dists[i, j] ≈ SimilaritySearch.Dist.evaluate(dist, X[ids[i, j]], Q[j])
                  for j in 1:length(Q), i in 1:10 if ids[i, j] != 0)

        # a custom sketch-index builder is honored
        S2 = SketchedSearch(model, 4, dist, X; factor=4,
                            index=(d, B) -> SimilaritySearch.Exact.ParallelExhaustiveSearch(d, B))
        @test S2.sketches isa SimilaritySearch.Exact.ParallelExhaustiveSearch
        @test recall(first(searchbatch(S2, GenericContext(), Q, 10))) > 0.7
    end
end
