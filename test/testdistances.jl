# This file is a part of SimilaritySearch.jl

using SimilaritySearch, Test, LinearAlgebra

@testset "Distances" begin
    @testset "Vector Distances" begin
        u = Float32[1.0, 2.0, 3.0]
        v = Float32[4.0, 6.0, 8.0]

        dists = [
            Dist.L1(),
            Dist.L2(),
            Dist.SqL2(),
            Dist.LInfty(),
            Dist.Lp(3.0f0),
            Dist.Lp(0.5f0),
            Dist.Angle(),
            Dist.Cosine(),
        ]

        for d in dists
            val_uv = Dist.evaluate(d, u, v)
            val_vu = Dist.evaluate(d, v, u)
            val_uu = Dist.evaluate(d, u, u)

            @test val_uv isa Float32
            @test val_vu isa Float32
            @test val_uu isa Float32
            @test isapprox(val_uv, val_vu; atol=1e-5)
            @test val_uv >= 0f0
        end

        # Test specific expected distance values: u=[1,2,3] v=[4,6,8]
        @test Dist.evaluate(Dist.L1(), u, v) == 12.0f0
        @test Dist.evaluate(Dist.SqL2(), u, v) == 50.0f0
        @test Dist.evaluate(Dist.L2(), u, v) == sqrt(50.0f0)
        @test Dist.evaluate(Dist.LInfty(), u, v) == 5.0f0
        @test Dist.evaluate(Dist.L2(), u, u) == 0.0f0
        # Lp(3): (3³+4³+5³)^(1/3) = (27+64+125)^(1/3) = 216^(1/3) = 6
        @test isapprox(Dist.evaluate(Dist.Lp(3.0f0), u, v), 6.0f0; atol=1e-5)
        # Lp(0.5): (√3+2+√5)² ≈ 35.618
        @test isapprox(Dist.evaluate(Dist.Lp(0.5f0), u, v), 35.61844f0; atol=1e-3)

        # 3-4-5 right triangle: p=[3,4] q=[0,0]
        p = Float32[3.0, 4.0]
        q = Float32[0.0, 0.0]
        @test Dist.evaluate(Dist.L1(), p, q) == 7.0f0
        @test Dist.evaluate(Dist.L2(), p, q) == 5.0f0       # exact: 3-4-5
        @test Dist.evaluate(Dist.SqL2(), p, q) == 25.0f0
        @test Dist.evaluate(Dist.LInfty(), p, q) == 4.0f0
        @test isapprox(Dist.evaluate(Dist.Lp(2.0f0), p, q), 5.0f0; atol=1e-5)

        # Orthogonal unit vectors: e=[1,0,0] f=[0,1,0]
        e = Float32[1.0, 0.0, 0.0]
        f = Float32[0.0, 1.0, 0.0]
        @test Dist.evaluate(Dist.L1(), e, f) == 2.0f0
        @test isapprox(Dist.evaluate(Dist.L2(), e, f), sqrt(2.0f0); atol=1e-6)
        @test Dist.evaluate(Dist.SqL2(), e, f) == 2.0f0
        @test Dist.evaluate(Dist.LInfty(), e, f) == 1.0f0
        # Angle between orthogonal vectors = π/2 (radians)
        @test isapprox(Dist.evaluate(Dist.Angle(), e, f), Float32(π/2); atol=1e-5)
        # Cosine distance for orthogonal vectors = 1 - cos(π/2) = 1
        @test Dist.evaluate(Dist.Cosine(), e, f) == 1.0f0
        # NormAngle and NormCosine (already unit vectors)
        @test isapprox(Dist.evaluate(Dist.NormAngle(), e, f), Float32(π/2); atol=1e-5)
        @test Dist.evaluate(Dist.NormCosine(), e, f) == 1.0f0
        # Self-distance = 0
        @test Dist.evaluate(Dist.NormCosine(), e, e) == 0.0f0
        @test isapprox(Dist.evaluate(Dist.NormAngle(), e, e), 0.0f0; atol=1e-6)

        # Normalized angle / cosine (non-unit u,v)
        un = normalize(u)
        vn = normalize(v)
        @test Dist.evaluate(Dist.NormAngle(), un, vn) isa Float32
        @test Dist.evaluate(Dist.NormCosine(), un, vn) isa Float32
        @test isapprox(Dist.evaluate(Dist.NormCosine(), un, un), 0.0f0; atol=1e-6)

        # DistanceWithIdentifiers wrapper
        db = MatrixDatabase(hcat(u, v))
        dw = Dist.Hacks.DistanceWithIdentifiers(Dist.SqL2(), db)
        @test Dist.evaluate(dw, 1, 2) == 50.0f0
        @test Dist.evaluate(dw, 1, 2) isa Float32
    end

    @testset "Sequence Distances" begin
        seq1 = [1, 2, 3, 4]
        seq2 = [1, 2, 5, 6]

        seq_dists = [
            Dist.Seqs.CommonPrefix(),
            Dist.Seqs.Levenshtein(),
            Dist.Seqs.LCS(),
            Dist.Seqs.Hamming()
        ]

        for d in seq_dists
            val_12 = Dist.evaluate(d, seq1, seq2)
            val_21 = Dist.evaluate(d, seq2, seq1)
            val_11 = Dist.evaluate(d, seq1, seq1)

            @test val_12 isa Float32
            @test val_21 isa Float32
            @test val_11 isa Float32
            @test val_12 >= 0f0
        end

        # seq1=[1,2,3,4] seq2=[1,2,5,6]: 2 common prefix elements, min=4 → 1-2/4=0.5
        @test Dist.evaluate(Dist.Seqs.CommonPrefix(), seq1, seq2) == 0.5f0
        @test Dist.evaluate(Dist.Seqs.Levenshtein(), seq1, seq2) == 2.0f0
        # LCS([1,2,3,4],[1,2,5,6])=[1,2] length=2 → dist = 4+4-2*2=4
        @test Dist.evaluate(Dist.Seqs.LCS(), seq1, seq2) == 4.0f0
        # Hamming: positions 3,4 differ → 2
        @test Dist.evaluate(Dist.Seqs.Hamming(), seq1, seq2) == 2.0f0

        # Second pair: seq3=[1,2,3,4,5] seq4=[1,2,3,6,7]
        seq3 = [1, 2, 3, 4, 5]
        seq4 = [1, 2, 3, 6, 7]
        # CommonPrefix: 3 common elements, min(5,5)=5 → 1-3/5 = 0.4 (Float32 ≈ 0.39999998)
        @test isapprox(Dist.evaluate(Dist.Seqs.CommonPrefix(), seq3, seq4), 0.4f0; atol=1e-6)
        # Hamming: positions 4,5 differ → 2
        @test Dist.evaluate(Dist.Seqs.Hamming(), seq3, seq4) == 2.0f0
        # LCS([1,2,3,4,5],[1,2,3,6,7])=[1,2,3] length=3 → dist = 5+5-2*3=4
        @test Dist.evaluate(Dist.Seqs.LCS(), seq3, seq4) == 4.0f0
        # Levenshtein: replace 4→6 and 5→7 → 2 edits
        @test Dist.evaluate(Dist.Seqs.Levenshtein(), seq3, seq4) == 2.0f0
        # Self-distances = 0
        @test Dist.evaluate(Dist.Seqs.Levenshtein(), seq3, seq3) == 0.0f0
        @test Dist.evaluate(Dist.Seqs.LCS(), seq3, seq3) == 0.0f0
    end

    @testset "Set Distances" begin
        set1 = [1, 2, 3, 4]
        set2 = [3, 4, 5, 6]

        set_dists = [
            Dist.Sets.Jaccard(),
            Dist.Sets.Dice(),
            Dist.Sets.Intersection(),
            Dist.Sets.RogersTanimoto(10),
            Dist.Sets.CosineSet()
        ]

        for d in set_dists
            val_12 = Dist.evaluate(d, set1, set2)
            val_21 = Dist.evaluate(d, set2, set1)
            val_11 = Dist.evaluate(d, set1, set1)

            @test val_12 isa Float32
            @test val_21 isa Float32
            @test val_11 isa Float32
            @test val_12 >= 0f0
        end

        # set1=[1,2,3,4] set2=[3,4,5,6]: inter={3,4}=2, union={1..6}=6, max=4
        @test Dist.evaluate(Dist.Sets.Jaccard(), set1, set1) == 0.0f0
        @test Dist.evaluate(Dist.Sets.Dice(), set1, set1) == 0.0f0
        # Intersection: 1 - 2/max(4,4) = 0.5
        @test Dist.evaluate(Dist.Sets.Intersection(), set1, set2) == 0.5f0
        # Jaccard(set1,set2): 1 - 2/6 = 2/3
        @test isapprox(Dist.evaluate(Dist.Sets.Jaccard(), set1, set2), 1.0f0 - 2.0f0/6.0f0; atol=1e-6)
        # Dice(set1,set2): 1 - 2*2/(4+4) = 0.5
        @test Dist.evaluate(Dist.Sets.Dice(), set1, set2) == 0.5f0
        # CosineSet(set1,set2): 1 - 2/(√4*√4) = 1 - 2/4 = 0.5
        @test Dist.evaluate(Dist.Sets.CosineSet(), set1, set2) == 0.5f0
        # RogersTanimoto(10) set1=[1,2,3,4] x set2=[3,4,5,6]:
        # tt=|{3,4}|=2, tf=|set1\set2|=|{1,2}|=2, ft=|set2\set1|=|{5,6}|=2, ff=σ-tt-tf-ft=10-2-2-2=4
        # → 1-(tt+ff)/(tt+ff+2*(tf+ft)) = 1-(2+4)/(2+4+2*4) = 1-6/14 = 4/7
        @test isapprox(Dist.evaluate(Dist.Sets.RogersTanimoto(10), set1, set2), 4.0f0/7.0f0; atol=1e-6)

        # Second pair: set3=[1,3,5] set4=[1,2,3]: inter={1,3}=2, union={1,2,3,5}=4
        set3 = [1, 3, 5]
        set4 = [1, 2, 3]
        # Jaccard: 1 - 2/4 = 0.5
        @test Dist.evaluate(Dist.Sets.Jaccard(), set3, set4) == 0.5f0
        # Dice: 1 - 2*2/(3+3) = 1/3
        @test isapprox(Dist.evaluate(Dist.Sets.Dice(), set3, set4), 1.0f0/3.0f0; atol=1e-6)
        # Intersection: 1 - 2/max(3,3) = 1/3
        @test isapprox(Dist.evaluate(Dist.Sets.Intersection(), set3, set4), 1.0f0/3.0f0; atol=1e-6)
        # CosineSet: 1 - 2/(√3*√3) = 1/3
        @test isapprox(Dist.evaluate(Dist.Sets.CosineSet(), set3, set4), 1.0f0/3.0f0; atol=1e-6)
    end

    @testset "Binary / Bit Distances" begin
        # b1=[0b1010,0b1100] b2=[0b1000,0b1100]: XOR=[0b0010,0b0000] → 1 differing bit
        b1 = UInt64[0b1010, 0b1100]
        b2 = UInt64[0b1000, 0b1100]

        @test Dist.evaluate(Dist.Bits.Hamming(), b1, b2) isa Float32
        @test Dist.evaluate(Dist.Bits.RogersTanimoto(), b1, b2) isa Float32
        @test Dist.evaluate(Dist.Bits.RussellRao(), b1, b2) isa Float32
        @test Dist.evaluate(Dist.Bits.Hamming(), b1, b1) == 0.0f0
        # Hamming: 1 bit differs across both words
        @test Dist.evaluate(Dist.Bits.Hamming(), b1, b2) == 1.0f0
        # RussellRao: tt=3 ones shared (bits 1,3 in word1 → 1 shared; bits 2,3 in word2 → 2 shared)
        #   1 - 3/(2*64) = 1 - 3/128 = 125/128 = 0.9765625
        @test Dist.evaluate(Dist.Bits.RussellRao(), b1, b2) == 0.9765625f0
        # RogersTanimoto: 1 mismatch bit → 1-127/129 ≈ 0.01550
        @test isapprox(Dist.evaluate(Dist.Bits.RogersTanimoto(), b1, b2), 1.0f0 - 127.0f0/129.0f0; atol=1e-6)

        # Scalar UInt64: b3=0b11110000 vs b4=0b00001111 — all 8 bits flip
        b3 = UInt64(0b1111_0000)
        b4 = UInt64(0b0000_1111)
        @test Dist.evaluate(Dist.Bits.Hamming(), b3, b4) == 8.0f0
        @test Dist.evaluate(Dist.Bits.Hamming(), b3, b3) == 0.0f0
        @test Dist.evaluate(Dist.Bits.Hamming(), b3, b4) isa Float32
    end
end
