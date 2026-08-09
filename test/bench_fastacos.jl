# This file is a part of SimilaritySearch.jl
# Benchmark and correctness check for fastacos vs clamp_acos.
#
# Run with:
#   julia -t auto --project=. test/bench_fastacos.jl

using SimilaritySearch, Test, BenchmarkTools, LinearAlgebra

const fastacos   = SimilaritySearch.Dist.fastacos
const clamp_acos = SimilaritySearch.Dist.clamp_acos

# ---------------------------------------------------------------------------
# Correctness: both must agree on interior and boundary points
# ---------------------------------------------------------------------------
@testset "fastacos vs clamp_acos correctness" begin
    interior = Float32[-0.999f0, -0.5f0, -0.1f0, 0.0f0, 0.1f0, 0.5f0, 0.999f0]
    for d in interior
        @test isapprox(fastacos(d), clamp_acos(d); atol=1e-6)
    end

    # Boundary and beyond: results must be finite Float32
    @test fastacos(-1.0f0)  == Float32(π)
    @test fastacos( 1.0f0)  == 0.0f0
    @test fastacos( 0.0f0)  == Float32(π/2)
    @test clamp_acos(-1.0f0) ≈ Float32(π)  atol=1e-6
    @test clamp_acos( 1.0f0) ≈ 0.0f0       atol=1e-6

    # Out-of-range inputs: both saturate at boundaries
    @test fastacos(-1.5f0) == Float32(π)
    @test fastacos( 1.5f0) == 0.0f0
    @test clamp_acos(-1.5f0) ≈ Float32(π) atol=1e-6
    @test clamp_acos( 1.5f0) ≈ 0.0f0      atol=1e-6

    # Return type
    @test fastacos(0.5f0)   isa Float32
    @test clamp_acos(0.5f0) isa Float32
    @test fastacos(0.5)     isa Float32   # AbstractFloat dispatch
    @test clamp_acos(0.5)   isa Float32
end

# ---------------------------------------------------------------------------
# Benchmark: three realistic input distributions
# ---------------------------------------------------------------------------

N = 100_000

# (1) Uniform random in [-1, 1]  — typical interior values
xs_uniform = rand(Float32, N) .* 2.0f0 .- 1.0f0

# (2) Near-boundary mix: 50% values >= 0.999 or <= -0.999
xs_boundary = [i % 2 == 0 ? rand(Float32)*0.001f0 + 0.999f0 : -(rand(Float32)*0.001f0 + 0.999f0)
               for i in 1:N]

# (3) Slightly out-of-range (can arise from FP error): values in [0.9999, 1.001]
xs_oor = rand(Float32, N) .* 0.002f0 .+ 0.9999f0

function run_fastacos(xs)
    s = 0.0f0
    @inbounds for x in xs
        s += fastacos(x)
    end
    s
end

function run_clamp_acos(xs)
    s = 0.0f0
    @inbounds for x in xs
        s += clamp_acos(x)
    end
    s
end

println("\n=== Benchmark: uniform random inputs in [-1, 1] ===")
@btime run_fastacos($xs_uniform)
@btime run_clamp_acos($xs_uniform)

println("\n=== Benchmark: near-boundary inputs (+/-[0.999, 1.0]) ===")
@btime run_fastacos($xs_boundary)
@btime run_clamp_acos($xs_boundary)

println("\n=== Benchmark: out-of-range inputs [0.9999, 1.001] ===")
@btime run_fastacos($xs_oor)
@btime run_clamp_acos($xs_oor)
