using SimilaritySearch
using Test

@testset "bit ops" begin
    u = zero(UInt128)
    v = setbit(u, 0)
    w = setbit(u, 1)
    @test v == 1 && w == 2
    v = setbit(v, 1)
    @test v == 3
    v = resetbit(v, 1)
    @test v == 1
end
