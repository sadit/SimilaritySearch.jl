# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

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
