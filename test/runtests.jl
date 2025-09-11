# This file is a part of SimilaritySearch.jl
using SimilaritySearch, LinearAlgebra
#using JET


if VERSION == v"1.10"
    using Aqua
    Aqua.test_all(SimilaritySearch, ambiguities=false)
    Aqua.test_ambiguities([SimilaritySearch])
end

function create_sequence(dim, sort)
    s = rand(1:10, dim)
    if sort
        sort!(s)
        s = unique(s)
    end

    s
end

include("testdb.jl")
include("testresults.jl")
include("testseq.jl")
include("testallknn.jl")
include("testhsp.jl")
include("testneardup.jl")
include("testfft.jl")
include("testadj.jl")
include("testclosestpair.jl")
include("testsearchgraph.jl")
