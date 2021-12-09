# This file is a part of SimilaritySearch.jl

using LinearAlgebra

function create_sequence(dim, sort)
    s = rand(1:10, dim)
    if sort
        sort!(s)
        s = unique(s)
    end

    s
end

#include("testresults.jl")
#include("testseq.jl")
include("testsearchgraph.jl")
