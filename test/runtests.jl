# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt


function create_vectors(n, dim, normalize=false)
    D = [rand(Float32, dim) for i in 1:n]
    if normalize
        for u in D
            normalize!(u)
        end
    end

    D
end

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
#include("testpivots.jl")
#include("testknr.jl")
include("testsearchgraph.jl")
