# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

module SimilaritySearch
abstract type Index end
abstract type AbstractSearchContext end
# abstract type AbstractMultithreadedSearchContext <: AbstractSearchContext end

import Distances: evaluate, PreMetric
export AbstractSearchContext, PreMetric, evaluate

"""
    search(searchctx::AbstractSearchContext, q, k::Integer)

This is the most generic search function. It calls almost all implementations whenever an integer k is given.
    
"""
function search(searchctx::AbstractSearchContext, q, k::Integer=0)
    empty!(searchctx.res, k)
    search(searchctx, q, searchctx.res)
end

# function search(searchctx::AbstractMultithreadedSearchContext, q, k::Integer=0)
#     i = Threads.threadid()
#     empty!(searchctx.reslist[i], k)
#     search(searchctx, q, searchctx.reslist[i])
# end


include("distances/bits.jl")
include("distances/sets.jl")
include("distances/strings.jl")
include("distances/vectors.jl")
include("distances/cos.jl")
include("utils/knn.jl")

include("utils/perf.jl")
include("indexes/pivotselection.jl")
include("indexes/seq.jl")
include("indexes/pivottable.jl")
include("indexes/pivotselectiontables.jl")
include("indexes/kvp.jl")

#include("knr/knr.jl")
include("graph/graph.jl")
# include("utils/aknn.jl")
# include("utils/classification.jl")
end
