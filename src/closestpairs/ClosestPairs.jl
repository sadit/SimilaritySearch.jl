# This file is a part of SimilaritySearch.jl

module ClosestPairs

using ..SimilaritySearch
using ..SimilaritySearch: getvstate, visit!, neighbors
using Distances: PreMetric
using Accessors

export closestpair, bichromatic_closestpair

include("bichromaticclosestpair.jl")
include("closestpair.jl")
include("datasetwrapper.jl")

end # module
