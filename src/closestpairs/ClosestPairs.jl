# This file is a part of SimilaritySearch.jl

module ClosestPairs

using ..SimilaritySearch
using ..SimilaritySearch: getvstate, visit!, neighbors
using Distances: PreMetric
using Accessors

export closestpair, bichromatic_closestpair, closestpairs, bichromatic_kclosestpairs

include("bichromaticclosestpair.jl")
include("bichromaticclosestpairs.jl")
include("closestpair.jl")
include("closestpairs.jl")
include("datasetwrapper.jl")

end # module
