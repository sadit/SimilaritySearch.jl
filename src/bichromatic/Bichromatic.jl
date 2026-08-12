# This file is a part of SimilaritySearch.jl

module Bichromatic

using ..SimilaritySearch
using ..SimilaritySearch: getvstate, visit!, neighbors, evaluate
using Distances: PreMetric
using Accessors
using Statistics: quantile

export closestpair, bichromatic_closestpair, closestpairs, bichromatic_kclosestpairs, bichromatic_metricjoin

include("bichromaticclosestpair.jl")
include("bichromaticclosestpairs.jl")
include("closestpair.jl")
include("closestpairs.jl")
include("datasetwrapper.jl")
include("bichromaticmetricjoin.jl")

end # module
