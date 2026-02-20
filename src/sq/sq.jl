# This file is a part of SimilaritySearch.jl
module ScalarQuant

using Distances: PreMetric, SemiMetric, Metric
import Distances: evaluate
using ..SimilaritySearch: AbstractDatabase
#using ..Dist: fastacos

struct SQMinC
    min::Float32
    c::Float32
end

include("u8.jl")
include("u4.jl")

end