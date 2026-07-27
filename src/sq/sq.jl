# This file is a part of SimilaritySearch.jl
module ScalarQuant

using Distances: PreMetric, SemiMetric, Metric
using Polyester
using Statistics: quantile
using StatsBase
import Distances: evaluate
using ..SimilaritySearch: AbstractDatabase, getminbatch
#using ..Dist: fastacos

"""
    SQMinC(min::Float32, c::Float32)

Internal helper struct that stores the per-vector dequantization parameters used by
the scalar quantization schemes in `ScalarQuant` (i.e., `SQu2`, `SQu4`, `SQu8`). Given a
quantized (integer) coordinate `q`, the corresponding approximate original value is
recovered as `q * c + min`.
"""
struct SQMinC
    min::Float32
    c::Float32
end

include("gu8.jl")
include("u8.jl")
include("u4.jl")
include("u2.jl")

end