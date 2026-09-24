# This file is a part of SimilaritySearch.jl
module ScalarQuant

using Distances: PreMetric, SemiMetric, Metric
using Statistics: quantile
using StatsBase
import Distances: evaluate
using ..SimilaritySearch: AbstractDatabase, getminbatch, Dist, @BATCHES
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

"""
    sqglobalscale(levels::Integer, min, max)

The scale factor shared by every *global* quantizer (`SQgu2`/`SQgu4`/`SQgu8`): maps the
range `[min, max]` onto the `levels + 1` integer codes `0:levels` as
`code = round(clamp((x - min) * c, 0, levels))`. The `1e-6` in the denominator keeps a
degenerate (`min == max`) range from producing `Inf`.

# Arguments
- `levels`: the largest code the target width can hold (`3`, `15` or `255`)
- `min`, `max`: the global value range being mapped
"""
sqglobalscale(levels::Integer, min, max) = Float32(levels / (max - min + 1e-6))

include("gu8.jl")
include("gu4.jl")
include("gu2.jl")
include("u8.jl")
include("u4.jl")
include("u2.jl")
include("gdb.jl")

end