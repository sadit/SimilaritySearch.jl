#module Dist
import Distances: evaluate
using Distances: PreMetric, SemiMetric
export evaluate, PreMetric, SemiMetric  # reexporting

include("bits.jl")
include("sets.jl")
include("strings.jl")
include("vectors.jl")
include("cos.jl")
include("cloud.jl")
include("hacks.jl")
#end
