#module Dist
    import Distances: evaluate, SemiMetric
    export evaluate, SemiMetric  # reexporting

    include("bits.jl")
    include("sets.jl")
    include("strings.jl")
    include("vectors.jl")
    include("cos.jl")
    include("cloud.jl")
    include("hacks.jl")
    @static if VERSION < v"1.11"
        include("turboed.jl")
    end
#end
