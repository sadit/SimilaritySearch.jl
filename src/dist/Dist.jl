module Dist

    using Distances: PreMetric, SemiMetric, Metric
    import Distances: evaluate
    export evaluate, PreMetric, SemiMetric, Metric  # reexporting

    include("vecs.jl")
    include("cos.jl")

    module Bits
        using Distances: PreMetric, SemiMetric, Metric
        import Distances: evaluate
        include("bits.jl")
    end

    module Sets
        using Distances: PreMetric, SemiMetric, Metric
        import Distances: evaluate
        include("sets.jl")
    end

    module Seqs
        using Distances: PreMetric, SemiMetric, Metric
        import Distances: evaluate
        include("seqs.jl")
    end

    module Cloud
        using Distances: PreMetric, SemiMetric, Metric
        import Distances: evaluate
        include("cloud.jl")
    end

    module Hacks
        using Distances: PreMetric, SemiMetric, Metric
        import Distances: evaluate
        include("hacks.jl")

    end

    module CastInt8Float32
        using Distances: PreMetric, SemiMetric, Metric
        import Distances: evaluate
        include("vecs-cast-i8-f32.jl")
    end

    module CastFloat32
        using Distances: PreMetric, SemiMetric, Metric
        import Distances: evaluate
        include("vecs-cast-f32.jl")
    end

end
