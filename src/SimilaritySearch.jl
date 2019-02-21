module SimilaritySearch
abstract type Index end
abstract type Result end

# export Index, DistanceType, Distance, Result
export Index, DistanceType, Result

include("distances/bits.jl")
include("distances/sets.jl")
include("distances/strings.jl")
include("distances/vectors.jl")
include("distances/cos.jl")
include("res/knn.jl")
include("nns/recall.jl")
include("nns/performance.jl")
include("indexes/pivotselection.jl")
include("indexes/seq.jl")
include("indexes/laesa.jl")
include("indexes/pivotselectiontables.jl")
include("knr/SimilarReferences.jl")
include("graph/NearNeighborGraph.jl")
end
