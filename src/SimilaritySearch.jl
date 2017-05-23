module SimilaritySearch
abstract type Index end
abstract type Result end

# export Index, DistanceType, Distance, Result
export Index, DistanceType, Result, fromjson, save, load
import JSON

include("distances/bits.jl")
include("distances/sets.jl")
include("distances/strings.jl")
include("distances/vectors.jl")
include("distances/cos.jl")
include("distances/documents.jl")
include("distances/rbow.jl")
include("distances/hbow.jl")
include("res/knn.jl")
include("nns/io.jl")
include("nns/recall.jl")
include("nns/performance.jl")
include("indexes/pivotselection.jl")
include("indexes/seq.jl")
include("indexes/laesa.jl")
include("indexes/pivotselectiontables.jl")
include("indexes/kvp.jl")
include("indexes/knr.jl")
include("indexes/graph.jl")
end
