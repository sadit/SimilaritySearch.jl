# This file is a part of SimilaritySearch.jl

module KCenters

using ..SimilaritySearch
using Distances: evaluate, SemiMetric
using Random: shuffle!, shuffle

# explicitly import internal/unexported names used by the algorithms
import ..SimilaritySearch:
    AbstractDatabase,
    GenericContext,
    ExhaustiveSearch,
    SubDatabase,
    knnqueue,
    KnnSorted,
    IdView,
    search,
    distance_evaluations,
    searchbatch!,
    getminbatch,
    @BATCHES,
    @BEGIN,
    @BEGINBATCH,
    @LOOP,
    @ENDBATCH,
    @END,
    @nbatches,
    @batchid

include("fft.jl")
include("dnet.jl")
include("rand.jl")
include("multirand.jl")

# We export these so that when the parent module uses this submodule, it can re-export them
export fft, dnet, randsel, multirandsel

end
