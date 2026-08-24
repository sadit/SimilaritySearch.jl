# This file is a part of SimilaritySearch.jl

"""
    module Selection

Algorithms that pick a subset of a database to stand for the whole of it, and the two result
types they share.

They come in two dual shapes, which is the thing to know before reading any of them:

- **fix the count, let the radius fall out** -- [`fft`](@ref), [`dnet`](@ref),
  [`randsel`](@ref), [`multirandsel`](@ref) are told how many centers to pick, and how well
  those centers cover the database is whatever it turns out to be. They return a
  [`CenterSelection`](@ref).
- **fix the radius, let the count fall out** -- [`neardup`](@ref) is told how close is too
  close, and the number of survivors is whatever it turns out to be. It returns a
  [`NearDupSelection`](@ref).

Both results name the same things the same way ([`AbstractSelection`](@ref)), so code that
reads one reads the other.
"""
module Selection

using ..SimilaritySearch
using Distances: evaluate, SemiMetric, PreMetric
using Random: shuffle!, shuffle

# explicitly import internal/unexported names used by the algorithms
import ..SimilaritySearch:
    AbstractReporter,
    InformativeLog,
    INFORM,
    @inform,
    verbose,
    AbstractSearchIndex,
    AbstractContext,
    AbstractDatabase,
    VectorDatabase,
    GenericContext,
    ExhaustiveSearch,
    SubDatabase,
    knnqueue,
    KnnSorted,
    IdView,
    search,
    searchbatch!,
    append_items!,
    distance,
    nearest,
    reuse!,
    distance_evaluations,
    block_evaluations,
    add_distance_evaluations!,
    getminbatch,
    @BATCHES,
    @BEGIN,
    @BEGINBATCH,
    @LOOP,
    @ENDBATCH,
    @END,
    @nbatches,
    @batchid

include("types.jl")
include("fft.jl")
include("dnet.jl")
include("rand.jl")
include("multirand.jl")
include("neardup.jl")

# We export these so that when the parent module uses this submodule, it can re-export them
export fft, dnet, randsel, multirandsel, neardup,
       AbstractSelection, CenterSelection, NearDupSelection

end
