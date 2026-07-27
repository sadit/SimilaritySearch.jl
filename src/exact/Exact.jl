# This file is a part of SimilaritySearch.jl

module Exact

using ..SimilaritySearch

using ..SimilaritySearch:
    AbstractContext, AbstractDatabase, AbstractKnn, AbstractSearchIndex,
    GenericContext, PreMetric, SemiMetric, Metric, AbstractDatabase, VectorDatabase,
    add_distance_evaluations!,
    IdDist, LOG

import ..SimilaritySearch:
    search, push_item!, append_items!, index!, distance, database, Dist

using Polyester

include("sequential-exhaustive.jl")
include("parallel-exhaustive.jl")
include("basket-list.jl")

end # module
