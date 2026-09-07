# This file is part of SimilaritySearch.jl

module BKTree

using ..SimilaritySearch
using ..SimilaritySearch:
    AbstractContext, AbstractDatabase, AbstractSearchIndex,
    AbstractKnnQueue, AbstractMetricQueue, GenericContext,
    add_distance_evaluations!, getminbatch, @BATCHES,
    AbstractReporter, AbstractObserver, OBSERVE, INFORM, @inform,
    push_item!, covradius, maxlength
import ..SimilaritySearch:
    search, index!, database, distance


export BKT, getcontext

include("bkt.jl")

end # module
