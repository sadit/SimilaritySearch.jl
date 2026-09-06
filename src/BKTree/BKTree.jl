# This file is part of SimilaritySearch.jl

module BKTree

using ..SimilaritySearch
using ..SimilaritySearch:
    AbstractContext, AbstractDatabase, AbstractSearchIndex,
    AbstractKnnQueue, AbstractMetricQueue, GenericContext,
    add_distance_evaluations!,
    AbstractReporter, AbstractObserver, OBSERVE, INFORM, @inform,
    push_item!, covradius, maxlength
import ..SimilaritySearch:
    search, index!, database, distance

using Random: AbstractRNG
import Random

export BKT, getcontext

include("bktree.jl")

end # module
