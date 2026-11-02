# This file is part of SimilaritySearch.jl

"""
    Anni

The singleton indexes of Tellez, Ruiz and Chávez, *Singleton indexes for nearest neighbor
search*, Information Systems 60 (2016) 50--68 (doi:10.1016/j.is.2016.03.003), first written for
the C# library natix: [`ANNI`](@ref), a pivot table where every object keeps only the distance to
its nearest center, and [`NANNI`](@ref), the same partition stored by cluster. Both are exact
indexes for metrics, built incrementally by promoting random objects to centers, and both tune
their number of centers on a sample of queries.
"""
module Anni

using ..SimilaritySearch
using ..SimilaritySearch:
    AbstractContext, AbstractDatabase, AbstractSearchIndex,
    AbstractKnnQueue, AbstractMetricQueue, GenericContext,
    MatrixDatabase, VectorDatabase, SubDatabase, PermutedSearchIndex,
    add_distance_evaluations!, getminbatch, beginbatch, verbose,
    push_item!, covradius,
    @BATCHES, @BEGIN, @BEGINBATCH, @LOOP, @ENDBATCH, @END, @batchid, @nbatches
import ..SimilaritySearch:
    search, index!, database, distance

using Random: AbstractRNG, default_rng, randperm

export ANNI, NANNI, getcontext, annipermutation, permuteanni, nannipermutation, permutenanni

include("anni.jl")
include("nanni.jl")

end # module
