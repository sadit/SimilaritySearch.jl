# This file is part of SimilaritySearch.jl

"""
    ExtremePivots

The extreme pivots of Ruiz, Chávez, Ruiz and Tellez, *Extreme pivots: a pivot selection strategy
for faster metric search*, Knowledge and Information Systems (2020),
doi:10.1007/s10115-019-01423-5, first written for the C# library natix: [`EPTable`](@ref), a
pivot table where every object is filtered, in each of a few groups, by the single pivot that
sees it farthest from its mean distance, and [`NEP`](@ref), its node-based version, where the
nearest pivot of every object defines its cell and its far pivots filter it inside the cell.
"""
module ExtremePivots

using ..SimilaritySearch
using ..SimilaritySearch:
    AbstractContext, AbstractDatabase, AbstractSearchIndex,
    AbstractMetricQueue, GenericContext,
    MatrixDatabase, VectorDatabase, SubDatabase, PermutedSearchIndex,
    add_distance_evaluations!, getminbatch, beginbatch, verbose,
    push_item!, covradius,
    @BATCHES, @BEGIN, @BEGINBATCH, @LOOP, @ENDBATCH, @END, @batchid, @nbatches
import ..SimilaritySearch:
    search, index!, database, distance

using Random: AbstractRNG, default_rng, randperm

export EPTable, NEP, getcontext, eptpermutation, permuteept, neppermutation, permutenep

include("ept.jl")
include("nep.jl")

end # module
