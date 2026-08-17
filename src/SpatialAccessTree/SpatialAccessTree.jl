# This file is part of SimilaritySearch.jl

module SpatialAccessTree

using ..SimilaritySearch
using ..SimilaritySearch:
    AbstractContext, AbstractDatabase, AbstractSearchIndex,
    AbstractKnnQueue, AbstractMetricQueue, GenericContext,
    PermutedSearchIndex, SubDatabase, MatrixDatabase,
    KnnSorted, IdDist, add_distance_evaluations!, add_block_evaluations!, getminbatch,
    BeamSearch, BeamSearchSpace, AbstractLog, InformativeLog, get_batch_scheduler,
    push_item!, pop_min!, covradius, maxlength, reuse!,
    check_visited_and_visit!, visit!,
    @BATCHES, @BEGIN, @BEGINBATCH, @LOOP, @ENDBATCH, @END, @batchid, @nbatches
import ..SimilaritySearch:
    search, index!, database, distance,
    optimization_space, setconfig!, runconfig, verbose, knnqueue,
    allknn_single_search!

using SearchModels: AbstractSolutionSpace, scale
import SearchModels: combine, mutate

using Random: shuffle!, shuffle, AbstractRNG
using Accessors
using Distances: evaluate

export Sat, SatInitialPartition, RandomInitialPartition,
    RandomSortSat, ProximalSortSat, DistalSortSat,
    satpermutation, satpermutation!, permutesat, getcontext,
    SatContext,
    BeamSearchSat, PruningSat, BeamSearchMultiSat, PrunParSat, BeamSearchParSat

include("sat.jl")
include("permsat.jl")
include("context.jl")
include("beamsearchsat.jl")
include("pruningsat.jl")
include("beamsearchmultisat.jl")
include("prunparsat.jl")
include("beamsearchparsat.jl")

end # module
