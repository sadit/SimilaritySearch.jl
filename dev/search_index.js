var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SimilaritySearch","category":"page"},{"location":"#SimilaritySearch.jl","page":"Home","title":"SimilaritySearch.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SimilaritySearch.jl is a library for nearest neighbor search. In particular, it contains the implementation for SearchGraph:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Tellez, E. S., Ruiz, G., Chavez, E., & Graff, M.A scalable solution to the nearest neighbor search problem through local-search methods on neighbor graphs. Pattern Analysis and Applications, 1-15.","category":"page"},{"location":"","page":"Home","title":"Home","text":"@article{tellezscalable,\n  title={A scalable solution to the nearest neighbor search problem through local-search methods on neighbor graphs},\n  author={Tellez, Eric S and Ruiz, Guillermo and Chavez, Edgar and Graff, Mario},\n  journal={Pattern Analysis and Applications},\n  pages={1--15},\n  publisher={Springer}\n}","category":"page"},{"location":"#Installing-SimilaritySearch","page":"Home","title":"Installing SimilaritySearch","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"You may install the package as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add SimilaritySearch","category":"page"},{"location":"","page":"Home","title":"Home","text":"also, you can run the set of tests as fol","category":"page"},{"location":"","page":"Home","title":"Home","text":"] test SimilaritySearch","category":"page"},{"location":"#Using-the-library","page":"Home","title":"Using the library","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Please see examples directory of this repository. Here you will find a list of Pluto's notebooks that exemplifies its usage.","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SimilaritySearch]","category":"page"},{"location":"#SimilaritySearch.ExhaustiveSearch","page":"Home","title":"SimilaritySearch.ExhaustiveSearch","text":"ExhaustiveSearch(dist::PreMetric, db::AbstractVector, knn::KnnResult)\nExhaustiveSearch(dist::PreMetric, db::AbstractVector, k::Integer)\n\nDefines an exhaustive search\n\n\n\n\n\n","category":"type"},{"location":"#SimilaritySearch.PivotedSearch","page":"Home","title":"SimilaritySearch.PivotedSearch","text":"PivotedSeach(index::PivotTable, db::AbstractVector, dist::PreMetric, knn::KnnResult)\nPivotedSeach(index::PivotTable, db::AbstractVector, dist::PreMetric, k::Integer=10)\n\nDefines a search index for Pivot tables\n\n\n\n\n\n","category":"type"},{"location":"#SimilaritySearch.PivotedSearch-Union{Tuple{T}, Tuple{PreMetric,AbstractArray{T,1},AbstractArray{T,1}}} where T","page":"Home","title":"SimilaritySearch.PivotedSearch","text":"PivotedSearch(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, pivots::Vector{T})\nPivotedSearch(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, numpivots::Integer)\n\nCreates a PivotTable index with the given pivots. If the number of pivots is specified, then they will be randomly selected from the dataset.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.SearchGraph-Tuple{PreMetric,AbstractArray{T,1} where T}","page":"Home","title":"SimilaritySearch.SearchGraph","text":"SearchGraph(dist::PreMetric, db::AbstractVector;\n    search_algo::LocalSearchAlgorithm=BeamSearch(),\n    neighborhood_algo::NeighborhoodAlgorithm=LogNeighborhood(),\n    automatic_optimization=false,\n    recall=0.9,\n    ksearch=10,\n    tol=0.001,\n    verbose=true)\n\nCreates a SearchGraph object, i.e., an index to perform approximate search on db using the given search and neighbohood strategies. If automatic_optimization is true, then the structure tries to reach the given recall under the given ksearch.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.SearchGraphOptions","page":"Home","title":"SimilaritySearch.SearchGraphOptions","text":"SearchGraphOptions\n\nDefines a number of options for the SearchGraph\n\n\n\n\n\n","category":"type"},{"location":"#Base.empty!","page":"Home","title":"Base.empty!","text":"empty!(res::KnnResult)\nempty!(res::KnnResult, k::Integer)\n\nClears the content of the result pool. If k is given then the size of the pool is changed; the internal buffer is adjusted as needed (only grows).\n\n\n\n\n\n","category":"function"},{"location":"#Base.first-Tuple{KnnResult}","page":"Home","title":"Base.first","text":"first(p::KnnResult)\n\nReturn the first item of the result set, the closest item\n\n\n\n\n\n","category":"method"},{"location":"#Base.last-Tuple{KnnResult}","page":"Home","title":"Base.last","text":"last(p::KnnResult)\n\nReturns the last item of the result set\n\n\n\n\n\n","category":"method"},{"location":"#Base.length-Tuple{KnnResult}","page":"Home","title":"Base.length","text":"length(p::KnnResult)\n\nlength returns the number of items in the result set\n\n\n\n\n\n","category":"method"},{"location":"#Base.popfirst!-Tuple{KnnResult}","page":"Home","title":"Base.popfirst!","text":"popfirst!(p::KnnResult)\n\nRemoves and returns the nearest neeighboor from the pool, an O(length(p.pool)) operation\n\n\n\n\n\n","category":"method"},{"location":"#Base.push!-Tuple{SearchGraph,Any}","page":"Home","title":"Base.push!","text":"push!(index::SearchGraph, item)\n\nAppends item into the index.\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{AngleDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::AngleDistance, a, b)\n\nComputes the angle  between twovectors.\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{CosineDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::CosineDistance, a, b)\n\nComputes the cosine distance between two vectors. Please use AngleDistance if you are expecting a metric function (cosine_distance is a faster alternative whenever the triangle inequality is not needed)\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{DiceDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::DiceDistance, a, b)\n\nComputes the Dice's distance of a and b both sets specified as sorted vectors.\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{GenericLevenshteinDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(GenericLevenshteinDistance, a, b)::Int\n\nComputes the edit distance between two strings, this is a low level function\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{IntersectionDissimilarity,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::IntersectionDissimilarity, a, b)\n\n(a, b)\n\nUses the intersection as a distance function (non-metric)\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{L1Distance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(L1Distance, a, b)\n\nComputes the Manhattan's distance between a and b\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{L2Distance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(L2Distance, a, b)\n\nComputes the Euclidean's distance betweem a and b\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{LInftyDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::LInftyDistance, a, b)\n\nComputes the max or Chebyshev'se distance\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{LpDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(lp::LpDistance, a, b)\n\nComputes generic Minkowski's distance\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{NormalizedAngleDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::AngleDistance, a, b)\n\nComputes the angle  between twovectors. It supposes that all vectors are normalized (see normalize! function)\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{NormalizedCosineDistance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::NormalizedCosineDistance, a, b)\n\nComputes the cosine distance between two vectors, it expects normalized vectors (see normalize! method). Please use NormalizedAngleDistance if you are expecting a metric function (cosine_distance is a faster alternative whenever the triangle inequality is not needed)\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Tuple{SqL2Distance,Any,Any}","page":"Home","title":"Distances.evaluate","text":"evaluate(::SqL2Distance, a, b)\n\nComputes the squared Euclidean's distance between a and b\n\n\n\n\n\n","category":"method"},{"location":"#Distances.evaluate-Union{Tuple{T}, Tuple{BinaryHammingDistance,T,T}} where T<:Unsigned","page":"Home","title":"Distances.evaluate","text":"evaluate(::BinaryHammingDistance, a, b)::Float64\nevaluate(::BinaryHammingDistance, a::AbstractVector, b::AbstractVector)::Float64 where T<:Unsigned\n\nComputes the binary hamming distance for bit types and arrays of bit types\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.LcsDistance-Tuple{}","page":"Home","title":"SimilaritySearch.LcsDistance","text":"LcsDistance(a, b)\n\nInstantiates a GenericLevenshteinDistance object to perform LCS distance\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.LevenshteinDistance-Tuple{}","page":"Home","title":"SimilaritySearch.LevenshteinDistance","text":"LevenshteinDistance(a, b)\n\nInstantiates a GenericLevenshteinDistance object to perform traditional levenshtein distance\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.common_prefix-Tuple{Any,Any}","page":"Home","title":"SimilaritySearch.common_prefix","text":"common_prefix(a, b)\n\nComputes the length of the common prefix among two strings represented as arrays\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.compute_distances-Union{Tuple{T}, Tuple{PreMetric,AbstractArray{T,1},AbstractArray{Int64,1},T}} where T","page":"Home","title":"SimilaritySearch.compute_distances","text":"compute_distances(dist::PreMetric, db::AbstractVector{T}, refs::AbstractVector{Int}, q::T) where T\n\nComputes the distances of q to the set of references refs (each index point to an item in db) It returns an array of tuples (distance, refID)\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.compute_distances-Union{Tuple{T}, Tuple{PreMetric,AbstractArray{T,1},T}} where T","page":"Home","title":"SimilaritySearch.compute_distances","text":"compute_distances(dist::PreMetric, refs::AbstractVector{T}, q::T) where T\n\nComputes the distances of q to the set of references refs It returns an array of tuples (distance, refID)\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.covrad-Tuple{KnnResult}","page":"Home","title":"SimilaritySearch.covrad","text":"covrad(p::KnnResult)\n\nReturns the coverage radius of the result set; if length(p) < K then typemax(Float32) is returned\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.distant_tournament-Union{Tuple{T}, Tuple{PreMetric,Array{T,1},Integer}, Tuple{PreMetric,Array{T,1},Integer,Integer}} where T","page":"Home","title":"SimilaritySearch.distant_tournament","text":"distant_tournament(dist::PreMetric, db::Array{T,1}, numrefs::Integer, tournamentsize::Integer=3) where T\n\nCreates a pivot table where each pivot is relativaly distant each other based on a distant tournament\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.find_neighborhood-Tuple{FixedNeighborhood,SearchGraph,Any}","page":"Home","title":"SimilaritySearch.find_neighborhood","text":"find_neighborhood(algo::FixedNeighborhood, index::SearchGraph, item)\n\nFinds a list of neighbors using the FixedNeighborhood criterion of item in the index\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.find_neighborhood-Tuple{SearchGraph,Any}","page":"Home","title":"SimilaritySearch.find_neighborhood","text":"find_neighborhood(index::SearchGraph{T}, item)\n\nSearches for item neighborhood in the index, i.e., if item were in the index whose items should be its neighbors (intenal function)\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.fix_order!-Tuple{Any,Integer}","page":"Home","title":"SimilaritySearch.fix_order!","text":"fix_order!(res::KnnResult, n)\n\nFixes the sorted state of the array. It implements a kind of insertion sort It is efficient due to the expected distribution of the items being inserted (it is expected just a few elements smaller than the current ones)\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.hill_climbing-Tuple{IHCSearch,SearchGraph,Any,KnnResult,Integer}","page":"Home","title":"SimilaritySearch.hill_climbing","text":"hill_climbing(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult, nodeID::Integer)\n\nRuns a single hill climbing search process starting in vertex nodeID\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.k_near_and_far-Union{Tuple{T}, Tuple{PreMetric,KnnResult,KnnResult,T,Array{T,1},Integer}} where T","page":"Home","title":"SimilaritySearch.k_near_and_far","text":"k_near_and_far(dist::PreMetric, near::KnnResult, far::KnnResult, obj::T, refs::Vector{T}, k::Integer) where T\n\nSearches for k near and far objects in the set of references\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.maxlength-Tuple{KnnResult}","page":"Home","title":"SimilaritySearch.maxlength","text":"maxlength(res::KnnResult)\n\nThe maximum allowed cardinality (the k of knn)\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.opt_expand_neighborhood-Tuple{Any,IHCSearch,Integer,Integer,Integer}","page":"Home","title":"SimilaritySearch.opt_expand_neighborhood","text":"opt_expand_neighborhood(fun, ihc::IHCSearch, n::Integer, iter::Integer, probes::Integer)\n\nGenerates configurations of the IHCSearch that feed the optimize! function (internal function)\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.optimize!-Tuple{Performance,LocalSearchAlgorithm,SearchGraph}","page":"Home","title":"SimilaritySearch.optimize!","text":"function optimize!(search_algo::LocalSearchAlgorithm,\n                   index::SearchGraph{T},\n                   recall::Float64,\n                   perf::Performance;\n                   bsize::Int=4,\n                   tol::Float64=0.01,\n                   probes::Int=0) where T\n\nOptimizes a local search index for an specific algorithm to get the desired performance. Note that optimizing for low-recall will yield to faster searches; the train queries are specified as part of the perf struct.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.optimize!-Tuple{Performance,SearchGraph}","page":"Home","title":"SimilaritySearch.optimize!","text":"optimize!(perf::Performance, index::SearchGraph;\nrecall=0.9, ksearch=10, verbose=index.opts.verbose, tol::Float64=0.01, maxiters::Integer=3, probes::Integer=0) \noptimize!(perf, index.search_algo, index; recall=recall, tol=tol, maxiters=3, probes=probes)\n\nOptimizes the index for the specified kind of queries.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.pop!-Tuple{KnnResult}","page":"Home","title":"SimilaritySearch.pop!","text":"pop!(p)\n\nRemoves and returns the last item in the pool, it is an O(1) operation\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.push_neighborhood!-Tuple{SearchGraph,Any,Array{Int32,1}}","page":"Home","title":"SimilaritySearch.push_neighborhood!","text":"push_neighborhood!(index::SearchGraph, item, L::AbstractVector{Int32})\n\nInserts the object item into the index, i.e., creates an edge from items listed in L and the vertex created for ìtem` (internal function)\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.search","page":"Home","title":"SimilaritySearch.search","text":"search(searchctx::AbstractSearchContext, q, k::Integer=maxlength(searchctx.res))\nsearch(searchctx::AbstractSearchContext, q)\n\nThis is the most generic search function. It calls almost all implementations whenever an integer k is given.\n\n\n\n\n\n","category":"function"},{"location":"#SimilaritySearch.search-Tuple{BeamSearch,SearchGraph,Any,KnnResult}","page":"Home","title":"SimilaritySearch.search","text":"Tries to reach the set of nearest neighbors specified in res for q.\n\nbs: the parameters of BeamSearch\nindex: the local search index\nq: the query\nres: The result object, it stores the results and also specifies the kind of query\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.search-Tuple{ExhaustiveSearch,Any,KnnResult}","page":"Home","title":"SimilaritySearch.search","text":"search(seq::ExhaustiveSearch, q, res::KnnResult)\n\nSolves the query evaluating all items in the given query.\n\nBy default, it uses an internal result buffer; multithreading applications must duplicate specify another res object.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.search-Tuple{IHCSearch,SearchGraph,Any,KnnResult}","page":"Home","title":"SimilaritySearch.search","text":"search(isearch::IHCSearch, index::SearchGraph, q, res::KnnResult)\n\nPerforms an iterated hill climbing search for q.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.search-Tuple{PivotedSearch,Any,KnnResult}","page":"Home","title":"SimilaritySearch.search","text":"search(index::PivotedSeach, q, res::KnnResult)\n\nSolves a query with the pivot index.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.search-Tuple{SearchGraph,Any,KnnResult}","page":"Home","title":"SimilaritySearch.search","text":"search(index::SearchGraph, q, res::KnnResult)\n\nSolves the specified query res for the query object q.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.search-Union{Tuple{T}, Tuple{Kvp,T,KnnResult}} where T","page":"Home","title":"SimilaritySearch.search","text":"search(kvp::Kvp, q::T, res::KnnResult) where T\n\nSearches for q in the kvp index\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.select_sss-Union{Tuple{T}, Tuple{PreMetric,AbstractArray{T,1},Float64}} where T","page":"Home","title":"SimilaritySearch.select_sss","text":"select_sss(dist::PreMetric, db::AbstractVector{T}, alpha::Float64; shuf::Bool=true) where T\n\nselectsss selects the necessary pivots to fulfill the SSS criterion using :param:alpha. If :param:shuffledb is true then the database is shuffled before the selection process; in any case, the estimation of the maximum distance introduces indeterminism, however it could be too small. If you need better a better random selection set :param:shuf as true\n\nIt returns a set of pivots as a list of integers pointing to elements in :param:db\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.select_tournament-Union{Tuple{T}, Tuple{PreMetric,AbstractArray{T,1},Int64,Int64}} where T","page":"Home","title":"SimilaritySearch.select_tournament","text":"select_tournament(dist::PreMetric, db::AbstractVector{T}, numrefs::Int, tournamentsize::Int) where T\n\nselect_tournament selects numrefs references from \u001bdb using a tournament criterion; each individual is selected among\u001btournamentsize\u001b individuals.\n\nIt returns a set of pivots as a list of integers pointing to elements in \u001b\u001b\u001b\u001b\u001b\u001b\u001b\u001b\u001b\u001b\u001bdb\u001b\u001b\u001b\u001b\u001b\u001b\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.sss-Union{Tuple{T}, Tuple{PreMetric,Array{T,1}}} where T","page":"Home","title":"SimilaritySearch.sss","text":"sss(dist::PreMetric, db::Array{T,1}; alpha::Real=0.35, shuf=false) where T\n\nCreates an SSS pivot table.\n\n\n\n\n\n","category":"method"},{"location":"#SimilaritySearch.union_intersection-Union{Tuple{T}, Tuple{T,T}} where T<:(AbstractArray{T,1} where T)","page":"Home","title":"SimilaritySearch.union_intersection","text":"union_intersection(a::T, b::T)\n\nComputes both the size of the unions an the size the intersections of a and b; specified as ordered sequences.\n\n\n\n\n\n","category":"method"}]
}