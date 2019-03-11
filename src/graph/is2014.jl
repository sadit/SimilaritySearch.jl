# #  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.


# export Is2014Search

# mutable struct Is2014Search <: LocalSearchAlgorithm
# end


# # From Malkov et al. 2014 Information System.
# # def KNNSearch(q, m, k) {
# #     TreeSet [object] tempRes, candidates, visitedSet, result
# #
# #     for (i = 0; i < m; i++) {
# #        put random entry point in candidates
# #        tempRes <-- null;
# #        repeat {
# #        	get element c closest from candidates to q;
# # 	        remove c from candidates;
# # 	        // check stop condition:
# # 	        if (c is further than k-th element from result) {
# # 	             break repeat;
# #     	    }
# # 	        // update list of candidates:
# # 	        for (every element e from friends of c) {
# # 	             if (e is not in visitedSet) {
# # 	                  add e to visitedSet, candidates, tempRes;
# # 		         }
# #           }
# # 	     }
# #    	 // aggregate the results:
# # 	     add objects from tempRes to result;
# #    }
# #    return best k elements from result;
# # }

# function greedy_search_is2014(index::SearchGraph{T}, q::T, res::R, tabu::BitVector, candidates::KnnResult) where {T,R <: Result}
#     nodeID::Int32 = rand(1:length(index.db))
    
#     # if !in(nodeID, tabu)
#     if !tabu[nodeID]
#         d::Float32 = convert(Float32, index.dist(index.db[nodeID], q))
#         # push!(tabu, nodeID)
#         tabu[nodeID] = true
#         push!(res, nodeID, d)
#         push!(candidates, nodeID, d)
#     end

#     while length(candidates) > 0
#         best = popfirst!(candidates)
#         # cov = last(res).dist
#         if best.dist > covrad(res)
#             break
#         end

#         @inbounds for childID in index.links[best.objID]
#             # if !in(childID, tabu)
#             if !tabu[childID]
#                 d = convert(Float32, index.dist(index.db[childID], q))
#                 # push!(tabu, childID)
#                 tabu[childID] = true
#                 push!(res, childID, d)
#                 push!(candidates, childID, d)
#             end
#         end
#     end
# end

# function search(algorithm::Is2014Search, index::SearchGraph{T}, q::T, res::R) where {T,R <: Result}
#     n = length(index.db)
#     tabu = falses(n)
#     restarts = min(abs(index.restarts), n)
#     candidates = KnnResult(128)

#     for i=1:restarts
#         greedy_search_is2014(index, q, res, tabu, candidates)
#     end

#     return res
# end
