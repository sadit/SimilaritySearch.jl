#  Copyright 2016-2019 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

module Graph

using ..SimilaritySearch
using JSON
import ..SimilaritySearch:
    push!, search, fit, KnnResult, optimize!

export LocalSearchAlgorithm, NeighborhoodAlgorithm, SearchGraph,
    compute_aknn, find_neighborhood, push_neighborhood!

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end

mutable struct SearchGraph{T} <: Index
    db::Vector{T}
    recall::Float64
    k::Int
    links::Vector{Vector{Int32}}
    search_algo::LocalSearchAlgorithm
    neighborhood_algo::NeighborhoodAlgorithm
end

@enum VertexSearchState begin
    UNKNOWN = 0
    VISITED = 1
    EXPLORED = 2
end


function fit(::Type{SearchGraph}, dist::Function, dataset::AbstractVector{T}; recall=0.9, k=10, search_algo=BeamSearch(), neighborhood_algo=LogSatNeighborhood(1.1)) where T
    links = Vector{Int32}[]
    index = SearchGraph(T[], recall, k, links, search_algo, neighborhood_algo)
    for item in dataset
        push!(index, dist, item)
    end

    index
end

include("opt.jl")

## neighborhoods
include("neighborhood/fixedneighborhood.jl")
include("neighborhood/logneighborhood.jl")
include("neighborhood/logsatneighborhood.jl")
include("neighborhood/gallopingneighborhood.jl")
include("neighborhood/essencialneighborhood.jl")
include("neighborhood/satneighborhood.jl")
include("neighborhood/galsatneighborhood.jl")
include("neighborhood/vorneighborhood.jl")

## search algorithms
include("ihc.jl")
include("tihc.jl")
include("beamsearch.jl")

### Basic operations on the index
const OPTIMIZE_LOGBASE = 10

function find_neighborhood(index::SearchGraph{T}, dist::Function, item::T) where T
    n = length(index.db)
    n == 0 && return (NnResult(), Int32[])
    k = ceil(Int, log(OPTIMIZE_LOGBASE, 1+n))
    k1::Int = ceil(Int, log(OPTIMIZE_LOGBASE, 2+n))

    if n > 1 && k != k1
        optimize!(index, dist, recall=index.recall)
    end

    return neighborhood(index.neighborhood_algo, index, dist, item)
end

function push_neighborhood!(index::SearchGraph{T}, item::T, L::AbstractVector{Int32}, n::Int) where T
    for objID in L
        push!(index.links[objID], 1+n)
    end

    push!(index.links, L)
    push!(index.db, item)
end

function push!(index::SearchGraph{T}, dist::Function, item::T) where T
    knn, neighbors = find_neighborhood(index, dist, item)
    push_neighborhood!(index, item, neighbors, length(index.db))
    length(index.db) % 5000 == 0 && @debug "added n=$(length(index.db)), neighborhood=$(length(neighbors)), $(now())"
    knn, neighbors
end

const EMPTY_INT_VECTOR = Int[]

function search(index::SearchGraph{T}, dist::Function, q::T, res::KnnResult; hints::Vector{Int}=EMPTY_INT_VECTOR) where T
    length(index.db) == 0 && return res
    navigation_state = Dict{Int,VertexSearchState}()
    sizehint!(navigation_state, maxlength(res))
    # navigation_state = zeros(UInt8, length(index.db) + 1)
    search(index.search_algo, index, dist, q, res, navigation_state, hints)
end

function optimize!(index::SearchGraph{T}, dist::Function; recall=0.9, k=10, num_queries=128, perf=nothing) where T
    if perf == nothing
        perf = Performance(index.db, dist; expected_k=k, num_queries=num_queries)
    end
    optimize!(index.search_algo, index, dist, recall, perf)
end


## function compute_aknn(index::SearchGraph{T}, dist::Function, k::Int) where T
##     n = length(index.db)
##     aknn = [KnnResult(k) for i=1:n]
##     tabu = falses(length(index.db))
## 
##     for i=1:n
##         if i > 1
##             fill!(tabu, false)
##         end
## 
##         j = 1
## 
##         q = index.db[i]
##         res = aknn[i]
##         for p in res
##             tabu[p.objID] = true
##         end
## 
##         function oracle(q::T)
##             # this can be a very costly operation, or can be a very fast estimator, please do some research about it!!
##             if length(res) > 0
##                 a = Iterators.flatten(index.links[p.objID] for p in res)
##                 return Iterators.flatten((a, index.links[i]))
##             else
##                 return index.links[i]
##             end
##         end
## 
##         beam_search(index.search_algo, index, dist, q, res, tabu, oracle)
##         for p in res
##             i < p.objID && push!(aknn[p.objID], i, p.dist)
##         end
## 
##         if (i % 10000) == 1
##             @debug "algorithm=$(index.search_algo), neighborhood_factor=$(index.neighborhood_algo), k=$(k); advance $i of n=$n"
##         end
##     end
## 
##     return aknn
## end

end