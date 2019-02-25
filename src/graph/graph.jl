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

using SimilaritySearch
using JSON
import SimilaritySearch: push!, search

export LocalSearchAlgorithm, NeighborhoodAlgorithm, LocalSearchIndex,
    optimize!, fit!, push!, compute_aknn, find_neighborhood, push_neighborhood!, search_at

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end


mutable struct LocalSearchIndex{T,D} <: Index
    search_algo::LocalSearchAlgorithm
    neighborhood_algo::NeighborhoodAlgorithm
    db::AbstractVector{T}
    dist::D
    recall::Float64
    k::Int
    links::Vector{Vector{Int32}}
end

function LocalSearchIndex(
    dtype::Type, dist::D;
    recall=0.9,
    k=10,
    search::LocalSearchAlgorithm=BeamSearch(),
    neighborhood::NeighborhoodAlgorithm=LogSatNeighborhood(1.1)
) where {D}
    LocalSearchIndex(search, neighborhood, Vector{dtype}(), dist, recall, k, Vector{Vector{Int32}}())
end


include("utils.jl")
include("opt.jl")
include("neighborhood/fixedneighborhood.jl")
include("neighborhood/logneighborhood.jl")
include("neighborhood/logsatneighborhood.jl")
include("neighborhood/gallopingneighborhood.jl")
include("neighborhood/essencialneighborhood.jl")
include("neighborhood/satneighborhood.jl")
include("neighborhood/galsatneighborhood.jl")
include("neighborhood/vorneighborhood.jl")
include("ihc.jl")
# include("is2014.jl")
include("neighborhoodsearch.jl")
include("shrinkingneighborhoodsearch.jl")
include("beamsearch.jl")
include("deltasearch.jl")


### Basic operations on the index
const OPTIMIZE_LOGBASE = 2

function find_neighborhood(index::LocalSearchIndex{T}, item::T) where {T}
    n::Int = length(index.db)
    n == 0 && return (NnResult(), Int32[])
    k::Int = ceil(Int, log(OPTIMIZE_LOGBASE, 1+n))
    k_1::Int = ceil(Int, log(OPTIMIZE_LOGBASE, 2+n))

    if n > 4 && k != k_1
        optimize!(index, index.recall)
    end

    return neighborhood(index.neighborhood_algo, index, item)
end

function push_neighborhood!(index::LocalSearchIndex{T}, item::T, L::AbstractVector{Int32}, n::Int) where {T}
    for objID in L
        push!(index.links[objID], 1+n)
    end

    push!(index.links, L)
    push!(index.db, item)
end

function push!(index::LocalSearchIndex{T}, item::T) where {T}
    knn, neighbors = find_neighborhood(index, item)
    push_neighborhood!(index, item, neighbors, length(index.db))
    length(index.db) % 5000 == 0 && @info "added n=$(length(index.db)), neighborhood=$(length(neighbors)), $(now())"
    knn, neighbors
end

function fit!(index::LocalSearchIndex{T,D}, dataset::AbstractVector{T}) where {T,D}
    for item in dataset
        push!(index, item)
    end
end

function search(index::LocalSearchIndex{T,D}, q::T, res::Result; oracle=nothing) where {T,D}
    search(index.search_algo, index, q, res, oracle=oracle)
    return res
end

function optimize!(index::LocalSearchIndex{T,D}, recall::Float64; perf=nothing) where {T,D}
    if perf == nothing
        perf = Performance(index.db, index.dist)
    end

    optimize_algo!(index.search_algo, index, recall, perf)
end

function search_at(index::LocalSearchIndex{T,D}, q::T, start::Integer, res::Result, tabu) where {T,D}
    length(index.db) == 0 && return res

    function oracle(_q)
        index.links[start]
    end

    beam_search(index.search_algo, index, q, res, tabu, oracle)
    return res

    res
end

function search_at(index::LocalSearchIndex{T,D}, q::T, start::Integer, res::Result) where {T,D}
    tabu = falses(length(index.db))
    search_at(index, q, start, res, tabu)
end

function compute_aknn(index::LocalSearchIndex{T,D}, k::Int) where {T,D}
    n = length(index.db)
    aknn = [KnnResult(k) for i=1:n]
    tabu = falses(length(index.db))

    for i=1:n
        if i > 1
            fill!(tabu, false)
        end

        j = 1

        q = index.db[i]
        res = aknn[i]
        for p in res
            tabu[p.objID] = true
        end

        function oracle(q::T)
            # this can be a very costly operation, or can be a very fast estimator, please do some research about it!!
            if length(res) > 0
                a = Iterators.flatten(index.links[p.objID] for p in res)
                return Iterators.flatten((a, index.links[i]))
            else
                return index.links[i]
            end
        end

        beam_search(index.search_algo, index, q, res, tabu, oracle)
        for p in res
            i < p.objID && push!(aknn[p.objID], i, p.dist)
        end

        if (i % 10000) == 1
            @info "algorithm=$(index.search_algo), neighborhood_factor=$(index.neighborhood_algo), k=$(k); advance $i of n=$n"
        end
    end

    return aknn
end
