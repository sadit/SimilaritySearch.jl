#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http:#www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

export LocalSearchAlgorithm, NeighborhoodAlgorithm, LocalSearchIndex, fit!, push!, compute_aknn, find_neighborhood, push_neighborhood!

abstract type LocalSearchAlgorithm end
abstract type NeighborhoodAlgorithm end

mutable struct LocalSearchOptions
    verbose::Bool
    oracle::Nullable{Function}
end

LocalSearchOptions(verbose::Bool) = LocalSearchOptions(verbose, Nullable{Function}())
LocalSearchOptions() = LocalSearchOptions(true)

mutable struct LocalSearchIndex{T,D} <: Index
    search_algo::LocalSearchAlgorithm
    neighborhood_algo::NeighborhoodAlgorithm
    db::Vector{T}
    dist::D
    recall::Float64
    k::Int
    links::Vector{Vector{Int32}}
    options::LocalSearchOptions
end

function LocalSearchIndex{D}(dtype::Type, dist::D;
                             recall=0.9,
                             k=10,
                             search::Nullable{LocalSearchAlgorithm}=Nullable{LocalSearchAlgorithm}(),
                             neighborhood::Nullable{NeighborhoodAlgorithm}=Nullable{NeighborhoodAlgorithm}(),
                             options::Nullable{LocalSearchOptions}=Nullable{LocalSearchOptions}()
                             )
    search = isnull(search) ? BeamSearch() : get(search)
    neighborhood = isnull(neighborhood) ? LogSatNeighborhood(1.1) : get(neighborhood)
    options = isnull(options) ? LocalSearchOptions() : get(options)
    
    LocalSearchIndex(search, neighborhood, Vector{dtype}(), dist, recall, k, Vector{Vector{Int32}}(), options)
end

function LocalSearchIndex(recall=0.9, k=10)
    LocalSearchIndex(
                     BeamSearch(),
                     LogSatNeighborhood(1.1),
                     Vector{Vector{Float32}}(),
                     L2SquaredDistance(),
                     recall,
                     k,
                     Vector{Vector{Int32}}(),
                     LocalSearchOptions()
                     )
end

include("graph/utils.jl")
include("graph/opt.jl")
include("graph/neighborhood/fixedneighborhood.jl")
include("graph/neighborhood/logneighborhood.jl")
include("graph/neighborhood/logsatneighborhood.jl")
include("graph/neighborhood/gallopingneighborhood.jl")
include("graph/neighborhood/essencialneighborhood.jl")
include("graph/neighborhood/satneighborhood.jl")
include("graph/neighborhood/galsatneighborhood.jl")
include("graph/neighborhood/vorneighborhood.jl")
include("graph/ihc.jl")
# include("graph/is2014.jl")
include("graph/neighborhoodsearch.jl")
include("graph/shrinkingneighborhoodsearch.jl")
include("graph/beamsearch.jl")
include("graph/deltasearch.jl")


#### save index
function save{T,D}(ostream, index::LocalSearchIndex{T,D}; saveitems=true, savelinks=true)
    header = Dict(
                  "search_algo" => string(index.search_algo),
                  "neighborhood_algo" => string(index.neighborhood_algo),
                  "type" => string(typeof(index)),
                  "recall" => index.recall,
                  "k" => index.k,
                  )
    write(ostream, JSON.json(header), "\n")
    if saveitems && savelinks
        for i in 1:length(index.links)
            save(ostream, index.db[i])
            save(ostream, Int32[x for x in index.links[i] if x < i])
        end
    elseif saveitems
        for i in 1:length(index.db)
            save(ostream, index.db[i])
        end
    elseif savelinks
        for i in 1:length(index.links)
            save(ostream, Int32[x for x in index.links[i] if x < i])
        end
    end
end


#### load index
function _load_index{T, D}(istream, ::Type{T}, dist::D)
    h = JSON.parse(readline(istream))
    index = LocalSearchIndex(T, dist, recall=h["recall"], k=h["k"])
    index.search_algo = eval(parse(h["search_algo"]))
    index.neighborhood_algo = eval(parse(h["neighborhood_algo"]))
    index.dist = dist
    index
end

function load{T, D}(istream, ::Type{LocalSearchIndex}, db::Vector{T}, dist::D)
    index = _load_index(istream, T, dist)
    load(istream, index, db)
    return index
end

function load{T, D}(istream, ::Type{LocalSearchIndex}, ::Type{T}, dist::D; loaditems=true)
    index = _load_index(istream, T, dist)
    loaditems && load(istream, index)
    return index
end

function load{T, D}(istream, index::LocalSearchIndex{T, D}, db::Vector{T})
    for i in 1:length(db)
        assert(!eof(istream))
        item = db[i]
        links = load(istream, Vector{Int32})
        push_neighborhood!(index, item, links, i-1)
    end
end

function load{T, D}(istream, index::LocalSearchIndex{T, D})
    while !eof(istream)
        n = length(index.db)
        item = load(istream, T)
        links = load(istream, Vector{Int32})
        push_neighborhood!(index, item, links, n)
    end
end

### Basic operations on the index
const NNS_PUSH_LOGBASE = 2

function find_neighborhood{T}(index::LocalSearchIndex{T}, item::T)
    n::Int = length(index.db)
    n == 0 && return (NnResult(), Int32[])
    k::Int = ceil(Int, log(NNS_PUSH_LOGBASE, 1+n))
    k_1::Int = ceil(Int, log(NNS_PUSH_LOGBASE, 2+n))

    if n > 4 && k != k_1
        optimize!(index, index.recall)
    end

    return neighborhood(index.neighborhood_algo, index, item)
end

function push_neighborhood!{T}(index::LocalSearchIndex{T}, item::T, L::Vector{Int32}, n::Int)
    for objID in L
        push!(index.links[objID], 1+n)
    end

    push!(index.links, L)
    push!(index.db, item)
end

function push!{T}(index::LocalSearchIndex{T}, item::T)
    knn, neighbors = find_neighborhood(index, item)
    push_neighborhood!(index, item, neighbors, length(index.db))
    index.options.verbose && length(index.db) % 5000 == 0 && info("added n=$(n+1), neighborhood=$(length(neighbors)), $(now())")
    knn, neighbors
end

function fit!{T,D}(index::LocalSearchIndex{T,D}, dataset::Vector{T})
    for item in dataset
        push!(index, item)
    end
end

function search{T, R <: Result}(index::LocalSearchIndex{T}, q::T, res::R)
    search(index.search_algo, index, q, res)
    return res
end

function search{T}(index::LocalSearchIndex{T}, q::T)
    return search(index, q, NnResult())
end

function optimize!{T}(index::LocalSearchIndex{T}, recall::Float64; perf::Nullable{Performance}=Nullable{Performance}())
    perf = isnull(perf) ? Performance(index.db, index.dist) : get(perf)
    optimize_algo!(index.search_algo, index, recall, perf)
end

function search_at(index::LocalSearchIndex{T}, q::T, start::I, res::R, tabu) where {T, I<:Integer, R<:Result}
    length(index.db) == 0 && return res

    function oracle(_q)
        index.links[start]
    end

    beam_search(index.search_algo, index, q, res, tabu, Nullable{Function}(oracle))
    return res

    res
end

function search_at(index::LocalSearchIndex{T}, q::T, start::I, res::R) where {T, I<:Integer, R<:Result}
    tabu = falses(length(index.db))
    search_at(index, q, start, res, tabu)
end

function compute_aknn{T}(index::LocalSearchIndex{T}, k::Int) # k=index.k, recall=index.recall)
    # optimize!(index, recall, perf=Performance(index.db, index.dist, k=k))
    n = length(index.db)
    aknn = [KnnResult(k) for i=1:n]
    tabu = falses(length(index.db))

    for i=1:n
        if i > 1
            tabu[:] = false
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

        beam_search(index.search_algo, index, q, res, tabu, Nullable{Function}(oracle))
        for p in res
            i < p.objID && push!(aknn[p.objID], i, p.dist)
        end

        if index.options.verbose && (i % 10000) == 1
            info("algorithm=$(index.search_algo), neighborhood_factor=$(index.neighborhood_algo), k=$(k); advance $i of n=$n")
        end
    end

    return aknn
    # newindex = LocalSearchIndex(index.search_algo, index.neighborhood_algo, index.db, index.dist, index.recall, index.k, index.restarts, index.beam_size, index.montecarlo_size, index.candidate_size, aknn)
    # return newindex
end
