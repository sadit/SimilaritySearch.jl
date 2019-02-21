#  Copyright 2016,2017 Eric S. Tellez <eric.tellez@infotec.mx>
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
using Dates

import SimilaritySearch:
    search

import Base:
    push!

export Knr, optimize!

mutable struct Knr{T, D} <: Index
    db::Vector{T}
    dist::D
    refs::Vector{T}
    k::Int
    ksearch::Int
    minmatches::Int
    invindex::Vector{Vector{Int32}}
end

function Knr(db::Vector{T}, dist::D, refs::Vector{T}, k::Int, minmatches::Int=1) where {T,D}
    @info "Knr> refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)"
    invindex = [Vector{Int32}(undef, 0) for i in 1:length(refs)]
    seqindex = Sequential(refs, dist)

    pc = round(Int, length(db) / 20)
    for i=1:length(db)
        if i % pc == 0
            @info "Knr> advance $(round(i/length(db), digits=4)), now: $(now())"
        end

        res = search(seqindex, db[i], KnnResult(k))
        for p in res
            push!(invindex[p.objID], i)
        end
    end

    Knr(db, dist, refs, k, k, minmatches, invindex)
end

function Knr(db::Array{T,1}, dist::D; numrefs::Int=1024, k::Int=7, minmatches::Int=1, tournamentsize::Int=3) where {T,D}
    # refs = rand(db, numrefs)
    refs = [db[x] for x in select_tournament(db, dist, numrefs, tournamentsize)]
    Knr(db, dist, refs, k, minmatches)
end

"""
    search(index::Knr{T, D}, q::T) where {T,D}

Solves the search specified by `q`and `res` using `index`
"""
function search(index::Knr{T,D}, q::T, res::Result) where {T,D}
    dz = zeros(Int16, length(index.db))
    # M = BitArray(length(index.db))
    seqindex = Sequential(index.refs, index.dist)
    kres = search(seqindex, q, KnnResult(index.ksearch))

    for p in kres
        @inbounds for objID in index.invindex[p.objID]
            c = dz[objID] + 1
            dz[objID] = c

            if c == index.minmatches
                d = index.dist(q, index.db[objID])
                push!(res, objID, d)
            end
        end
    end

    return res
end

function push!(index::Knr{T, D}, obj::T) where {T,D}
    push!(index.db, obj)
    seqindex = Sequential(index.refs, index.dist)
    res = search(seqindex, obj, KnnResult(index.k))
    for p in res
        push!(index.invindex[p.objID], length(index.db))
    end
    return length(index.db)
end

function optimize!(index::Knr{T, D}; recall::Float64=0.9, k::Int=1, numqueries::Int=128, use_distances::Bool=false) where {T,D}
    @info "Knr> optimizing index for recall=$(recall)"
    perf = Performance(index.db, index.dist; numqueries=numqueries, expected_k=k)
    index.minmatches = 1
    index.ksearch = 1
    p = probe(perf, index, use_distances=use_distances)

    while p.recall < recall && index.ksearch < length(index.refs)
        index.ksearch += 1
        @info "Knr> opt step ksearch=$(index.ksearch), performance $(p)"
        p = probe(perf, index, use_distances=use_distances)

    end
    @info "Knr> reached performance $(p)"
    return index
end

