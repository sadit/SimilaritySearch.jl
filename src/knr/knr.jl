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
    search, fit, push!, optimize!

export Knr

mutable struct Knr{T} <: Index
    db::Vector{T}
    refs::Vector{T}
    k::Int
    ksearch::Int
    minmatches::Int
    invindex::Vector{Vector{Int32}}
end

function fit(::Type{Knr}, dist::Function, db::AbstractVector{T}, refs::AbstractVector{T}, k::Int, minmatches::Int=1) where T
    @info "Knr> refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)"
    invindex = [Vector{Int32}(undef, 0) for i in 1:length(refs)]
    seqindex = fit(Sequential, refs)

    pc = round(Int, length(db) / 20)
    for i=1:length(db)
        if i % pc == 0
            @info "Knr> advance $(round(i/length(db), digits=4)), now: $(now())"
        end

        res = search(seqindex, dist, db[i], KnnResult(k))
        for p in res
            push!(invindex[p.objID], i)
        end
    end

    Knr(db, refs, k, k, minmatches, invindex)
end

function fit(::Type{Knr}, dist::Function, db::AbstractVector{T}; numrefs::Int=1024, k::Int=3, minmatches::Int=1, tournamentsize::Int=3) where T
    # refs = rand(db, numrefs)
    refs = [db[x] for x in select_tournament(dist, db, numrefs, tournamentsize)]
    fit(Knr, dist, db, refs, k, minmatches)
end

"""
search

Solves the search specified by `q`and `res` using `index`
"""
function search(index::Knr{T}, dist::Function, q::T, res::KnnResult) where {T}
    dz = zeros(Int16, length(index.db))
    # M = BitArray(length(index.db))
    seqindex = fit(Sequential, index.refs)
    kres = search(seqindex, dist, q, KnnResult(index.ksearch))

    for p in kres
        @inbounds for objID in index.invindex[p.objID]
            c = dz[objID] + 1
            dz[objID] = c

            if c == index.minmatches
                d = dist(q, index.db[objID])
                push!(res, objID, d)
            end
        end
    end

    return res
end

function push!(index::Knr{T}, dist::Function, obj::T) where T
    push!(index.db, obj)
    seqindex = fit(Sequential, index.refs)
    res = search(seqindex, dist, obj, KnnResult(index.k))
    for p in res
        push!(index.invindex[p.objID], length(index.db))
    end
    return length(index.db)
end

function optimize!(index::Knr{T}, dist::Function; recall=0.9, k=10, num_queries=128, perf=nothing) where T
    @info "Knr> optimizing index for recall=$(recall)"
    if perf == nothing
        perf = Performance(index.db, dist; expected_k=k, num_queries=num_queries)
    end
    index.minmatches = 1
    index.ksearch = 1
    p = probe(perf, index, dist)

    while p.recall < recall && index.ksearch < length(index.refs)
        index.ksearch += 1
        @info "Knr> opt step ksearch=$(index.ksearch), performance $(p)"
        p = probe(perf, index, dist)

    end
    @info "Knr> reached performance $(p)"
    return index
end