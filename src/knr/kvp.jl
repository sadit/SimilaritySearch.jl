#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
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

import SimilaritySearch:
    search

import Base:
    push!

export Kvp, near_and_far

mutable struct Kvp{T, D} <: Index
    db::Vector{T}
    dist::D
    refs::Vector{T}
    sparsetable::Vector{Vector{Item}}
    k::Int
end

function Kvp(db::Vector{T}, dist::D, k::Int, refList::Vector{Int}) where {T,D}
    @info "Kvp, refs=$(typeof(db)), k=$(k), numrefs=$(length(refList)), dist=$(dist)"
    sparsetable = Vector{Item}[]
    refs = [db[x] for x in refList]
    for i=1:length(db)
        if (i % 10000) == 0
            @info "advance $(i)/$(length(db))"
        end
        row = near_and_far(db[i], refs, k, dist)
        # println(row)
        push!(sparsetable, row)
    end

    return Kvp(db, dist, refs, sparsetable, k)
end

function Kvp(db::Vector{T}, dist::D, k::Int, numrefs::Int) where {T,D}
    refs = rand(1:length(db), numrefs)
    Kvp(db, dist, k, refs)
end

function near_and_far(obj::T, refs::Vector{T}, k::Int, dist::D) where {T,D}
    near = KnnResult(k)
    far = KnnResult(k)
    for (refID, ref) in enumerate(refs)
        d = dist(obj, ref)
        push!(near, refID, d)
        push!(far, refID, -d)
    end

    row = Vector{Item}(undef, k+k)
    for (j, item) in enumerate(near)
        row[j] = item
    end

    for (j, item) in enumerate(far)
        row[k+j] = Item(item.objID, -item.dist)
    end

    return row
end

function search(index::Kvp{T,D}, q::T, res::Result) where {T,D}
    # for i in range(1, length(index.db))
    d::Float64 = 0.0
    dist = index.dist
    qI = [dist(q, piv) for piv in index.refs]

    for i = 1:length(index.db)
        obj::T = index.db[i]
        objSparseRow = index.sparsetable[i]

        discarded::Bool = false
        @inbounds for item in objSparseRow
            pivID = item.objID
            dop = item.dist
            if abs(dop - qI[pivID]) > covrad(res)
                discarded = true
                break
            end
        end
        if discarded
            continue
        end
        d =dist(q, obj)
        push!(res, i, d)
    end

    return res
end

function push!(index::Kvp{T}, obj::T) where {T}
    push!(index.db, obj)
    row = near_and_far(obj, index.refs, index.k, index.dist)
    push!(index.sparsetable, row)
    return length(index.db)
end
