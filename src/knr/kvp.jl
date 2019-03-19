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

import SimilaritySearch:
    search, fit, push!

export Kvp, near_and_far, fit, search, push!

mutable struct Kvp{T} <: Index
    db::Vector{T}
    refs::Vector{T}
    sparsetable::Vector{Vector{Item}}
    k::Int
end

function fit(::Type{Kvp}, dist::Function, db::Vector{T}, k::Int, refs::Vector{T}) where T
    @info "Kvp, refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)"
    sparsetable = Vector{Item}[]

    for i in 1:length(db)
        if (i % 10000) == 0
            @info "advance $(i)/$(length(db))"
        end

        row = near_and_far(dist, db[i], refs, k)
        push!(sparsetable, row)
    end

    Kvp(db, refs, sparsetable, k)
end

function fit(::Type{Kvp}, dist::Function, db::Vector{T}, k::Int, numrefs::Int) where T
    refList = rand(1:length(db), numrefs)
    refs = [db[x] for x in refList]
    fit(Kvp, dist, db, k, refs)
end

function near_and_far(dist::Function, obj::T, refs::Vector{T}, k::Int) where T
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

    row
end

function search(index::Kvp{T}, dist::Function, q::T, res::KnnResult) where T
    d::Float64 = 0.0
    qI = [dist(q, piv) for piv in index.refs]

    for i in 1:length(index.db)
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
        d = dist(q, obj)
        push!(res, i, d)
    end

    return res
end

function push!(index::Kvp{T}, dist::Function, obj::T) where T
    push!(index.db, obj)
    row = near_and_far(dist, obj, index.refs, index.k)
    push!(index.sparsetable, row)
    return length(index.db)
end
