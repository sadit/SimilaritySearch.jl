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

# abstract Sequential

export Laesa
# abstract PivotRowType <: Array{Float64, 1}

struct Laesa{T, D} <: Index
    db::Vector{T}
    dist::D
    pivots::Vector{T}
    table::Vector{Vector{Float64}}
end

function Laesa{T, D}(db::Vector{T}, dist::D, pivots::Vector{T})
    info("Creating a pivot table with $(length(pivots)) pivots and distance=$(dist)")
    table = Vector{Vector{Float64}}(length(db))
    for i=1:length(db)
        obj = db[i]
        row = table[i] = Vector{Float64}(length(pivots))
        for pivID in 1:length(pivots)
            row[pivID] = dist(pivots[pivID], obj)
        end
    end

    Laesa(db, dist, pivots, table)
end

function Laesa{T, D}(db::Vector{T}, dist::D, numPivots::Int)
    pivots = rand(db, numPivots)
    Laesa(db, dist, pivots)
end

function search{T, D, R <: Result}(index::Laesa{T,D}, q::T, res::R)
    # for i in range(1, length(index.db))
    d::Float64 = 0.0
    qD = [index.dist(q, piv) for piv in index.pivots]

    for i = 1:length(index.db)
        obj::T = index.db[i]
        objD::Array{Float64,1} = index.table[i]

        discarded::Bool = false
        @inbounds for pivID in 1:length(qD)
            if abs(objD[pivID] - qD[pivID]) > covrad(res)
                discarded = true
                break
            end
        end
        if discarded
            continue
        end
        d = index.dist(q, obj)
        push!(res, i, d)
    end

    return res
end

function search{T, D}(index::Laesa{T,D}, q::T)
    return search(index, q, NnResult())
end

function push!{T, D}(index::Laesa{T,D}, obj::T)
    push!(index.db, obj)
    row = Array(Float64, length(index.pivots))
    for pivID in 1:length(index.pivots)
        row[pivID] = index.dist(pivots[pivID], obj)
    end
    push!(index.table, row)
    return length(index.db)
end
