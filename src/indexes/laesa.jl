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

# abstract Sequential

export Laesa
# abstract PivotRowType <: Array{Float64, 1}

struct Laesa{T,D} <: Index
    db::Vector{T}
    dist::D
    pivots::Vector{T}
    table::Vector{Vector{Float64}}
end

function Laesa(db::Vector{T}, dist::D, pivots::Vector{T}) where {T,D}
    info("Creating a pivot table with $(length(pivots)) pivots and distance=$(dist)")
    table = Vector{Vector{Float64}}(length(db))
    for i in 1:length(db)
        table[i] = [dist(piv, db[i]) for piv in pivots]
    end

    Laesa(db, dist, pivots, table)
end

function Laesa(db::Vector{T}, dist::D, numPivots::Int) where {T,D}
    pivots = rand(db, numPivots)
    Laesa(db, dist, pivots)
end

function search(index::Laesa{T,D}, q::T, res::Result) where {T,D}
    dist = index.dist
    dqp = [dist(q, piv) for piv in index.pivots]
    for i = 1:length(index.db)
        dpu = index.table[i]

        evaluate = true
        @inbounds for pivID in 1:length(index.pivots)
            if abs(dqp[pivID] - dpu[pivID]) > covrad(res)
                evaluate = false
                break
            end
        end

        if evaluate
            d = dist(q, index.db[i])
            push!(res, i, d)
        end
    end

    return res
end

function push!(index::Laesa{T,D}, obj::T) where {T,D}
    dist = index.dist
    push!(index.db, obj)
    row = Array(Float64, length(index.pivots))
    for pivID in 1:length(index.pivots)
        row[pivID] = dist(index.pivots[pivID], obj)
    end
    push!(index.table, row)
    return length(index.db)
end
