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


export Laesa

mutable struct Laesa{T} <: Index
    db::Vector{T}
    pivots::Vector{T}
    table::Matrix{Float64} # rows: number of pivots; cols: number of objects 
end

function fit(::Type{Laesa}, dist::Function, db::AbstractVector{T}, pivots::Vector{T})  where T
    @info "Creating a pivot table with $(length(pivots)) pivots and distance=$(dist)"
    table = Matrix{Float64}(undef, length(pivots), length(db))

    for j in 1:length(db)
        for i in 1:length(pivots)
            table[i, j] = dist(db[j], pivots[i])
        end
    end

    Laesa(db, pivots, table)
end

function fit(::Type{Laesa}, dist::Function, db::AbstractVector{T}, numPivots::Integer) where T
    pivots = rand(db, numPivots)
    fit(Laesa, dist, db, pivots)
end

function search(index::Laesa{T}, dist::Function, q::T, res::KnnResult) where T
    dqp = [dist(q, piv) for piv in index.pivots]
    for i in 1:length(index.db)
        dpu = @view index.table[:, i]
        
        evaluate = true
        for pivID in 1:length(index.pivots)
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

function push!(index::Laesa{T}, dist::Function, obj::T) where T
    push!(index.db, obj)
    vec = Vector{Float64}(undef, length(index.pivots))
    for pivID in 1:length(vec)
        vec[pivID] = dist(index.pivots[pivID], obj)
    end

    index.table = hcat(index.table, vec)
    length(index.db)
end
