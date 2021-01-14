# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export PivotTable

mutable struct PivotTable{T} <: Index
    db::Vector{T}
    pivots::Vector{T}
    table::Matrix{Float64} # rows: number of pivots; cols: number of objects 
end

"""
    fit(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, pivots::Vector{T})
    fit(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, numPivots::Integer)

Creates a `PivotTable` index with the given pivots. If the number of pivots is specified,
then they will be randomly selected from the dataset.
"""
function fit(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, pivots::AbstractVector{T})  where T
    @info "Creating a pivot table with $(length(pivots)) pivots and distance=$(dist)"
    table = Matrix{Float64}(undef, length(pivots), length(db))

    for j in 1:length(db)
        for i in 1:length(pivots)
            table[i, j] = evaluate(dist, db[j], pivots[i])
        end
    end

    PivotTable(db, pivots, table)
end

function fit(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, numPivots::Integer) where T
    pivots = rand(db, numPivots)
    fit(PivotTable, dist, db, pivots)
end

"""
    search(index::PivotTable, dist, q, res::KnnResult)

Solves a query with the PivotTable index.
"""
function search(index::PivotTable, dist::PreMetric, q, res::KnnResult)
    dqp = [evaluate(dist, q, piv) for piv in index.pivots]
    for i in eachindex(index.db)
        dpu = @view index.table[:, i]
        
        need_eval = true
        for pivID in 1:length(index.pivots)
            if abs(dqp[pivID] - dpu[pivID]) > covrad(res)
                need_eval = false
                break
            end
        end

        if need_eval
            d = evaluate(dist, q, index.db[i])
            push!(res, i, d)
        end
    end

    return res
end

"""
    push!(index::PivotTable, dist, obj)

Inserts `obj` into the index
"""
function push!(index::PivotTable, dist, obj)
    push!(index.db, obj)
    vec = Vector{Float64}(undef, length(index.pivots))
    for pivID in 1:length(vec)
        vec[pivID] = evaluate(dist, index.pivots[pivID], obj)
    end

    index.table = hcat(index.table, vec)
    length(index.db)
end
