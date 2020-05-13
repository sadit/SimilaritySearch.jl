# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Laesa

mutable struct Laesa{T} <: Index
    db::Vector{T}
    pivots::Vector{T}
    table::Matrix{Float64} # rows: number of pivots; cols: number of objects 
end

"""
    fit(::Type{Laesa}, dist, db::AbstractVector{T}, pivots::Vector{T})
    fit(::Type{Laesa}, dist, db::AbstractVector{T}, numPivots::Integer)

Creates a `Laesa` index with the given pivots. If the number of pivots is specified,
then they will be randomly selected from the dataset.
"""
function fit(::Type{Laesa}, dist, db::AbstractVector{T}, pivots::AbstractVector{T})  where T
    @info "Creating a pivot table with $(length(pivots)) pivots and distance=$(dist)"
    table = Matrix{Float64}(undef, length(pivots), length(db))

    for j in 1:length(db)
        for i in 1:length(pivots)
            table[i, j] = dist(db[j], pivots[i])
        end
    end

    Laesa(db, pivots, table)
end

function fit(::Type{Laesa}, dist, db::AbstractVector{T}, numPivots::Integer) where T
    pivots = rand(db, numPivots)
    fit(Laesa, dist, db, pivots)
end

"""
    search(index::Laesa, dist, q, res::KnnResult)

Solves a query with the Laesa index.
"""
function search(index::Laesa, dist, q, res::KnnResult)
    dqp = [dist(q, piv) for piv in index.pivots]
    for i in eachindex(index.db)
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
            push!(res, Item(i, d))
        end
    end

    return res
end

"""
    push!(index::Laesa, dist, obj)

Inserts `obj` into the index
"""
function push!(index::Laesa, dist, obj)
    push!(index.db, obj)
    vec = Vector{Float64}(undef, length(index.pivots))
    for pivID in 1:length(vec)
        vec[pivID] = dist(index.pivots[pivID], obj)
    end

    index.table = hcat(index.table, vec)
    length(index.db)
end
