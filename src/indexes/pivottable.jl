# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export PivotedSearch

"""
    PivotedSeach(index::PivotTable, db::AbstractVector, dist::PreMetric, knn::KnnResult)
    PivotedSeach(index::PivotTable, db::AbstractVector, dist::PreMetric, k::Integer=10)

Defines a search context for PivotTables
"""
struct PivotedSearch{DataType<:AbstractVector, DistanceType<:PreMetric} <: AbstractSearchContext
    dist::DistanceType
    db::DataType
    pivots::DataType
    table::Vector{Vector{Float32}} # pivot table
    dqp::Vector{Float32} # query mapped to the pivot space
    res::KnnResult
end

Base.copy(index::PivotedSearch; dist=index.dist, db=index.db, pivots=index.pivots, table=index.table, dqp=index.dqp, res=index.res) =
    PivotedSearch(dist, db, pivots, table, dqp, res)
Base.string(p::PivotedSearch) = "{PivotedSearch: dist=$(p.dist), n=$(length(p.db)), pivs=$(length(p.pivots)), knn=$(p.res)}"

PivotedSearch(dist::PreMetric, db, pivots, table, k::Integer=10) =
    PivotedSearch(dist, db, pivots, table, zeros(Float32, length(pivots)), KnnResult(k))

"""
    PivotedSearch(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, pivots::Vector{T})
    PivotedSearch(::Type{PivotTable}, dist::PreMetric, db::AbstractVector{T}, numpivots::Integer)

Creates a `PivotTable` index with the given pivots. If the number of pivots is specified,
then they will be randomly selected from the dataset.
"""
function PivotedSearch(dist::PreMetric, db::AbstractVector{T}, pivots::AbstractVector{T}) where T
    @info "Creating a pivot table with $(length(pivots)) pivots and distance=$(dist)"
    table = Vector{Vector{Float32}}(undef, length(db))
    for i in 1:length(db)
        u = db[i]
        table[i] = [evaluate(dist, u, pivots[j]) for j in 1:length(pivots)]
    end

    PivotedSearch(dist, db, pivots, table, 1)
end

function PivotedSearch(dist::PreMetric, db::AbstractVector{T}, numpivots::Integer) where T
    pivots = rand(db, numpivots)
    PivotedSearch(dist, db, pivots)
end

"""
    search(index::PivotedSeach, q, res::KnnResult)

Solves a query with the pivot index.
"""
function search(index::PivotedSearch, q, res::KnnResult)
    @inbounds for i in eachindex(index.pivots)
        index.dqp[i] = evaluate(index.dist, q, index.pivots[i])
    end
    
    for i in eachindex(index.db)
        dpu = index.table[i]
        need_eval = true

        for pivID in 1:length(index.pivots)
            if abs(index.dqp[pivID] - dpu[pivID]) > covrad(res)
                need_eval = false
                break
            end
        end

        if need_eval
            d = evaluate(index.dist, q, index.db[i])
            push!(res, i, d)
        end
    end

    res
end
