# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch
export Kvp, k_near_and_far, fit, search, push!

struct Kvp{DataType<:AbstractVector, DistanceType<:PreMetric} <: AbstractSearchContext
    dist::DistanceType
    db::DataType
    refs::DataType
    sparsetable::Vector{Vector{Pair{Int32,Float32}}}
    ksparse::Int
    res::KnnResult
end

Kvp(dist::PreMetric, db, refs, sparsetable, ksparse::Integer; ksearch::Integer=10) = 
    Kvp(dist, db, refs, sparsetable, ksparse, KnnResult(ksearch))

Base.copy(kvp::Kvp;
        dist::PreMetric=kvp.dist,
        db::AbstractVector=kvp.db,
        refs::AbstractVector=kvp.refs,
        sparsetable::AbstractVector=kvp.sparsetable,
        ksparse::Integer=kvp.ksparse,
        res::KnnResult=KnnResult(maxlength(kvp.res))
    ) =
    Kvp(dist, db, refs, sparsetable, ksparse, res)

Base.string(p::Kvp) = "{Kvp: dist=$(p.dist), n=$(length(p.db)), refs=$(length(p.refs)), ksparse=$(p.ksparse)}"


"""
    k_near_and_far(dist::PreMetric, near::KnnResult, far::KnnResult, obj::T, refs::Vector{T}, k::Integer) where T

Searches for `k` near and far objects in the set of references
"""
function k_near_and_far(dist::PreMetric, near::KnnResult, far::KnnResult, obj::T, refs::Vector{T}, k::Integer) where T
    empty!(near, k)
    empty!(far, k)

    for refID in eachindex(refs)
        d = evaluate(dist, obj, refs[refID])
        push!(near, refID, d)
        push!(far, refID, -d)
    end

    row = [near[i] for i in eachindex(near)]
    for i in eachindex(near)
        row[i] = near[i]
    end
    
    for i in length(far):-1:1
        p = far[i]
        push!(row, i + k => p.id => -p.dist)
    end

    row
end

"""
    Kvp(dist::PreMetric, db, refs, sparsetable, ksparse::Integer; ksearch::Integer=10)
    Kvp(dist::PreMetric, db::AbstractVector, refs::AbstractVector, ksparse::Integer)
    Kvp(dist::PreMetric,
        db::AbstractVector;
        numpivots::Integer=ceil(Int, sqrt(length(db))),
        ksparse::Integer=ceil(Int, log2(length(db)))
        )
Creates a K vantage points index: a sparse pivot table storing only `ksparse` near and far pivots using references as pivots.

"""

function Kvp(dist::PreMetric, db::AbstractVector, refs::AbstractVector, ksparse::Integer; ksearch::Integer=10)
    @info "Kvp, refs=$(typeof(db)), k=$(ksparse), numrefs=$(length(refs)), dist=$(dist)"
    near = KnnResult(ksparse)
    far = KnnResult(ksparse)

    sparsetable = [k_near_and_far(dist, near, far, db[i], refs, ksparse) for i in 1:length(db)]
    Kvp(dist, db, refs, sparsetable, ksparse; ksearch=ksearch)
end

function Kvp(dist::PreMetric,
        db::AbstractVector;
        numpivots::Integer=ceil(Int, sqrt(length(db))),
        ksparse::Integer=ceil(Int, log2(length(db)))
        )
    L = unique(rand(1:length(db), numpivots))
    Kvp(dist, db, db[L], ksparse)
end

"""
    search(kvp::Kvp, q::T, res::KnnResult) where T

Searches for `q` in the `kvp` index
"""
function search(kvp::Kvp, q::T, res::KnnResult) where T
    # d::Float64 = 0.0
    qI = [evaluate(kvp.dist, q, piv) for piv in kvp.refs]

    for i in eachindex(kvp.db)
        obj::T = kvp.db[i]
        objSparseRow = kvp.sparsetable[i]

        discarded = false
        @inbounds for item in objSparseRow
            pivID = item.id
            dop = item.dist
            if abs(dop - qI[pivID]) > covrad(res)
                discarded = true
                break
            end
        end

        discarded && continue
        push!(res, i, evaluate(kvp.dist, q, obj))
    end

    res
end

## 
## function push!(index::Kvp{T}, dist, obj::T) where T
##     push!(index.db, obj)
##     row = k_near_and_far(dist, obj, index.refs, index.k)
##     push!(index.sparsetable, row)
##     length(index.db)
## end
