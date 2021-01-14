# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using SimilaritySearch

import SimilaritySearch:
    search, fit, push!

export Kvp, k_near_and_far, fit, search, push!

mutable struct Kvp{T} <: Index
    db::Vector{T}
    refs::Vector{T}
    sparsetable::Vector{Vector{Item}}
    k::Int
end

function k_near_and_far(dist::PreMetric, obj::T, refs::Vector{T}, k::Int) where T
    near = KnnResult(k)
    far = KnnResult(k)

    for refID in eachindex(refs)
        d = evaluate(dist, obj, refs[refID])
        push!(near, refID, d)
        push!(far, refID, -d)
    end

    row = Item[]
    sizehint!(row, 2*k)
    for p in near
        push!(row, p)
    end
    
    for p in far
        push!(row, Item(p.id, -p.dist))
    end

    row
end

function fit(::Type{Kvp}, dist::PreMetric, db::Vector{T}, k::Int, refs::Vector{T}) where T
    @info "Kvp, refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)"
    sparsetable = Vector{Item}[]

    for i in 1:length(db)
        if (i % 10000) == 0
            println(stderr, "advance $(i)/$(length(db))")
        end

        row = k_near_and_far(dist, db[i], refs, k)
        push!(sparsetable, row)
    end

    Kvp(db, refs, sparsetable, k)
end

function fit(::Type{Kvp}, dist::PreMetric, db::Vector{T}, k::Int, numrefs::Int) where T
    refList = rand(1:length(db), numrefs)
    refs = [db[x] for x in refList]
    fit(Kvp, dist, db, k, refs)
end

function search(index::Kvp{T}, dist::PreMetric, q::T, res::KnnResult) where T
    d::Float64 = 0.0
    qI = [evaluate(dist, q, piv) for piv in index.refs]

    for i in eachindex(index.db)
        obj::T = index.db[i]
        objSparseRow = index.sparsetable[i]

        discarded::Bool = false
        @inbounds for item in objSparseRow
            pivID = item.id
            dop = item.dist
            if abs(dop - qI[pivID]) > covrad(res)
                discarded = true
                break
            end
        end

        if discarded
            continue
        end
        d = evaluate(dist, q, obj)
        push!(res, i, d)
    end

    res
end

function push!(index::Kvp{T}, dist, obj::T) where T
    push!(index.db, obj)
    row = k_near_and_far(dist, obj, index.refs, index.k)
    push!(index.sparsetable, row)
    length(index.db)
end
