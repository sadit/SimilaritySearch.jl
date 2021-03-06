# This file is a part of SimilaritySearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export compute_distances, select_sss, select_tournament
using Random
"""
    compute_distances(dist::PreMetric, db::AbstractVector{T}, refs::AbstractVector{Int}, q::T) where T

Computes the distances of `q` to the set of references `refs` (each index point to an item in `db`)
It returns an array of tuples `(distance, refID)`
"""
function compute_distances(dist::PreMetric, db::AbstractVector{T}, refs::AbstractVector{Int}, q::T) where T
    [(evaluate(dist, q, db[refs[refID]]), refID) for refID in 1:length(refs)]
end

"""
    compute_distances(dist::PreMetric, refs::AbstractVector{T}, q::T) where T

Computes the distances of `q` to the set of references `refs`
It returns an array of tuples `(distance, refID)`
"""
function compute_distances(dist::PreMetric, refs::AbstractVector{T}, q::T) where T
    [(evaluate(dist, q, ref), refID) for (refID, ref) in enumerate(refs)]
end

"""
    select_tournament(dist::PreMetric, db::AbstractVector{T}, numrefs::Int, tournamentsize::Int) where T

select_tournament selects `numrefs` references from `db` using a tournament criterion; each
individual is selected among`tournamentsize` individuals.

It returns a set of pivots as a list of integers pointing to elements in `db`
"""
function select_tournament(dist::PreMetric, db::AbstractVector{T}, numrefs::Int, tournamentsize::Int) where T
    @info "select_tournament: db=$(typeof(db)), numrefs=$(numrefs), distance=$(dist), tournamentsize=$(tournamentsize)"

    refs = Vector{Int}()
    perm = 1:length(db) |> collect
    shuffle!(perm)
    push!(refs, perm[end])

    for i=1:(numrefs-1)
        sample = perm[(i-1)*tournamentsize+1:i*tournamentsize]
        distant = 0
        distantRef = 0
        for x in sample
            M = compute_distances(dist, db, refs, db[x])
            d, refID = minimum(M)
            if d > distant
                distant = d
                distantRef = x
            end
        end
        # println(sample, distant, distantRef)
        if distantRef > 0
            push!(refs, distantRef)
        end
    end

    return refs
end

"""
    select_sss(dist::PreMetric, db::AbstractVector{T}, alpha::Float64; shuf::Bool=true) where T

select_sss selects the necessary pivots to fulfill the SSS criterion using :param:alpha.
If :param:shuffle_db is true then the database is shuffled before the selection process; in any case,
the estimation of the maximum distance introduces indeterminism, however it could be too small.
If you need better a better random selection set :param:shuf as true

It returns a set of pivots as a list of integers pointing to elements in :param:db
"""
function select_sss(dist::PreMetric, db::AbstractVector{T}, alpha::Float64; shuf::Bool=true) where T
    @info "select_sss: db=$(typeof(db)), alpha=$(alpha), distance=$(dist), shuf=$(shuf)"
    dmax::Float64 = 0.0
    s = Int(round(sqrt(length(db))/2))
    sample1, sample2 = rand(db, s), rand(db, s)
    dmax = maximum([evaluate(dist, sample1[i], sample2[i]) for i=1:s])
    # plot(x=X, Geom.histogram(bincount=100))

    @info "the maximum distance estimated as $(dmax), now selecting pivots"
    xdb = Array(1:length(db))
    if shuf
        shuffle!(xdb)
    end
    #pivots::Array{T,1} = Array(T, 1)
    #pivots[1] = xdb[1]
    pivots = Int[xdb[1]]

    for i=2:length(xdb)
        obj = db[xdb[i]]
        if minimum([evaluate(dist, db[pivID], obj) for pivID in pivots]) / dmax < alpha
            continue
        end

        push!(pivots, i)
    end

    pivots
end
