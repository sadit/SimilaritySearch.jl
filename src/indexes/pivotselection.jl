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

export compute_distances, select_sss, select_tournament

"""
Computes the distances of :param:q to the set of references :param:refs (each index point to an item in :param:db)
It returns an array of tuples (distance, refID)
"""
function compute_distances(db::Array{T,1}, dist::D, q::T, refs::Array{Int,1}) where {T,D}
    [(dist(q, db[refs[refID]]), refID) for refID in 1:length(refs)]
end

"""
Computes the distances of :param:q to the set of references :param:refs
It returns an array of tuples (distance, refID)
"""
function compute_distances(refs::Array{T,1}, dist::D, q::T) where {T,D}
    [(dist(q, ref), refID) for (refID, ref) in enumerate(refs)]
end

"""
select_tournament selects :param:numrefs references from :param:db using a tournament criterion; each
tournament uses :param:tournamentsize individuals. When :param:tournamentsize is zero it is set to a
pesimistic value based on numrefs

It returns a set of pivots as a list of integers pointing to elements in :param:db
"""
function select_tournament(db::Array{T,1}, dist::D, numrefs::Int, tournamentsize::Int=0) where {T,D}
    if tournamentsize == 0
        tournamentsize = min(div(length(db), numrefs) - 1, Int(ceil(log2(numrefs)^2)))
    end

    refs = Array(Int, 1)
    xdb = Array(1:length(db)); shuffle!(xdb)
    refs[1] = xdb[end]

    for i=1:(numrefs-1)
        sample = xdb[(i-1)*tournamentsize+1:i*tournamentsize]
        distant = 0
        distantRef = 0
        for x in sample
            M = compute_distances(db, dist, db[x], refs)
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
select_sss selects the necessary pivots to fullfill the SSS criterion using :param:alpha.
If :param:shuffle_db is true then the database is shuffled before the selection process; in any case,
the estimation of the maximum distance introduces indeterminism, however it could be too small.
If you need better a better random selection set :param:shuffle_db as true

It returns a set of pivots as a list of integers pointing to elements in :param:db
"""
function select_sss(db::Array{T,1}, dist::D, alpha::Float64, shuffle_db::Bool=true) where {T,D}
    info("select_sss: db=$(typeof(db)), alpha=$(alpha), distance=$(dist), shuffle_db=$(shuffle_db)")
    dmax::Float64 = 0.0
    s = Int(round(sqrt(length(db))/2))
    sample1, sample2 = rand(db, s), rand(db, s)
    dmax = maximum([dist(sample1[i], sample2[i]) for i=1:s])
    # info("computing", length(X))
    # plot(x=X, Geom.histogram(bincount=100))

    info("the maximum distance estimated as $(dmax), now selecting pivots")
    xdb = Array(1:length(db))
    if shuffle_db
        shuffle!(xdb)
    end
    #pivots::Array{T,1} = Array(T, 1)
    #pivots[1] = xdb[1]
    pivots::Array{Int,1} = Array(Int, 1)
    pivots[1] = xdb[1]
    for i=2:length(xdb)
        obj = db[xdb[i]]
        if minimum([dist(db[pivID], obj) for pivID in pivots]) / dmax < alpha
            continue
        end
        # push!(pivots, obj)
        push!(pivots, i)
    end

    return pivots
end
