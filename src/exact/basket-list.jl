"""BasketList: simple inverted-list representation of baskets.

A BasketList stores a collection of baskets (each basket is a collection
of metric items)
"""

export BasketList

mutable struct BasketList{DIST<:PreMetric,DB<:AbstractDatabase} <: AbstractSearchIndex
    dist::DIST
    db::DB
    baskets::Vector{Vector{IdDist}}
    numitems::Int # number of total objects in all baskets
end

distance(bl::BasketList) = bl.dist
database(bl::BasketList) = bl.db
database(bl::BasketList, i) = bl.db[i]
Base.length(bl::BasketList) = bl.numitems

function BasketList(dist::PreMetric, db::AbstractDatabase, k::Int)
    if !(dist isa Metric)
        @warn "BasketList is designed for metric distances, using it with a non-metric distance ($(typeof(dist))) may yield incorrect results and is not recommended"
    end

    C = SimilaritySearch.fft(dist, db, k)
    # (; centers, nn, dists=nndists, dmax)
    codes = Dict{UInt32,UInt32}()

    for nn in C.nn
        basketID = get(codes, nn, zero(UInt32))
        if basketID === zero(UInt32)
            codes[nn] = length(codes) + 1
        end
    end

    baskets = [IdDist[] for _ in 1:length(codes)]

    for (objID, (nn, d)) in enumerate(zip(C.nn, C.dists))
        basketID = codes[nn] # get the basket id for this center
        L = baskets[basketID]
        if length(L) == 0
            push!(L, IdDist(nn, d)) # the header contains the center's object ID and the covering radius
        end

        push!(L, IdDist(objID, d)) # the 2nd to the end stores the oject ID and its distance to the center (dco)
        
        if L[1].dist < d # updates the covering radius if needed (at header)
            L[1] = IdDist(nn, d)
        end
    end

    BasketList{typeof(dist),typeof(db)}(dist, db, baskets, length(db))
end

# search for an item in baskets using metric properties to discard baskets that cannot contain it
function search(bl::BasketList, _::AbstractContext, query, res::AbstractKnn)
    DIST = distance(bl)
    cost = 0
    k = maxlength(res)
    α = 1f0 # an α factor to reduce the radius of the query ball to increase the chance of discarding baskets, can be tuned for better performance   
    for L in bl.baskets
        length(L) == 0 && continue

        center = L[1]
        cost += 1
        dcq = Dist.evaluate(DIST, database(bl, center.id), query)  # get the distance from the query to the center of the basket

        # check if any item in the basket can be in the ball center at center
        if length(res) < k || dcq <= (maximum(res) + center.dist) * α # the ball centered at center.id with radius center.dist intersects with the query ball
            for item in L[2:end]
                dco = item.dist  # distance from the center to the item
                if length(res) < k || abs(dcq - dco) <= maximum(res) * α  # check if the item can be in the ball of radius maximum(res) around the query
                    cost += 1
                    d = Dist.evaluate(DIST, database(bl, item.id), query)
                    push_item!(res, IdDist(item.id, d))
                end
            end
        end
    end

    add_distance_evaluations!(res, cost)
    res
end

