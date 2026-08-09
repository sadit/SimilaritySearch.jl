"""BasketList: simple inverted-list representation of baskets.

A BasketList stores a collection of baskets (each basket is a collection
of metric items)
"""

export BasketList

mutable struct BasketList{DIST<:PreMetric,DB<:AbstractDatabase} <: AbstractSearchIndex
    dist::DIST
    db::DB
    baskets_ids::Vector{Vector{UInt32}}
    baskets_dists::Vector{Vector{Float32}}
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

    baskets_ids = [UInt32[] for _ in 1:length(codes)]
    baskets_dists = [Float32[] for _ in 1:length(codes)]

    for (objID, (nn, d)) in enumerate(zip(C.nn, C.dists))
        basketID = codes[nn] # get the basket id for this center
        L_ids = baskets_ids[basketID]
        L_dists = baskets_dists[basketID]
        if length(L_ids) == 0
            push!(L_ids, nn) # the header contains the center's object ID and the covering radius
            push!(L_dists, d)
        end

        push!(L_ids, objID) # the 2nd to the end stores the oject ID and its distance to the center (dco)
        push!(L_dists, d)
        
        if L_dists[1] < d # updates the covering radius if needed (at header)
            L_ids[1] = nn
            L_dists[1] = d
        end
    end

    BasketList{typeof(dist),typeof(db)}(dist, db, baskets_ids, baskets_dists, length(db))
end

# search for an item in baskets using metric properties to discard baskets that cannot contain it
function search(bl::BasketList, ctx::AbstractContext, query, res::AbstractKnn)
    DIST = distance(bl)
    cost = 0
    k = maxlength(res)
    α = 1f0 # an α factor to reduce the radius of the query ball to increase the chance of discarding baskets, can be tuned for better performance   
    for (L_ids, L_dists) in zip(bl.baskets_ids, bl.baskets_dists)
        length(L_ids) == 0 && continue

        c_id = L_ids[1]
        c_dist = L_dists[1]
        cost += 1
        dcq = Dist.evaluate(DIST, database(bl, c_id), query)  # get the distance from the query to the center of the basket

        # check if any item in the basket can be in the ball center at center
        if length(res) < k || dcq <= (maximum(res) + c_dist) * α # the ball centered at center.id with radius center.dist intersects with the query ball
            for i in 2:length(L_ids)
                item_id = L_ids[i]
                dco = L_dists[i]  # distance from the center to the item
                if length(res) < k || abs(dcq - dco) <= maximum(res) * α  # check if the item can be in the ball of radius maximum(res) around the query
                    cost += 1
                    d = Dist.evaluate(DIST, database(bl, item_id), query)
                    push_item!(res, item_id, d)
                end
            end
        end
    end

    add_distance_evaluations!(ctx, cost)
    res
end
