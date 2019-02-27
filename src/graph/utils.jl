
function estimate_knearest(dist::Function, db::AbstractVector{T}, choosek::Int, from::Int, q::T, tabu::M, res::Result, near::Result)::Result where {T, M}
    n::Int32 = length(db)
    # from = max(choosek, from)  ## it has no sense from < choosek!
    nrange = 1:n

    for i in 1:from
        nodeID = convert(Int32, rand(nrange))
        @inbounds if !tabu[nodeID]
            d = convert(Float32, dist(db[nodeID], q))
            tabu[nodeID] = true
            push!(near, nodeID, d) && push!(res, nodeID, d)
        end
    end

    near
end

function estimate_knearest(dist::Function, db::AbstractVector{T}, choosek::Int, from::Int, q::T, tabu::M, res::Result)::KnnResult where {T, M}
    near = KnnResult(choosek)
    estimate_knearest(dist, db, choosek, from, q, tabu, res, near)
end

function estimate_from_oracle(index::LocalSearchIndex{T}, dist::Function, q::T, beam::Result, tabu::M, res::Result, oracle::Function) where {T, M}
    for childID in oracle(q)
      if !tabu[childID]
        tabu[childID] = true
        d = convert(Float32, dist(index.db[childID], q))
        push!(beam, childID, d) && push!(res, childID, d)
      end
    end
end
