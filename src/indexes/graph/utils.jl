
function estimate_knearest{T, D, R, R2, M}(db::Vector{T}, dist::D, choosek::Int, from::Int, q::T, tabu::M, res::R, near::R2)::R2
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

function estimate_knearest{T, D, R, M}(db::Vector{T}, dist::D, choosek::Int, from::Int, q::T, tabu::M, res::R)::KnnResult
    near = KnnResult(choosek)
    estimate_knearest(db, dist, choosek, from, q, tabu, res, near)
end

function estimate_from_oracle{T, D, R, M}(index::LocalSearchIndex{T,D}, q::T, beam::SlugKnnResult, tabu::M, res::R, oracle::Function)
    for childID in oracle(q)
      if !tabu[childID]
        tabu[childID] = true
        d = convert(Float32, index.dist(index.db[childID], q))
        push!(beam, childID, d) && push!(res, childID, d)
      end
    end
end
