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

export Knr, optimize!

mutable struct Knr{T, D} <: Index
    db::Vector{T}
    dist::D
    refs::Vector{T}
    k::Int
    ksearch::Int
    minmatches::Int
    invindex::Vector{Vector{Int32}}
    # seqlist::Array{Array{I,1},1}
end

function save{T, D}(index::Knr{T, D}, filename::AbstractString)
    saveDB(index.refs, "$(filename).refs")
    saveDB(index.invindex, "$(filename).invindex")

    f = open(filename, "w")
    header = Dict(
        "length" => length(index.db),
        "type" => string(typeof(index)),
        "numrefs" => length(index.refs),
        "k" => index.k,
        "ksearch" => index.ksearch,
        "minmatches" => index.minmatches,
    )
    write(f, JSON.json(header, 2), "\n")
    close(f)

end

function Knr{T, D}(filename::AbstractString, db::Array{T,1}, dist::D)
    refs = loadDB(T, "$(filename).refs")
    invindex = loadDB(Array{Int32,1}, "$(filename).invindex")
    header = JSON.parsefile(filename)
    if length(refs) != header["numrefs"] || length(db) != header["length"]
        warn("length(db) or length(refs) doesn't match with those given in $(filename)")
    end

    return Knr(db, dist, refs, header["k"], header["ksearch"], header["minmatches"], invindex)
end

function Knr{T, D}(db::Array{T,1}, dist::D, refs::Array{T,1}, k::Int, minmatches::Int=1)
    info("Knr, refs=$(typeof(db)), k=$(k), numrefs=$(length(refs)), dist=$(dist)")
    invindex = [Array(Int32, 0) for i in 1:length(refs)]
    seqindex = Sequential(refs, dist)

    pc = Int(round(length(db) / 20))
    for i=1:length(db)
        if i % pc == 0
            info("advance $(round(i/length(db), 4)), now: $(now())")
        end

        res = search(seqindex, db[i], KnnResult(k))
        for p in res
            push!(invindex[p.objID], i)
        end
    end

    Knr(db, dist, refs, k, k, minmatches, invindex)
end

function Knr{T, D}(db::Array{T,1}, dist::D, numrefs::Int, k::Int, minmatches::Int=1)
    # refs = rand(db, numrefs)
    refs = [db[x] for x in select_tournament(db, dist, numrefs)]
    Knr(db, dist, refs, k, minmatches)
end

function search{T, D, R <: Result}(index::Knr{T, D}, q::T, res::R)
    dz = zeros(Int8, length(index.db))
    # M = BitArray(length(index.db))
    seqindex = Sequential(index.refs, index.dist)
    kres = search(seqindex, q, KnnResult(index.ksearch))

    for p in kres
        @inbounds for objID in index.invindex[p.objID]
            c = dz[objID] + 1
            dz[objID] = c

            if c == index.minmatches
                # if !M[objID]
                # M[objID] = true
                @inbounds d = index.dist(q, index.db[objID])
                push!(res, objID, convert(Float32, d))
            end
        end
    end

    return res
end

# type knr_join_tuple
#     curr::Int
#     list::Vector{Int}
# end

# function isless(a::knr_join_tuple, b::knr_join_tuple)
#     return @inbounds isless(a.list[a.curr], b.list[b.curr])
# end

# function search{T, D <: Distance, R <: Result}(index::Knr{T, D}, q::T, res::R)
#     seqindex = Sequential(index.refs, index.dist)
#     knr = search(seqindex, q, KnnResult(index.ksearch))
#     queue = Array(knr_join_tuple, 0)
    
#     for item in knr
#         @inbounds list = index.invindex[item.objID]
#         if length(list) > 0
#             t = knr_join_tuple(1, list)
#             @inbounds push!(queue, t)
#         end
#     end
    
#     Collections.heapify!(queue)
#     prevID::Int = 0
#     while length(queue) > 0
#         t = Collections.heappop!(queue)
#         @inbounds currID::Int = t.list[t.curr]
#         if currID != prevID
#             # if c == index.minmatches
#             # end
#             @inbounds d = index.dist(q, index.db[currID])
#             @inbounds push!(res, currID, d)
#         end
#         prevID = currID
        
#         if t.curr < length(t.list)
#             t.curr +=1
#             @inbounds Collections.heappush!(queue, t)
#         end
#     end
#     return res
# end

function search{T, D}(index::Knr{T, D}, q::T)
    return search(index, q, NnResult())
end

function push!{T,  D}(index::Knr{T, D}, obj::T)
    push!(index.db, obj)
    seqindex = Sequential(index.refs, index.dist)
    res = search(seqindex, obj, KnnResult(index.k))
    for p in res
        push!(index.invindex[p.objID], length(index.db))
    end
    return length(index.db)
end

function optimize!{T,  D}(index::Knr{T, D}; recall::Float64=0.9, k::Int=1, numqueries::Int=128)
    info("optimizing Knr index for recall=$(recall)")
    perf = Performance(index.db, index.dist, numqueries, k)
    index.minmatches = 1
    index.ksearch = 1
    p = probe(perf, index)

    while p.recall < recall && index.ksearch < length(index.refs)
        index.ksearch += 1
        p = probe(perf, index)
    end
    info("reached performance $(p)")
    return index
end
