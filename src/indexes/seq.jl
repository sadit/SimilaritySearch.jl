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

# abstract Sequential

#type Seq <: Sequential
#include("Result.jl")
import Base: search, push!

export Sequential, search, push!, save

struct Sequential{T, D} <: Index
    db::Array{T,1}
    dist::D
end

function Sequential{T, D}(filename::AbstractString, db::Array{T,1}, dist::D)
    header = JSON.parse(open(readlines, filename)[1])
    if header["length"] != length(db)
        throw(ArgumentError("the database's length doesn't match"))
    end
    index = Sequential(db, dist)
    if header["type"] != string(typeof(index))
        throw(ArgumentError("the index's type doesn't match"))
    end
    return index
end

function save{T, D}(index::Sequential{T, D}, filename::AbstractString)
    f = open(filename, "w")
    header = Dict(
        "length" => length(index.db),
        "type" => string(typeof(index)),
    )
    write(f, JSON.json(header), "\n")
    close(f)
end

function search{T, D, R <: Result}(index::Sequential{T, D}, q::T, res::R)
    # for i in range(1, length(index.db))
    i::Int32 = 1
    d::Float64 = 0.0
    for obj in index.db
        d = index.dist(q, obj)
        push!(res, i, convert(Float32,d))
        i += 1
    end

    return res
end

function search{T, D}(index::Sequential{T, D}, q::T)
    return search(index, q, NnResult())
end

function push!{T, D}(index::Sequential{T, D}, item::T)
    push!(index.db, item)
    return length(index.db)
end
