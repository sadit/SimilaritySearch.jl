#  Copyright 2016, 2017 Eric S. Tellez <eric.tellez@infotec.mx>
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# abstract Sequential

import Base: search, push!

export Sequential, search, push!

"""
    Sequential{T, D}

A simple exhaustive search index
"""
struct Sequential{T, D} <: Index
    db::Array{T,1}
    dist::D
end

function search(index::Sequential{T,D}, q::T, res::Result) where {T, D}
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

function push!(index::Sequential{T, D}, item::T) where {T, D}
    push!(index.db, item)
    return length(index.db)
end
