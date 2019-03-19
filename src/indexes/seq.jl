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

import Base: push!

export Sequential, search, push!, fit

"""
    Sequential{T}

A simple exhaustive search index
"""
struct Sequential{T} <: Index
    db::Vector{T}
end

function fit(::Type{Sequential}, db::Vector{T}) where T
    Sequential(db)
end

function search(index::Sequential{T}, dist::Function, q::T, res::KnnResult) where T
    i::Int32 = 1
    d::Float64 = 0.0
    for obj in index.db
        d = dist(q, obj)
        push!(res, i, convert(Float32, d))
        i += 1
    end

    res
end

function push!(index::Sequential{T}, dist::Function, item::T) where T
    push!(index.db, item)
    length(index.db)
end
