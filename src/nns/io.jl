#  Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>
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

# using StaticArrays
export loadDB, saveDB, loadWordVectors, load, save, loads, saves

function loads(line::String, ::Type{Vector{T}})::Vector{T} where {T <: Real}
    T[parse(T, x) for x in split(line)]
end

function saves(vec::Vector{T}) where {T <: Real}
    join([string(x) for x in vec], ' ')
end

function load(istream::IO, ::Type{Vector{T}})::Vector{T} where {T <: Real}
    # T[parse(T, x) for x in split(readline(istream))]
    len = read(istream, Int32)
    vec = Vector{T}(len)
    for i in 1:len
        vec[i] = read(istream, T)
    end
    vec
end

function save(ostream::IO, vec::Vector{T}) where {T <: Real}
    write(ostream, length(vec) |> Int32)
    for i in 1:length(vec)
        write(ostream, vec[i])
    end
end

function loadDB(t::Type, filename::String)
    db = Vector{t}(0)
    f = open(filename)
    i = 0

    for line in eachline(f)
        x = loads(line, t)
        push!(db, x)
        i += 1
        i % 10000 == 0 && info("reading $(filename), advance  $(i)")
    end

    close(f)
    return db
end

function saveDB(db::Vector{T}, filename::AbstractString) where {T}
    f = open(filename, "w")
    for (i, item) in enumerate(db)
        i % 10000 == 0 && info("saving $(filename), advance  $(i) of $(length(db))")
        write(f, saves(item), '\n')
    end
    close(f)
end

function loadWordVectors(filename::AbstractString)
    words = Vector{String}(0)
    db = Vector{Vector{Float32}}(0)

    open(filename) do f
        for (i, line) in enumerate(eachline(f))
            i % 10000 == 0 && info("GloveDB: reading $(filename), advance  $(i)")
            arr = split(line)
            @inbounds item = Float32[parse(Float64, arr[i]) for i=2:length(arr)]
            push!(words, arr[1])
            push!(db, item)
        end
    end

    return (words, db)
end
