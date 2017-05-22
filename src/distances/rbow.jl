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

export RBOW, Token

struct Token
    id::String
    weight::Float32
end

mutable struct RBOW
    terms::Vector{Token}
    invnorm::Float32
end

function RBOW(terms::Vector{Token})
    xnorm::Float32 = 0.0
    @fastmath @inbounds @simd for i = 1:length(terms)
        xnorm += terms[i].weight ^ 2
    end

    if length(terms) > 0
        @fastmath xnorm = 1/sqrt(xnorm)
    end
    RBOW(terms, xnorm)
end

function RBOW(bow::Dict{String,F}) where {F <: Real}
    M = Vector{Token}(length(bow))

    i = 1
    for (key, value) in bow
        M[i] = Token(key, convert(Float32, value))
        i+=1
    end

    sort!(M, by=(x)->x.id)
    RBOW(M)
end

Base.length(a::RBOW) = length(a.terms)

function (o::CosineDistance)(a::RBOW, b::RBOW)::Float64
    o.calls += 1
    return 1.0 - cos(a, b)
end

function (o::AngleDistance)(a::RBOW, b::RBOW)
    o.calls += 1
    c = cos(a, b)
    c = max(c, -1)
    c = min(c, 1)
    return acos(c)
end

function cos(a::RBOW, b::RBOW)::Float64
    n1=length(a.terms); n2=length(b.terms)
    # (n1 == 0 || n2 == 0) && return 0.0

    sum::Float64 = 0.0
    i = 1; j = 1

    @fastmath @inbounds while i <= n1 && j <= n2
        c = cmp(a.terms[i].id, b.terms[j].id)
        if c == 0
            sum += a.terms[i].weight * b.terms[j].weight
            i += 1
            j += 1
        elseif c < 0
            i += 1
        else
            j += 1
        end
    end

    return sum * a.invnorm * b.invnorm
end

function save(ostream, item::RBOW)
    write(ostream, length(item.terms) |> Int32)
    for x in item.terms
        write(ostream, x.id, x.weight)
    end
    write(ostream, item.invnorm)
end

function load(istream, ::Type{RBOW})::RBOW
    len = read(istream, Int32)
    vec = Vector{Token}(len)
    @inbounds for i in 1:len
        vec[i] = Token(read(istream, Int32), read(istream, Float32))
    end
    invnorm = read(istream, Float32)
    return RBOW(vec, invnorm)
end

function saves(item::RBOW)
    join([string(t.id, ' ', t.weight) for x in item.vec], ' ')
end

function loads(line::String, ::Type{RBOW})::RBOW
    X = split(line, ' ')
    vec = Vector{Token}()
    for i in 1:2:length(X)
        term = Token(X[i], parse(Float32, X[i+1]))
        push!(vec, term)
    end

    return RBOW(vec)
end
