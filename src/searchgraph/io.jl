# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::SearchGraph, meta::Dict)

Saves a SearchGraph index optimizing for large indexes
"""
function saveindex(filename::AbstractString, index::SearchGraph, meta::Dict)
    n = length(index.links)
    s = 0
    for L in index.links  # saving/loading vectors of vectors seems to be pretty bad
        s += length(L)
    end

    links = Vector{Int32}(undef, n + s + 1)
    links[1] = n
    i = 2

    for L in index.links
        links[i] = length(L)
        i += 1
        for l in L
            links[i] = l
            i += 1
        end
    end

    index = copy(index; links=Vector{Int32}[])
    jldsave(filename; index, meta, links)
end

function restoreindex(index::SearchGraph, meta::Dict, f)
    L = f["links"]::Vector{Int32}
    n = L[1]
    i = 2
    links = Vector{Vector{Int32}}(undef, n)
    for objID in 1:n
        len = L[i]
        i += 1
        children = Vector{Int32}(undef, len)

        for j in 1:len
            children[j] = L[i]
            i += 1
        end

        links[objID] = children
    end
 
    copy(index; links), meta
end
