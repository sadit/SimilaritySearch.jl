# This file is part of SimilaritySearch.jl

function satpermutation!(π, sat::Sat)
    p = 1
    π[p] = sat.root
    cand = [sat.root]
    sizehint!(cand, ceil(Int, sqrt(length(π))))

    while length(cand) > 0
        i = popfirst!(cand)
        for c in (sat.children[i])::Vector{UInt32}
            p += 1
            π[p] = c
            if sat.children[c] !== nothing
                push!(cand, c)
            end
        end
    end

    π
end

satpermutation(sat::Sat) = satpermutation!(Vector{UInt32}(undef, length(sat)), sat)

"""
    permutesat(sat::Sat, π=satpermutation(sat), π′=invperm(π); DBType=MatrixDatabase)

Permute `sat` to optimize cache access patterns; the database is copied (materialized via
`DBType`) and permuted. The permuted index is stored in a `PermutedSearchIndex` to allow
plug-and-play index interchange with the unpermuted `sat`.

# Arguments
- `sat`: input `Sat` index.
- `π`: permutation.
- `π′`: inverse permutation.
- `DBType`: database type used to materialize the permuted copy (defaults to
  `MatrixDatabase`; use `VectorDatabase` for non-vector object types).
"""
function permutesat(sat::Sat, π=satpermutation(sat), π′=invperm(π); DBType::Type{<:AbstractDatabase}=MatrixDatabase)
    db = DBType(SubDatabase(database(sat), π))
    children = similar(sat.children)
    fill!(children, nothing)

    for ii in eachindex(children)
        C = sat.children[ii]
        C === nothing && continue
        CC = copy(C)
        children[π′[ii]] = CC

        for (i, c) in enumerate(C)
            CC[i] = π′[c]
        end
    end

    cov = similar(sat.cov)
    for i in eachindex(cov)
        cov[π′[i]] = sat.cov[i]
    end

    s = Sat(distance(sat), db, π[sat.root], children, cov)
    PermutedSearchIndex(; index=s, π, π′)
end
