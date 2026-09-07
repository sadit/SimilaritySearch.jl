# This file is a part of SimilaritySearch.jl

#export CommonPrefix, Levenshtein, Hamming, LCS

"""
    CommonPrefix()

Uses the common prefix as a measure of dissimilarity between two strings
"""
struct CommonPrefix <: SemiMetric
end

"""
    common_prefix(a, b)

Computes the length of the common prefix among two strings represented as arrays
"""
function common_prefix(a, b)
    len_a = length(a)
    len_b = length(b)
    i = 1
    min_len = min(len_a, len_b)
    @inbounds while i <= min_len && a[i] == b[i]
        i += 1
    end

    i - 1
end

"""
    evaluate(::CommonPrefix, a, b)

Computes a dissimilarity based on the common prefix between two strings
"""
evaluate(::CommonPrefix, a, b)::Float32 = 1.0f0 - Float32(common_prefix(a, b) / min(length(a), length(b)))


"""
    Levenshtein(; icost=1, dcost=1, rcost=1)
    Levenshtein(ctx; icost=1, dcost=1, rcost=1)

The levenshtein distance measures the minimum number of edit operations to convert one string into another.
The costs insertion `icost`, deletion cost `dcost`, and replace cost `rcost`.

## Scratch buffers, and why there is no lock

`evaluate` needs a row of `Int16`s (plus, for `AbstractString` inputs, a `Vector{Char}`).
A `Levenshtein` built the ordinary way owns **no** scratch and allocates what it needs per
call, which is what makes it safe to share across tasks unconditionally -- there is no
mutable state to race on, under any scheduler or any concurrency model of your own.

To skip even that allocation, ask [`beginbatch`](@ref) for a batch-local copy at the top of
a `@BATCHES` batch (`bdist = beginbatch(distance(index))`) and use *that* inside the batch.
The copy owns private buffers it grows once and reuses; a batch is single-tasked, so nothing
guards them and nothing needs to.

This replaced a `Channel`-based buffer pool. The pool was safe, but its `take!`/`put!` pair
per call turned out to cost far more than the work it was protecting whenever evaluations
are cheap: on a 200k-element parallel map of 5-10 character words over 64 threads it was
**~80x slower** than allocating (302ms vs 3.6ms), and it lost even single-threaded.

`ctx` is still accepted and ignored, so old call sites keep working -- it used to size the
pool. It will go away in 2.0.

## `AbstractString` inputs (`String`, `SubString`, ...)

`a`/`b` can be passed as plain `String`/`SubString` directly -- Unicode included -- with
no need to `collect` them into a `Vector{Char}` first. A dedicated method (see below)
walks each string with Julia's string-iteration protocol (`for c in s`, the efficient,
allocation-free equivalent of repeatedly calling `nextind`) instead of integer-indexing
`s[i]` for `i in 1:length(s)`, which is what the *generic* `evaluate(::Levenshtein, a, b)`
method above does and why it throws `StringIndexError` on a `String`/`SubString`
containing non-ASCII characters (a `String` is indexed by *codeunit* -- a byte, for its
UTF-8 encoding -- not by character, and only ASCII characters take exactly one codeunit
each). The shorter of the two strings is decoded once into a second pooled scratch buffer
(see the scratch section above) so it can be randomly indexed inside the O(alen*blen)
dynamic-programming loop; the longer one is walked forward-only and never needs random
access, so it costs nothing beyond that same forward pass.
"""
struct Levenshtein <: Metric
    icost::Int32 # insertion cost
    dcost::Int32 # deletion cost
    rcost::Int32 # replace cost

    # Private scratch, empty unless this is a `beginbatch` copy -- see the docstring. Empty
    # means "own nothing, allocate per call", which is what makes a shared instance safe.
    C::Vector{Int16}
    B::Vector{Char}
end

Levenshtein(; icost=1, dcost=1, rcost=1) = Levenshtein(icost, dcost, rcost, Int16[], Char[])
Levenshtein(ctx; icost=1, dcost=1, rcost=1) = Levenshtein(; icost, dcost, rcost)

"""
    beginbatch(lev::Levenshtein)

A copy owning private scratch buffers, for use inside a single `@BATCHES` batch. See
[`Levenshtein`](@ref) and [`beginbatch`](@ref).
"""
beginbatch(lev::Levenshtein) =
    Levenshtein(lev.icost, lev.dcost, lev.rcost, Vector{Int16}(undef, 64), Vector{Char}(undef, 64))

"Scratch of length `n`: the batch-local buffer when there is one, a fresh one otherwise."
@inline _scratch(buf::Vector, n::Integer) = isempty(buf) ? similar(buf, n) : (resize!(buf, n); buf)

"""
    evaluate(::Levenshtein, a, b)

Computes the edit distance between two sequences (e.g. arrays with `==`-comparable
elements), this is a low level function. See [`Levenshtein`](@ref) for the
`AbstractString`-specialized method used for `String`/`SubString` inputs.
"""
function evaluate(lev::Levenshtein, a, b)::Float32
    if length(a) < length(b)
        a, b = b, a
    end

    alen = length(a)
    blen = length(b)

    alen == 0 && return Float32(blen)
    blen == 0 && return Float32(alen)

    C = _scratch(lev.C, blen + 1)
    begin
        @inbounds for i in 0:blen
            C[i+1] = i
        end

        prevA = 0
        @inbounds for i in 1:alen
            prevA = i
            prevC = C[1]
            j = 1

            while j <= blen
                cost = a[i] == b[j] ? 0 : lev.rcost
                C[j] = prevA
                j += 1
                prevA = min(C[j] + lev.dcost, prevA + lev.icost, prevC + cost)
                prevC = C[j]
            end

            C[j] = prevA
        end

        Float32(prevA)
    end
end

"""
    evaluate(::Levenshtein, a::AbstractString, b::AbstractString)

Computes the edit distance between two `AbstractString`s (`String`, `SubString`, ...),
handling Unicode correctly without requiring the caller to `collect` into a `Vector{Char}`.
See [`Levenshtein`](@ref) for how this differs from the generic array method.
"""
function evaluate(lev::Levenshtein, a::AbstractString, b::AbstractString)::Float32
    if length(a) < length(b)
        a, b = b, a
    end

    alen = length(a)
    blen = length(b)

    alen == 0 && return Float32(blen)
    blen == 0 && return Float32(alen)

    Bbuf = _scratch(lev.B, blen)
    begin
        @inbounds for (j, c) in enumerate(b)
            Bbuf[j] = c
        end

        C = _scratch(lev.C, blen + 1)
        begin
            @inbounds for i in 0:blen
                C[i+1] = i
            end

            prevA = 0
            @inbounds for (i, ai) in enumerate(a)
                prevA = i
                prevC = C[1]
                j = 1

                while j <= blen
                    cost = ai == Bbuf[j] ? 0 : lev.rcost
                    C[j] = prevA
                    j += 1
                    prevA = min(C[j] + lev.dcost, prevA + lev.icost, prevC + cost)
                    prevC = C[j]
                end

                C[j] = prevA
            end

            Float32(prevA)
        end
    end
end


"""
    DamerauLevenshtein(; icost=1, dcost=1, rcost=1, tcost=1)
    DamerauLevenshtein(ctx; icost=1, dcost=1, rcost=1, tcost=1)

The restricted Damerau-Levenshtein distance (a.k.a. Optimal String Alignment, OSA):
[`Levenshtein`](@ref) extended with a fourth edit operation, the transposition of two
*adjacent* characters, at cost `tcost`. This captures a common typo pattern, e.g.
`"form"` -> `"from"` (the middle `"or"` swapped to `"ro"`), as a single edit instead of
two substitutions.

This is the *restricted* variant: it disallows editing a substring that already
participated in a transposition again, which is what keeps the algorithm inside the same
row-by-row scratch-buffer scheme as [`Levenshtein`](@ref) (a small extra lookback row,
rather than a full `O(alen*blen)` matrix). The consequence is that this distance is a
`SemiMetric`, not a `Metric`: it satisfies `d(a,a) == 0` and `d(a,b) == d(b,a)`, but *not*
the triangle inequality (e.g. `evaluate(dl, "ca", "abc")` can exceed
`evaluate(dl, "ca", "ac") + evaluate(dl, "ac", "abc")`) -- the unrestricted/"true"
Damerau-Levenshtein distance that does satisfy it needs the full matrix and is not
implemented here.

## `AbstractString` inputs (`String`, `SubString`, ...)

`a`/`b` can be passed as plain `String`/`SubString` directly -- Unicode included -- via a
dedicated method (see below); see [`Levenshtein`](@ref)'s docstring for why the generic
`evaluate(::DamerauLevenshtein, a, b)` method above throws `StringIndexError` on those
inputs and how the `AbstractString` method avoids it (string-iteration instead of
`s[i]`-indexing, plus a scratch buffer holding the shorter string's characters).

`evaluate(::DamerauLevenshtein, a, b)` handles scratch exactly as [`Levenshtein`](@ref)
does -- allocated per call unless [`beginbatch`](@ref) handed out a batch-local copy, never
locked -- the only difference being that three rolling rows (current, previous, and
two-rows-back, for the transposition lookback) share one buffer instead of one row using it.
"""
struct DamerauLevenshtein <: SemiMetric
    icost::Int32 # insertion cost
    dcost::Int32 # deletion cost
    rcost::Int32 # replace cost
    tcost::Int32 # transposition cost

    # private scratch, empty unless this is a `beginbatch` copy -- see [`Levenshtein`](@ref)
    C::Vector{Int16}
    B::Vector{Char}
end

DamerauLevenshtein(; icost=1, dcost=1, rcost=1, tcost=1) =
    DamerauLevenshtein(icost, dcost, rcost, tcost, Int16[], Char[])

DamerauLevenshtein(ctx; icost=1, dcost=1, rcost=1, tcost=1) =
    DamerauLevenshtein(; icost, dcost, rcost, tcost)

"""
    beginbatch(dl::DamerauLevenshtein)

A copy owning private scratch buffers, for use inside a single `@BATCHES` batch. See
[`Levenshtein`](@ref) and [`beginbatch`](@ref).
"""
beginbatch(dl::DamerauLevenshtein) =
    DamerauLevenshtein(dl.icost, dl.dcost, dl.rcost, dl.tcost,
        Vector{Int16}(undef, 3 * 64), Vector{Char}(undef, 64))

"""
    evaluate(::DamerauLevenshtein, a, b)

Computes the restricted Damerau-Levenshtein (OSA) distance between two sequences (e.g.
arrays with `==`-comparable elements), this is a low level function. See
[`DamerauLevenshtein`](@ref) for the `AbstractString`-specialized method used for
`String`/`SubString` inputs.
"""
function evaluate(dl::DamerauLevenshtein, a, b)::Float32
    if length(a) < length(b)
        a, b = b, a
    end

    alen = length(a)
    blen = length(b)

    alen == 0 && return Float32(blen)
    blen == 0 && return Float32(alen)

    w = blen + 1
    buf = _scratch(dl.C, 3w)
    begin
        twoAgo = view(buf, 1:w)
        prevRow = view(buf, w+1:2w)
        curRow = view(buf, 2w+1:3w)

        @inbounds for j in 0:blen
            prevRow[j+1] = j
        end

        @inbounds for i in 1:alen
            curRow[1] = i
            ai = a[i]

            for j in 1:blen
                cost = ai == b[j] ? 0 : dl.rcost
                del = prevRow[j+1] + dl.dcost
                ins = curRow[j] + dl.icost
                sub = prevRow[j] + cost
                best = min(del, ins, sub)

                if i > 1 && j > 1 && ai == b[j-1] && a[i-1] == b[j]
                    best = min(best, twoAgo[j-1] + dl.tcost)
                end

                curRow[j+1] = best
            end

            twoAgo, prevRow, curRow = prevRow, curRow, twoAgo
        end

        Float32(prevRow[blen+1])
    end
end

"""
    evaluate(::DamerauLevenshtein, a::AbstractString, b::AbstractString)

Computes the restricted Damerau-Levenshtein (OSA) distance between two `AbstractString`s
(`String`, `SubString`, ...), handling Unicode correctly without requiring the caller to
`collect` into a `Vector{Char}`. See [`DamerauLevenshtein`](@ref) for how this differs
from the generic array method.
"""
function evaluate(dl::DamerauLevenshtein, a::AbstractString, b::AbstractString)::Float32
    if length(a) < length(b)
        a, b = b, a
    end

    alen = length(a)
    blen = length(b)

    alen == 0 && return Float32(blen)
    blen == 0 && return Float32(alen)

    Bbuf = _scratch(dl.B, blen)
    begin
        @inbounds for (j, c) in enumerate(b)
            Bbuf[j] = c
        end

        w = blen + 1
        buf = _scratch(dl.C, 3w)
        begin
            twoAgo = view(buf, 1:w)
            prevRow = view(buf, w+1:2w)
            curRow = view(buf, 2w+1:3w)

            @inbounds for j in 0:blen
                prevRow[j+1] = j
            end

            aim1 = first(a)
            @inbounds for (i, ai) in enumerate(a)
                curRow[1] = i

                for j in 1:blen
                    bj = Bbuf[j]
                    cost = ai == bj ? 0 : dl.rcost
                    del = prevRow[j+1] + dl.dcost
                    ins = curRow[j] + dl.icost
                    sub = prevRow[j] + cost
                    best = min(del, ins, sub)

                    if i > 1 && j > 1 && ai == Bbuf[j-1] && aim1 == bj
                        best = min(best, twoAgo[j-1] + dl.tcost)
                    end

                    curRow[j+1] = best
                end

                twoAgo, prevRow, curRow = prevRow, curRow, twoAgo
                aim1 = ai
            end

            Float32(prevRow[blen+1])
        end
    end
end


"""
    Hamming()

The hamming distance counts the differences between two equally sized strings
"""
struct Hamming <: Metric
end

"""
     evaluate(::Hamming, a, b)
     
Computes the hamming distance between two sequences of the same length
"""
function evaluate(::Hamming, a, b)::Float32
    d = 0

    @inbounds for i in 1:length(a)
        d += Int(a[i] != b[i])
    end

    Float32(d)
end


"""
    LCS()
    LCS(ctx)

Instantiates a Levenshtein object to perform LCS distance. See [`Levenshtein`](@ref) for
the meaning of `ctx` (optional; sizes the internal scratch pool from `ctx.maxbatches`).
"""
struct LCS <: Metric
    lev::Levenshtein
    LCS() = new(Levenshtein(rcost=2))
    LCS(ctx) = new(Levenshtein(ctx; rcost=2))
end

@inline evaluate(lcs::LCS, a, b) = evaluate(lcs.lev, a, b)

# function kerrormatch(a::T1, b::T2, errors::Integer)::Bool where {T1 <: Any,T2 <: Any}
#     # if length(a) < length(b)
#     #     a, b = b, a
#     # end

#     alen::Int = length(a)
#     blen::Int = length(b)

#     alen == 0 && return alen == blen
#     blen == 0 && return true

#     C::Vector{Int} = Vector{Int}(0:blen)

#     @inbounds for i in 1:alen
#         prevA::Int = 0
#         prevC::Int = C[1]
#         j::Int = 1

#         while j <= blen
#             cost::Int = 1
#             if a[i] == b[j]
#                 cost = 0
#             end
#             C[j] = prevA
#             j += 1
#             prevA = min(C[j]+1, prevA+1, prevC+cost)
#             prevC = C[j]
# 	    end

#         C[j] = prevA
#         if prevA <= errors
#             return true
#         end
#     end

#     return false
# end

# function best_match_levenshtein(a::T1, b::T2)::Int where {T1 <: Any,T2 <: Any}
#     # if length(a) < length(b)
#     #     a, b = b, a
#     # end

#     alen::Int = length(a)
#     blen::Int = length(b)

#     alen == 0 && return blen
#     blen == 0 && return alen

#     C::Vector{Int} = 1:blen |> collect

#     mindist = alen
#     @inbounds for i in 1:alen
#         prevA::Int = 0
#         prevC::Int = C[1]
#         j::Int = 1

#         while j <= blen
#             cost::Int = 1
#             if a[i] == b[j]
#                 cost = 0
#             end
#             C[j] = prevA
#             j += 1
#             prevA = min(C[j]+1, prevA+1, prevC+cost)
#             prevC = C[j]
#         end

#         C[j] = prevA
#         if prevA < mindist
#             mindist = prevA
#         end
#     end

#     return mindist
# end
