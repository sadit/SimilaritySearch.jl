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

`evaluate(::Levenshtein, a, b)` uses a small pool of scratch buffers (`Cpool`, a
`Channel{Vector{Int16}}`): each call `take!`s a buffer, uses it, and `put!`s it back
(inside a `try/finally`, so a thrown exception can't leak it). This has no dependency on
thread identity at all -- unlike `Threads.threadid()`-indexing, it is safe under *every*
`@BATCHES` scheduler (`:static`/`:default`/`:greedy`), and under any other concurrency
model too (e.g. calling `evaluate` from a user's own `Threads.@spawn` code), since
correctness never relies on which thread/task happens to run a given call. A smaller pool
only ever costs *throughput* (a `take!` blocks until another call returns a buffer), never
correctness.

`ctx` (a `GenericContext`/`SearchGraphContext`, anything with a `.maxbatches` field) is
accepted so the pool's size can be driven by the same `maxbatches` knob used everywhere
else in this package, instead of a bare `Threads.maxthreadid()`; either way the pool is
clamped to at least 1 buffer (a zero-sized pool would deadlock on the first call).
"""
struct Levenshtein <: Metric
    icost::Int32 # insertion cost
    dcost::Int32 # deletion cost
    rcost::Int32 # replace cost

    Cpool::Channel{Vector{Int16}}
end

function _levenshtein_pool(capacity::Integer)
    n = max(1, Int(capacity))
    pool = Channel{Vector{Int16}}(n)
    for _ in 1:n
        put!(pool, Vector{Int16}(undef, 64))
    end
    pool
end

Levenshtein(; icost=1, dcost=1, rcost=1) =
    Levenshtein(icost, dcost, rcost, _levenshtein_pool(Threads.maxthreadid()))

Levenshtein(ctx; icost=1, dcost=1, rcost=1) =
    Levenshtein(icost, dcost, rcost, _levenshtein_pool(ctx.maxbatches))

"""
    evaluate(::Levenshtein, a, b)

Computes the edit distance between two strings, this is a low level function
"""
function evaluate(lev::Levenshtein, a, b)::Float32
    if length(a) < length(b)
        a, b = b, a
    end

    alen = length(a)
    blen = length(b)

    alen == 0 && return Float32(blen)
    blen == 0 && return Float32(alen)

    C = take!(lev.Cpool)
    try
        resize!(C, blen + 1)
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
    finally
        put!(lev.Cpool, C)
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

`evaluate(::DamerauLevenshtein, a, b)` uses the same `Cpool` scratch-buffer-pool trick as
[`Levenshtein`](@ref) (see its docstring for the rationale); the only difference is that
three rolling rows (current, previous, and two-rows-back, for the transposition lookback)
share one scratch buffer instead of one.
"""
struct DamerauLevenshtein <: SemiMetric
    icost::Int32 # insertion cost
    dcost::Int32 # deletion cost
    rcost::Int32 # replace cost
    tcost::Int32 # transposition cost

    Cpool::Channel{Vector{Int16}}
end

function _damerau_levenshtein_pool(capacity::Integer)
    n = max(1, Int(capacity))
    pool = Channel{Vector{Int16}}(n)
    for _ in 1:n
        put!(pool, Vector{Int16}(undef, 3 * 64))
    end
    pool
end

DamerauLevenshtein(; icost=1, dcost=1, rcost=1, tcost=1) =
    DamerauLevenshtein(icost, dcost, rcost, tcost, _damerau_levenshtein_pool(Threads.maxthreadid()))

DamerauLevenshtein(ctx; icost=1, dcost=1, rcost=1, tcost=1) =
    DamerauLevenshtein(icost, dcost, rcost, tcost, _damerau_levenshtein_pool(ctx.maxbatches))

"""
    evaluate(::DamerauLevenshtein, a, b)

Computes the restricted Damerau-Levenshtein (OSA) distance between two strings, this is a
low level function
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
    buf = take!(dl.Cpool)
    try
        resize!(buf, 3w)
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
    finally
        put!(dl.Cpool, buf)
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
