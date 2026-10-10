# This file is a part of SimilaritySearch.jl


#=struct Vector{UInt64}Bits
    B::Vector{UInt64}
end=#

@inline function _b64indices(i_::UInt64)
    i = i_ - one(UInt64)
    (i >>> 6) + 1, (i & 63)
end

@inline _b64block(i_::UInt64) = ((i_ - one(UInt64)) >>> 6) + 1

"""
    reuse!(B::AbstractVector{UInt64}, n::Integer)

Resets (zeroes out) the bit-vector `B` so that it can be reused to track up to `n` visited vertices, resizing it if needed.
"""
function reuse!(B::AbstractVector{UInt64}, n::Integer)
    n > 0 && let n = convert(UInt64, n), m = _b64block(n)
        m > length(B) && resize!(B, m)
        @inbounds @simd for i in 1:m
            B[i] = xor(B[i], B[i])
        end
    end

    B
end

"""
    check_visited_and_visit!(vstate, i_::Integer)::Bool

Checks that `i_` is already visited:
    - returns true if already visited
    - returns false if is not visited yet, but marks it as visited before return
"""
@inline function check_visited_and_visit!(vstate::AbstractVector{UInt64}, i_::Integer)::Bool
    b, i = _b64indices(i_)
    @inbounds v = Bool((vstate[b] >>> i) & one(UInt64))
    !v && (@inbounds vstate[b] |= (one(UInt64) << i))
    v
end

"""
    visited(vstate::AbstractVector{UInt64}, i_::UInt64)::Bool

Checks whether vertex `i_` is marked as visited in the bit-vector `vstate`.
"""
@inline function visited(vstate::AbstractVector{UInt64}, i_::UInt64)::Bool
    b, i = _b64indices(i_)
    @inbounds (vstate[b] >>> i) & one(UInt64)
end

"""
    visit!(vstate::AbstractVector{UInt64}, i_::UInt64)

Marks vertex `i_` as visited in the bit-vector `vstate`.
"""
@inline function visit!(vstate::AbstractVector{UInt64}, i_::UInt64)
    b, i = _b64indices(i_)
    @inbounds vstate[b] |= (one(UInt64) << i)
    nothing
end

#### Int set

"""
    reuse!(v::Set{UInt32}, n::Integer)

Empties the set `v` so that it can be reused to track visited vertices, pre-sizing it for a dataset of `n` elements.
"""
function reuse!(v::Set{UInt32}, n::Integer)
    empty!(v)
    sizehint!(v, ceil(Int, sqrt(n)))
    v
end

"""
    visited(vstate::Set{UInt32}, i::Integer)::Bool

Checks whether vertex `i` is a member of the visited-vertices set `vstate`.
"""
@inline visited(vstate::Set{UInt32}, i::Integer)::Bool = i ∈ vstate

"""
    visit!(vstate::Set{UInt32}, i::Integer)

Marks vertex `i` as visited by adding it to the set `vstate`.
"""
@inline visit!(vstate::Set{UInt32}, i::Integer) = push!(vstate, i)
@inline function check_visited_and_visit!(vstate::Set{UInt32}, i::Integer)
    v = visited(vstate, i)
    !v && visit!(vstate, i)
    v
end

#### Visited-vertices types

"""
    AbstractVisited

The set of vertices a graph search has already reached, one per batch slot of a
[`SearchGraphContext`](@ref) (its `vstates`). Each search starts with
[`reuse!`](@ref)`(vstate, n)` and then asks [`check_visited_and_visit!`](@ref),
[`visited`](@ref) and [`visit!`](@ref). Three implementations:

- [`BitVisited`](@ref): one bit per vertex of the graph, zeroed at every `reuse!` (the
  original representation; `reuse!` costs `n/8` bytes of writes per search).
- [`ByteVisited`](@ref): one byte per vertex with the generation of the search that reached it
  (FAISS's `VisitedTable`); zeroed only every 255 searches, 8 times the bitset's memory.
- [`HashVisited`](@ref): an exact open-addressing table of the vertices reached by the current
  search, tagged with a generation number; `reuse!` only advances the generation, and the
  table grows with the largest search seen, not with `n`.
- [`LossyHashVisited`](@ref): a fixed-size table that may *forget* a vertex (it is then
  evaluated again) but never reports one that was not reached; it never grows, so it stays in
  cache. The search guards the result queue against the duplicates a forgotten vertex brings.

The plain `Vector{UInt64}` (a bitset) and `Set{UInt32}` keep working as `vstates` entries.
"""
abstract type AbstractVisited end

"""
    mayforget(vstate) -> Bool

Whether `vstate` can report an already reached vertex as not visited ([`LossyHashVisited`](@ref)).
The search then checks the result queue before pushing a vertex into it.
"""
@inline mayforget(::Any) = false

"""
    newvisited(proto::AbstractVisited) -> AbstractVisited

A fresh, empty visited set configured like `proto` (one per batch slot of a context).
"""
function newvisited end

"""
    BitVisited(nbits=2^21)

One bit per vertex: `reuse!(v, n)` resizes the bitset to `n` bits and zeroes it, so its cost per
search is `n/8` bytes of writes whatever the search visits (2.85 MB at 24M vertices).
"""
struct BitVisited <: AbstractVisited
    B::Vector{UInt64}
end

BitVisited(nbits::Integer=2^21) = BitVisited(Vector{UInt64}(undef, cld(nbits, 64)))
newvisited(v::BitVisited) = BitVisited(64length(v.B))
reuse!(v::BitVisited, n::Integer) = (reuse!(v.B, n); v)
@inline check_visited_and_visit!(v::BitVisited, i::Integer) = check_visited_and_visit!(v.B, convert(UInt64, i))
@inline visited(v::BitVisited, i::Integer) = Bool(visited(v.B, convert(UInt64, i)))
@inline visit!(v::BitVisited, i::Integer) = visit!(v.B, convert(UInt64, i))

"""
    ByteVisited(n=0)

One byte per vertex holding the generation of the search that reached it (FAISS's `VisitedTable`):
"visited" is `tags[i] == gen`. `reuse!` advances the generation and zeroes the table only when it
wraps, every 255 searches; it costs 8 times the memory of [`BitVisited`](@ref) (`n` bytes per batch
slot) but no per-search reset and no bit manipulation.
"""
mutable struct ByteVisited <: AbstractVisited
    tags::Vector{UInt8}
    gen::UInt8
end

ByteVisited(n::Integer=0) = ByteVisited(zeros(UInt8, n), 0x00)
newvisited(v::ByteVisited) = ByteVisited(length(v.tags))

function reuse!(v::ByteVisited, n::Integer)
    if n > length(v.tags)
        m = length(v.tags)
        resize!(v.tags, n)
        @inbounds fill!(view(v.tags, m+1:n), 0x00)
    end
    if v.gen == 0xff
        fill!(v.tags, 0x00)
        v.gen = 0x00
    end
    v.gen += 0x01
    v
end

@inline function check_visited_and_visit!(v::ByteVisited, i::Integer)::Bool
    @inbounds t = v.tags[i]
    t == v.gen && return true
    @inbounds v.tags[i] = v.gen
    false
end

@inline visited(v::ByteVisited, i::Integer)::Bool = @inbounds v.tags[i] == v.gen
@inline visit!(v::ByteVisited, i::Integer) = (@inbounds v.tags[i] = v.gen; nothing)

# a slot holds (generation << 32) | id; generation 0 is never current, so a zeroed slot is empty
@inline _vkey(gen::UInt64, i::Integer) = (gen << 32) | (convert(UInt64, i) & 0xffffffff)
@inline _vhash(i::Integer, bits::Int) = ((convert(UInt64, i) * 0x9E3779B97F4A7C15) >>> (64 - bits)) % Int

"""
    HashVisited(; capacity=2^12)

An exact visited set: an open-addressing table (linear probing) of `UInt64` slots, each holding
the vertex id and the generation of the search that inserted it. `reuse!` advances the
generation instead of clearing (the table is zeroed only when the 32-bit generation wraps), so
entries left by earlier searches read as empty. The table doubles when the current search fills
half of it and keeps its size for the following searches; its size follows the largest visit,
not the graph. Vertex ids must fit in 32 bits.
"""
mutable struct HashVisited <: AbstractVisited
    slots::Vector{UInt64}
    bits::Int
    gen::UInt64
    count::Int
end

function HashVisited(; capacity::Integer=2^12)
    bits = max(4, ceil(Int, log2(capacity)))
    HashVisited(UInt64[], bits, UInt64(0), 0)   # allocated on the first `reuse!`
end

newvisited(v::HashVisited) = HashVisited(; capacity=1 << v.bits)

function reuse!(v::HashVisited, ::Integer)
    if isempty(v.slots)
        v.slots = zeros(UInt64, 1 << v.bits)
        v.gen = 0
    end
    v.gen += 1
    if v.gen > 0xffffffff
        fill!(v.slots, 0)
        v.gen = 1
    end
    v.count = 0
    v
end

@inline function _probe(v::HashVisited, key::UInt64, i::Integer)
    mask = (1 << v.bits) - 1
    p = _vhash(i, v.bits)
    @inbounds while true
        s = v.slots[p + 1]
        (s == key || (s >>> 32) != v.gen) && return p + 1, s == key
        p = (p + 1) & mask
    end
end

function _grow!(v::HashVisited)
    old, gen = v.slots, v.gen
    v.bits += 1
    v.slots = zeros(UInt64, 1 << v.bits)
    @inbounds for s in old
        if (s >>> 32) == gen
            p, _ = _probe(v, s, s & 0xffffffff)
            v.slots[p] = s
        end
    end
    v
end

@inline function check_visited_and_visit!(v::HashVisited, i::Integer)::Bool
    key = _vkey(v.gen, i)
    p, found = _probe(v, key, i)
    found && return true
    @inbounds v.slots[p] = key
    v.count += 1
    2v.count > length(v.slots) && _grow!(v)
    false
end

@inline visited(v::HashVisited, i::Integer)::Bool = last(_probe(v, _vkey(v.gen, i), i))
@inline visit!(v::HashVisited, i::Integer) = (check_visited_and_visit!(v, i); nothing)

"""
    LossyHashVisited(; capacity=2^15)

A fixed-size visited set that may forget: `capacity` `UInt64` slots (256 KB by default, within a
core's L2) in buckets of 8 (one cache line), each slot holding the vertex id and the generation of
its search, as in [`HashVisited`](@ref). A vertex goes to a stale slot of its bucket, or, when all 8
belong to the current search, overwrites one of them. A lookup finds a vertex only if it is still
there, so the set never reports a vertex that was not reached; a forgotten one is evaluated again,
which costs a distance evaluation and is counted as one. `reuse!` advances the generation.
"""
mutable struct LossyHashVisited <: AbstractVisited
    slots::Vector{UInt64}
    bbits::Int   # log2 of the number of buckets
    gen::UInt64
end

function LossyHashVisited(; capacity::Integer=2^15)
    bbits = max(1, ceil(Int, log2(capacity)) - 3)
    LossyHashVisited(UInt64[], bbits, UInt64(0))
end

newvisited(v::LossyHashVisited) = LossyHashVisited(; capacity=8 << v.bbits)
@inline mayforget(::LossyHashVisited) = true

function reuse!(v::LossyHashVisited, ::Integer)
    if isempty(v.slots)
        v.slots = zeros(UInt64, 8 << v.bbits)
        v.gen = 0
    end
    v.gen += 1
    if v.gen > 0xffffffff
        fill!(v.slots, 0)
        v.gen = 1
    end
    v
end

@inline function check_visited_and_visit!(v::LossyHashVisited, i::Integer)::Bool
    key = _vkey(v.gen, i)
    h = _vhash(i, v.bbits + 3)
    base = (h & ~7) + 1
    free = 0
    @inbounds for j in 0:7
        s = v.slots[base + j]
        s == key && return true
        free == 0 && (s >>> 32) != v.gen && (free = base + j)
    end
    @inbounds v.slots[free == 0 ? base + (h & 7) : free] = key
    false
end

@inline function visited(v::LossyHashVisited, i::Integer)::Bool
    key = _vkey(v.gen, i)
    base = (_vhash(i, v.bbits + 3) & ~7) + 1
    @inbounds for j in 0:7
        v.slots[base + j] == key && return true
    end
    false
end

@inline visit!(v::LossyHashVisited, i::Integer) = (check_visited_and_visit!(v, i); nothing)

# The result queue of a search whose visited set may forget: a forgotten vertex is evaluated again
# and must not enter `res` twice. A linear scan over `res`, only for such sets (`mayforget` is
# resolved by dispatch, so the exact sets pay nothing).
@inline _pushres!(vstate, res, id, d) = mayforget(vstate) ? _pushunique!(res, id, d) : push_item!(res, id, d)

@inline function _pushunique!(res, id, d)
    ids = IdView(res)
    u = convert(UInt32, id)
    @inbounds for j in 1:length(res)
        ids[j] == u && return false
    end
    push_item!(res, id, d)
end
