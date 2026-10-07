# This file is a part of SimilaritySearch.jl

### The width layer of ScalarQuant.
###
### Everything that depends on the code width `B` (2, 4 or 8 bits per coordinate) lives here
### and is dispatched on `Val{B}`: how many codes a byte holds, how a vector is packed into
### codes and read back, and the three integer kernels every distance in this module is built
### from. The per-vector and global quantizers, and the databases over them, are
### width-agnostic callers of these functions.
###
### Codes are laid out low bits first: at 4 bits coordinate `2k+1` is the low nibble of byte
### `k+1` and `2k+2` the high one; at 2 bits coordinate `4k+p+1` sits in bits `2p:2p+1` of
### byte `k+1`. Every kernel reads *every* field of every byte, padding included, so a vector
### whose length is not a multiple of `codesperbyte(B)` is compared as if its unused fields
### were coordinates equal to `min`.

"Number of coordinates one byte holds at width `B`."
@inline codesperbyte(::Val{8}) = 1
@inline codesperbyte(::Val{4}) = 2
@inline codesperbyte(::Val{2}) = 4

"The largest code width `B` can hold, `2^B - 1`: 255, 15 or 3."
@inline levels(::Val{B}) where {B} = (1 << B) - 1

"Bytes needed to store `dim` coordinates at width `B`."
@inline nbytes(B::Val, dim::Integer) = cld(dim, codesperbyte(B))

"""
    packcodes!(::Val{B}, vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32) -> vout

Quantizes `v` into `vout` with the affine map `code = round(clamp((x - min) * c, 0, levels))`
and packs the codes `codesperbyte(B)` to a byte, low bits first. `vout` must hold
`nbytes(B, length(v))` bytes; a length that is not a multiple of `codesperbyte(B)` leaves the
unused fields of the last byte at zero.

`c` is the *quantization* multiplier (`levels / range`, see [`sqglobalscale`](@ref)), the
inverse of the step a [`SQMinC`](@ref) carries for dequantization.
"""
function packcodes!(::Val{8}, vout, v, min::Float32, c::Float32)
    for j in eachindex(v)
        x = round((v[j] - min) * c; digits=0)
        vout[j] = clamp(x, 0, 255)
    end

    vout
end

function packcodes!(::Val{4}, vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    m = length(v)
    k = 1
    j = 1
    @inbounds while j <= m
        a = round((v[j] - min) * c; digits=0)
        a = UInt8(clamp(a, 0, 15))
        b = zero(UInt8)
        if j+1 <= m
            b = let b = round((v[j+1] - min) * c; digits=0)
                UInt8(clamp(b, 0, 15))
            end
        end

        vout[k] = a | (b << 4)
        j += 2
        k += 1
    end

    vout
end

function packcodes!(::Val{2}, vout::AbstractVector{UInt8}, v::AbstractVector, min::Float32, c::Float32)
    m = length(v)
    k = 1
    j = 1
    @inbounds while j <= m
        x = zero(UInt8)
        for p in 0:3
            i = j + p
            i > m && break
            a = round((Float32(v[i]) - min) * c; digits=0)
            x |= UInt8(clamp(a, 0, 3)) << 2p
        end

        vout[k] = x
        j += 4
        k += 1
    end

    vout
end

"The `i`-th code stored in the packed bytes `V`, as a `UInt8` in `0:levels(B)`."
Base.@propagate_inbounds getcode(::Val{8}, V, i::Integer) = V[i]

Base.@propagate_inbounds function getcode(::Val{4}, V, i::Integer)
    isodd(i) ? V[(i + 1) >> 1] & 0x0f : V[i >> 1] >> 4
end

Base.@propagate_inbounds function getcode(::Val{2}, V, i::Integer)
    j = i - 1
    (V[(j >> 2) + 1] >> (2 * (j & 3))) & 0x03
end

### Integer kernels.
###
### A dequantized coordinate is `a*c + m`, and `c`/`m` belong to the *vector* in the
### per-vector family, so two codes cannot be compared directly the way the global family's
### shared-scale ones can. Expanding anyway:
###
###   (a*cA + mA) - (b*cB + mB) = cA*a - cB*b + k,        k = mA - mB
###   Σ(...)² = cA²Σa² + cB²Σb² - 2cAcB Σab + 2k(cA Σa - cB Σb) + n k²
###
### Everything there but `Σab` depends on a single vector, so it is computed once when the
### vector is quantized (`codesums`) and a distance costs one integer dot product
### (`dotcodes`) instead of dequantizing 2n coordinates into floats. The same expansion
### serves the dot product behind the cosines. When both scales are equal -- every
### comparison in the global family, every self-comparison in the per-vector one -- the
### whole thing collapses to `c² Σ(a-b)²`, which `sqdiffcodes` computes exactly.
###
### Overflow: at 8 bits each Int32 lane accumulates at most 255² = 65025 per step, so a single
### unwidened pass is safe up to ~33k steps, i.e. ~528k coordinates -- far beyond any vector
### this is used with. At 4 bits a code is in 0:15 (at most 225 per field), at 2 bits in 0:3
### (at most 9), so the narrower widths are safer still.
###
### Shape: each kernel runs a 32-lane pass, then at most one 16-lane pass, then the scalar
### remainder. A 16..31 code remainder left to scalar code costs more than vectorizing it (a
### 2-bit sketch of 64 hyperplanes is exactly 16 bytes: ~67ns scalar against ~14ns here). The
### 2-bit kernels once accumulated in Int16 lanes to cover 32 bytes per operation with four
### chains, and lost to one 32-lane Int32 accumulator (31.0ns against 18.2ns at 256 codes,
### scanning 32768 vectors): the chains keep more live vector state than the FMA latency they
### hide, and widening every block costs more than it saves. The 16-lane pass reduces its
### fields into one vector before the horizontal sum: four horizontal sums instead of one
### cost 2.5ns on a 16-byte sketch, a sixth of its whole distance.

"Σ of the codes and Σ of their squares, as `Float32`s: the per-vector halves of the expansion above."
@inline function codesums(::Val{8}, v::AbstractVector{UInt8})
    n = length(v); i = 1
    sa = zero(Vec{32,Int32}); saa = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        x = convert(Vec{32,Int32}, vload(Vec{32,UInt8}, v, i))
        sa += x
        saa = muladd(x, x, saa)
        i += 32
    end
    a = Int(sum(sa)); aa = Int(sum(saa))
    @inbounds if i + 15 <= n
        x = convert(Vec{16,Int32}, vload(Vec{16,UInt8}, v, i))
        a += Int(sum(x)); aa += Int(sum(x * x))
        i += 16
    end
    @inbounds while i <= n
        x = Int(v[i]); a += x; aa += x * x; i += 1
    end

    Float32(a), Float32(aa)
end

@inline function codesums(::Val{4}, v::AbstractVector{UInt8})
    n = length(v); i = 1; m = 0x0f
    sa = zero(Vec{32,Int32}); saa = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        b = vload(Vec{32,UInt8}, v, i)
        lo = convert(Vec{32,Int32}, b & m); hi = convert(Vec{32,Int32}, b >>> 4)
        sa += lo + hi
        saa = muladd(lo, lo, muladd(hi, hi, saa))
        i += 32
    end
    a = Int(sum(sa)); aa = Int(sum(saa))
    @inbounds if i + 15 <= n
        b = vload(Vec{16,UInt8}, v, i)
        lo = convert(Vec{16,Int32}, b & m); hi = convert(Vec{16,Int32}, b >>> 4)
        a += Int(sum(lo + hi)); aa += Int(sum(muladd(lo, lo, hi * hi)))
        i += 16
    end
    @inbounds while i <= n
        b = v[i]; lo = Int(b & m); hi = Int(b >>> 4)
        a += lo + hi; aa += lo * lo + hi * hi; i += 1
    end

    Float32(a), Float32(aa)
end

@inline function codesums(::Val{2}, v::AbstractVector{UInt8})
    n = length(v); i = 1; m = 0x03
    sa = zero(Vec{32,Int32}); saa = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        b = vload(Vec{32,UInt8}, v, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            x = convert(Vec{32,Int32}, (b >>> sh) & m)
            sa += x
            saa = muladd(x, x, saa)
        end
        i += 32
    end
    a = Int(sum(sa)); aa = Int(sum(saa))
    @inbounds if i + 15 <= n
        b = vload(Vec{16,UInt8}, v, i)
        sa16 = zero(Vec{16,Int32}); saa16 = zero(Vec{16,Int32})
        for sh in (0x00, 0x02, 0x04, 0x06)
            x = convert(Vec{16,Int32}, (b >>> sh) & m)
            sa16 += x
            saa16 = muladd(x, x, saa16)
        end
        a += Int(sum(sa16)); aa += Int(sum(saa16))
        i += 16
    end
    @inbounds while i <= n
        b = v[i]
        for sh in 0:2:6
            x = Int((b >>> sh) & m); a += x; aa += x * x
        end
        i += 1
    end

    Float32(a), Float32(aa)
end

"Σ aᵢbᵢ over the unpacked codes -- the only term of the expansion that depends on both vectors."
@inline function dotcodes(::Val{8}, x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        acc = muladd(convert(Vec{32,Int32}, vload(Vec{32,UInt8}, x, i)),
                     convert(Vec{32,Int32}, vload(Vec{32,UInt8}, y, i)), acc)
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        s += Int(sum(convert(Vec{16,Int32}, vload(Vec{16,UInt8}, x, i)) *
                     convert(Vec{16,Int32}, vload(Vec{16,UInt8}, y, i))))
        i += 16
    end
    @inbounds while i <= n; s += Int(x[i]) * Int(y[i]); i += 1; end
    s
end

@inline function dotcodes(::Val{4}, x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x0f; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        acc = muladd(convert(Vec{32,Int32}, bx & m), convert(Vec{32,Int32}, by & m), acc)
        acc = muladd(convert(Vec{32,Int32}, bx >>> 4), convert(Vec{32,Int32}, by >>> 4), acc)
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        s += Int(sum(muladd(convert(Vec{16,Int32}, bx & m), convert(Vec{16,Int32}, by & m),
                            convert(Vec{16,Int32}, bx >>> 4) * convert(Vec{16,Int32}, by >>> 4))))
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        s += Int(bx & m) * Int(by & m) + Int(bx >>> 4) * Int(by >>> 4)
        i += 1
    end
    s
end

@inline function dotcodes(::Val{2}, x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x03; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            acc = muladd(convert(Vec{32,Int32}, (bx >>> sh) & m),
                         convert(Vec{32,Int32}, (by >>> sh) & m), acc)
        end
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        acc16 = zero(Vec{16,Int32})
        for sh in (0x00, 0x02, 0x04, 0x06)
            acc16 = muladd(convert(Vec{16,Int32}, (bx >>> sh) & m),
                           convert(Vec{16,Int32}, (by >>> sh) & m), acc16)
        end
        s += Int(sum(acc16))
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        for sh in 0:2:6
            s += Int((bx >>> sh) & m) * Int((by >>> sh) & m)
        end
        i += 1
    end
    s
end

"""
    dotquery(::Val{8}, x, d::AbstractVector{Int16})
    dotquery(::Val{4}, x, dlo::AbstractVector{Int8}, dhi::AbstractVector{Int8})
    dotquery(::Val{2}, x, d0, d1, d2, d3)

Σ aᵢdᵢ between the unpacked codes `x` and the signed integer image of a query, one plane per
field of a byte (see `SQQuery`): plane `p` holds the coordinates field `p` unpacks into, so
every plane is read at the byte's own index and nothing is interleaved. Lanes accumulate in
`Int32` -- at 8 bits a product is at most 255·16384 and a lane sees `n/32` of them, which
holds to 16K dimensions -- and the horizontal sum is taken in `Int64`.
"""
@inline function dotquery(::Val{8}, x::AbstractVector{UInt8}, d::AbstractVector{Int16})::Int64
    n = length(x); i = 1; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        acc = muladd(convert(Vec{32,Int32}, vload(Vec{32,UInt8}, x, i)),
                     convert(Vec{32,Int32}, vload(Vec{32,Int16}, d, i)), acc)
        i += 32
    end
    s = sum(convert(Vec{32,Int64}, acc))
    @inbounds if i + 15 <= n
        s += sum(convert(Vec{16,Int64}, convert(Vec{16,Int32}, vload(Vec{16,UInt8}, x, i)) *
                                         convert(Vec{16,Int32}, vload(Vec{16,Int16}, d, i))))
        i += 16
    end
    @inbounds while i <= n; s += Int64(x[i]) * Int64(d[i]); i += 1; end
    s
end

@inline function dotquery(::Val{4}, x::AbstractVector{UInt8}, dlo::AbstractVector{Int8}, dhi::AbstractVector{Int8})::Int64
    n = length(x); i = 1; m = 0x0f; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        b = vload(Vec{32,UInt8}, x, i)
        acc = muladd(convert(Vec{32,Int32}, b & m), convert(Vec{32,Int32}, vload(Vec{32,Int8}, dlo, i)), acc)
        acc = muladd(convert(Vec{32,Int32}, b >>> 4), convert(Vec{32,Int32}, vload(Vec{32,Int8}, dhi, i)), acc)
        i += 32
    end
    s = sum(convert(Vec{32,Int64}, acc))
    @inbounds if i + 15 <= n
        b = vload(Vec{16,UInt8}, x, i)
        s += sum(convert(Vec{16,Int64}, muladd(convert(Vec{16,Int32}, b & m), convert(Vec{16,Int32}, vload(Vec{16,Int8}, dlo, i)),
                                                 convert(Vec{16,Int32}, b >>> 4) * convert(Vec{16,Int32}, vload(Vec{16,Int8}, dhi, i)))))
        i += 16
    end
    @inbounds while i <= n
        b = x[i]
        s += Int64(b & m) * Int64(dlo[i]) + Int64(b >>> 4) * Int64(dhi[i])
        i += 1
    end
    s
end

@inline function dotquery(::Val{2}, x::AbstractVector{UInt8}, d0::AbstractVector{Int8}, d1::AbstractVector{Int8},
                          d2::AbstractVector{Int8}, d3::AbstractVector{Int8})::Int64
    n = length(x); i = 1; m = 0x03; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        b = vload(Vec{32,UInt8}, x, i)
        acc = muladd(convert(Vec{32,Int32}, b & m), convert(Vec{32,Int32}, vload(Vec{32,Int8}, d0, i)), acc)
        acc = muladd(convert(Vec{32,Int32}, (b >>> 2) & m), convert(Vec{32,Int32}, vload(Vec{32,Int8}, d1, i)), acc)
        acc = muladd(convert(Vec{32,Int32}, (b >>> 4) & m), convert(Vec{32,Int32}, vload(Vec{32,Int8}, d2, i)), acc)
        acc = muladd(convert(Vec{32,Int32}, b >>> 6), convert(Vec{32,Int32}, vload(Vec{32,Int8}, d3, i)), acc)
        i += 32
    end
    s = sum(convert(Vec{32,Int64}, acc))
    @inbounds if i + 15 <= n
        b = vload(Vec{16,UInt8}, x, i); acc16 = zero(Vec{16,Int32})
        acc16 = muladd(convert(Vec{16,Int32}, b & m), convert(Vec{16,Int32}, vload(Vec{16,Int8}, d0, i)), acc16)
        acc16 = muladd(convert(Vec{16,Int32}, (b >>> 2) & m), convert(Vec{16,Int32}, vload(Vec{16,Int8}, d1, i)), acc16)
        acc16 = muladd(convert(Vec{16,Int32}, (b >>> 4) & m), convert(Vec{16,Int32}, vload(Vec{16,Int8}, d2, i)), acc16)
        acc16 = muladd(convert(Vec{16,Int32}, b >>> 6), convert(Vec{16,Int32}, vload(Vec{16,Int8}, d3, i)), acc16)
        s += sum(convert(Vec{16,Int64}, acc16))
        i += 16
    end
    @inbounds while i <= n
        b = x[i]
        s += Int64(b & m) * Int64(d0[i]) + Int64((b >>> 2) & m) * Int64(d1[i]) +
             Int64((b >>> 4) & m) * Int64(d2[i]) + Int64(b >>> 6) * Int64(d3[i])
        i += 1
    end
    s
end

"""
Σ (aᵢ-bᵢ)² over the raw codes: the whole answer whenever both vectors share a scale, and
what every `SqL2` over raw codes evaluates.

At 8 bits two shapes coexisted, and neither dominates: a single 32-lane accumulator, and a
4x-unrolled loop with four of them. Measured scanning 32768 vectors on one thread, the single
accumulator takes 7.8/12.1/15.7/25.1 ns at 64/128/256/384 codes against 16.8/17.3/21.1/26.2
for the unrolled loop, which then wins at 512/768/1024/1536 codes with 31.7/50.7/77.2/123.5
against 33.3/52.9/85.4/129.2. Two accumulators lost to both everywhere (9.6 to 172.0 ns), so
the crossover is dispatched on length rather than tuned away.
"""
@inline function sqdiffcodes(::Val{8}, x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    length(x) >= 512 ? _sqdiff8_unrolled(x, y) : _sqdiff8(x, y)
end

@inline function _sqdiff8(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        d = convert(Vec{32,Int32}, vload(Vec{32,UInt8}, x, i)) -
            convert(Vec{32,Int32}, vload(Vec{32,UInt8}, y, i))
        acc = muladd(d, d, acc)
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        d = convert(Vec{16,Int32}, vload(Vec{16,UInt8}, x, i)) -
            convert(Vec{16,Int32}, vload(Vec{16,UInt8}, y, i))
        s += Int(sum(d * d))
        i += 16
    end
    @inbounds while i <= n; d = Int(x[i]) - Int(y[i]); s += d * d; i += 1; end
    s
end

@inline function _sqdiff8_unrolled(x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    N = 32
    UNROLL = 4
    CHUNK = N * UNROLL # 128 elements per iteration

    # Int32 rather than UInt32 to safely hold negative differences
    acc1 = zero(Vec{N, Int32})
    acc2 = zero(Vec{N, Int32})
    acc3 = zero(Vec{N, Int32})
    acc4 = zero(Vec{N, Int32})

    n = length(x)
    limit_unrolled = n - CHUNK + 1
    i = 1

    @inbounds while i <= limit_unrolled
        diff1 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i)) -
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i))
        acc1 = muladd(diff1, diff1, acc1)

        diff2 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i + N)) -
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i + N))
        acc2 = muladd(diff2, diff2, acc2)

        diff3 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i + 2N)) -
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i + 2N))
        acc3 = muladd(diff3, diff3, acc3)

        diff4 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i + 3N)) -
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i + 3N))
        acc4 = muladd(diff4, diff4, acc4)

        i += CHUNK
    end

    # Reduced to a scalar here, before the cleanup loops, and not after them: keeping the
    # vector accumulator live across a loop whose trip count the compiler cannot prove is zero
    # costs 2.6x on this kernel (69.7ns against 26.4ns at 512 codes, measured with the cleanup
    # never actually running). The cleanup phases below accumulate into `res` instead.
    res = Int(sum(acc1)) + Int(sum(acc2)) + Int(sum(acc3)) + Int(sum(acc4))

    @inbounds while i + N - 1 <= n
        diff = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i)) -
               convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i))
        res += Int(sum(diff * diff))
        i += N
    end

    @inbounds while i + 15 <= n
        vx = vload(Vec{16, UInt8}, x, i)
        vy = vload(Vec{16, UInt8}, y, i)
        d = convert(Vec{16, Int32}, vx) - convert(Vec{16, Int32}, vy)
        res += Int(sum(d * d))
        i += 16
    end

    @inbounds while i <= n
        scalar_diff = Int(x[i]) - Int(y[i])
        res += scalar_diff * scalar_diff
        i += 1
    end

    res
end

@inline function sqdiffcodes(::Val{4}, x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x0f; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        dlo = convert(Vec{32,Int32}, bx & m) - convert(Vec{32,Int32}, by & m)
        dhi = convert(Vec{32,Int32}, bx >>> 4) - convert(Vec{32,Int32}, by >>> 4)
        acc = muladd(dlo, dlo, muladd(dhi, dhi, acc))
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        dlo = convert(Vec{16,Int32}, bx & m) - convert(Vec{16,Int32}, by & m)
        dhi = convert(Vec{16,Int32}, bx >>> 4) - convert(Vec{16,Int32}, by >>> 4)
        s += Int(sum(muladd(dlo, dlo, dhi * dhi)))
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        dlo = Int(bx & m) - Int(by & m); dhi = Int(bx >>> 4) - Int(by >>> 4)
        s += dlo * dlo + dhi * dhi
        i += 1
    end
    s
end

@inline function sqdiffcodes(::Val{2}, x::AbstractVector{UInt8}, y::AbstractVector{UInt8})
    n = length(x); i = 1; m = 0x03; acc = zero(Vec{32,Int32})
    @inbounds while i + 31 <= n
        bx = vload(Vec{32,UInt8}, x, i); by = vload(Vec{32,UInt8}, y, i)
        for sh in (0x00, 0x02, 0x04, 0x06)
            d = convert(Vec{32,Int32}, (bx >>> sh) & m) - convert(Vec{32,Int32}, (by >>> sh) & m)
            acc = muladd(d, d, acc)
        end
        i += 32
    end
    s = Int(sum(acc))
    @inbounds if i + 15 <= n
        bx = vload(Vec{16,UInt8}, x, i); by = vload(Vec{16,UInt8}, y, i)
        acc16 = zero(Vec{16,Int32})
        for sh in (0x00, 0x02, 0x04, 0x06)
            d = convert(Vec{16,Int32}, (bx >>> sh) & m) - convert(Vec{16,Int32}, (by >>> sh) & m)
            acc16 = muladd(d, d, acc16)
        end
        s += Int(sum(acc16))
        i += 16
    end
    @inbounds while i <= n
        bx, by = x[i], y[i]
        for sh in 0:2:6
            d = Int((bx >>> sh) & m) - Int((by >>> sh) & m); s += d * d
        end
        i += 1
    end
    s
end
