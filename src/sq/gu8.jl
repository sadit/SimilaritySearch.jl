export sq_global_u8, SQgu8NormCosine, SQgu8SqL2, SQgu4SqL2 

function quant_global_u8!(vout, v, min::Float32, c::Float32)
    # c = 255f0 / (max - min)
    for j in eachindex(v)
        x = round((v[j] - min) * c; digits=0)
        vout[j] = clamp(x, 0, 255)
    end

    vout
end

"""
    sq_global_u8(X::AbstractMatrix; minmax=nothing, quant=[0.025, 0.975], samplesize=0)

Scalar-quantizes every entry of `X` to 8 bits (`UInt8`) using a single, global pair of
dequantization parameters shared by all columns, unlike [`SQu8`](@ref) which computes an
independent `min`/scale per column. This is useful, e.g., when the columns of `X` are
known to share a comparable value range and a single global range provides enough
precision while being cheaper to compute and store.

The global `[min, max]` range is estimated by sampling entries of `X` and taking
quantiles of the sample (to be robust to outliers), unless it is provided explicitly via
`minmax`. Every entry `x` is then mapped as `round(clamp((x - min) * c, 0, 255))` with
`c = 255 / (max - min + 1e-6)`.

# Arguments
- `X`: the matrix to quantize; each entry is quantized independently but using shared
  `min`/`max` values
- `minmax`: an optional `(min, max)` tuple giving the value range to use; when `nothing`
  (the default) the range is estimated from a random sample of the entries of `X` using
  `quant`
- `quant`: the lower and upper quantiles (of the sampled entries of `X`) used to estimate
  `min` and `max` when `minmax` is not given
- `samplesize`: the number of entries sampled (with replacement) from `X` to estimate the
  quantiles; when `0` (the default) it is set to `ceil(Int, length(X)^0.5)`

# Examples

```julia
julia> using SimilaritySearch

julia> X = rand(Float32, 8, 1000);

julia> Q = ScalarQuant.sq_global_u8(X; minmax=(0f0, 1f0));  # explicit range

julia> size(Q), eltype(Q)  # (8, 1000), UInt8
```
"""
function sq_global_u8(X::AbstractMatrix;
        minmax=nothing,
        quant=[0.025, 0.975],
        samplesize=0
    )
    m, n = size(X)
    Q = Matrix{UInt8}(undef, m, n)
    
    min, max = if minmax === nothing
        let  V = vec(X),
             n = length(V),
             samplesize = samplesize === 0 ? ceil(Int, n^0.5) : samplesize
             S = rand(V, samplesize)
            quantile(S, quant)
        end
    else
        minmax
    end

    c = Float32(255 / (max - min + 1e-6))
    min = Float32(min)

    minbatch = getminbatch(n)
    @batch per=thread minbatch=minbatch for i in 1:n
        quant_global_u8!(view(Q, :, i), view(X, :, i), min, c)
    end

    Q
end


### the following code was made with the help of Gemini IA
using SIMD
struct SQgu8NormCosine <: Dist.SemiMetric
end

function Dist.evaluate(::SQgu8NormCosine, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Vectors must be the same length"))
    
    N = 32
    UNROLL = 4
    CHUNK = N * UNROLL # Processes 128 elements per loop iteration
    
    # Initialize 4 independent accumulators to break dependency chains
    acc1 = zero(Vec{N, UInt32})
    acc2 = zero(Vec{N, UInt32})
    acc3 = zero(Vec{N, UInt32})
    acc4 = zero(Vec{N, UInt32})
    
    n = length(x)
    limit_unrolled = n - CHUNK + 1
    i = 1
    
    # --- PHASE 1: The Unrolled Loop (Chunks of 128) ---
    @inbounds while i <= limit_unrolled
        # 1. Load 4 distinct chunks from each array
        vx1 = vload(Vec{N, UInt8}, x, i)
        vy1 = vload(Vec{N, UInt8}, y, i)
        
        vx2 = vload(Vec{N, UInt8}, x, i + N)
        vy2 = vload(Vec{N, UInt8}, y, i + N)
        
        vx3 = vload(Vec{N, UInt8}, x, i + 2N)
        vy3 = vload(Vec{N, UInt8}, y, i + 2N)
        
        vx4 = vload(Vec{N, UInt8}, x, i + 3N)
        vy4 = vload(Vec{N, UInt8}, y, i + 3N)
        
        # 2. Widen and accumulate into independent registers
        acc1 = muladd(convert(Vec{N, UInt32}, vx1), convert(Vec{N, UInt32}, vy1), acc1)
        acc2 = muladd(convert(Vec{N, UInt32}, vx2), convert(Vec{N, UInt32}, vy2), acc2)
        acc3 = muladd(convert(Vec{N, UInt32}, vx3), convert(Vec{N, UInt32}, vy3), acc3)
        acc4 = muladd(convert(Vec{N, UInt32}, vx4), convert(Vec{N, UInt32}, vy4), acc4)
        
        i += CHUNK
    end
    
    # Combine the 4 parallel accumulators into one
    acc_total = acc1 + acc2 + acc3 + acc4
    
    # --- PHASE 2: Single SIMD Loop Cleanup (Chunks of 32) ---
    # Catches the remaining vectors if the array length isn't a perfect multiple of 128
    limit_single = n - N + 1
    @inbounds while i <= limit_single
        vx = vload(Vec{N, UInt8}, x, i)
        vy = vload(Vec{N, UInt8}, y, i)
        
        acc_total = muladd(convert(Vec{N, UInt32}, vx), convert(Vec{N, UInt32}, vy), acc_total)
        i += N
    end
    
    # Horizontal reduction
    res = sum(acc_total)
    
    # --- PHASE 3: Scalar Tail Cleanup ---
    # Catches the absolute tail if there are fewer than 32 elements left
    @inbounds while i <= n
        res += UInt32(x[i]) * UInt32(y[i])
        i += 1
    end
    
    convert(Float32, -res)
end



struct SQgu8SqL2 <: Dist.SemiMetric
end

function Dist.evaluate(::SQgu8SqL2, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Vectors must be the same length"))
    
    N = 32
    UNROLL = 4
    CHUNK = N * UNROLL # 128 elements per iteration
    
    # We use Int32 here instead of UInt32 to safely handle negative differences
    acc1 = zero(Vec{N, Int32})
    acc2 = zero(Vec{N, Int32})
    acc3 = zero(Vec{N, Int32})
    acc4 = zero(Vec{N, Int32})
    
    n = length(x)
    limit_unrolled = n - CHUNK + 1
    i = 1
    
    # --- PHASE 1: The Unrolled Loop ---
    @inbounds while i <= limit_unrolled
        # Chunk 1: Load, widen to Int32, subtract, and square-accumulate
        diff1 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i)) - 
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i))
        acc1 = muladd(diff1, diff1, acc1)
        
        # Chunk 2
        diff2 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i + N)) - 
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i + N))
        acc2 = muladd(diff2, diff2, acc2)
        
        # Chunk 3
        diff3 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i + 2N)) - 
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i + 2N))
        acc3 = muladd(diff3, diff3, acc3)
        
        # Chunk 4
        diff4 = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i + 3N)) - 
                convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i + 3N))
        acc4 = muladd(diff4, diff4, acc4)
        
        i += CHUNK
    end
    
    # Combine the parallel accumulators
    acc_total = acc1 + acc2 + acc3 + acc4
    
    # --- PHASE 2: Single SIMD Loop Cleanup ---
    limit_single = n - N + 1
    @inbounds while i <= limit_single
        diff = convert(Vec{N, Int32}, vload(Vec{N, UInt8}, x, i)) - 
               convert(Vec{N, Int32}, vload(Vec{N, UInt8}, y, i))
        acc_total = muladd(diff, diff, acc_total)
        i += N
    end
    
    # Horizontal reduction
    res = sum(acc_total)
    
    # --- PHASE 3: Scalar Tail Cleanup ---
    @inbounds while i <= n
        # Widen to Int32 before subtracting!
        scalar_diff = Int32(x[i]) - Int32(y[i])
        res += scalar_diff * scalar_diff
        i += 1
    end
    
    Float32(res)
end


struct SQgu4SqL2 <: Dist.SemiMetric
end

function Dist.evaluate(::SQgu4SqL2, x::AbstractArray{UInt8}, y::AbstractArray{UInt8})
    @boundscheck length(x) == length(y) || throw(DimensionMismatch("Byte arrays must be the same length"))
    
    # We use N=16 here instead of 32. 
    # Why? Because every 1 byte splits into TWO 32-bit accumulators. 
    # N=16 prevents "register spilling" on AVX2 architectures, keeping everything in the CPU's fast registers.
    N = 16
    UNROLL = 4
    CHUNK = N * UNROLL # 64 bytes (128 dimensions) per iteration
    
    # We need 8 accumulators total: 4 for the lower nibbles, 4 for the upper nibbles
    acc1_low  = zero(Vec{N, Int32})
    acc1_high = zero(Vec{N, Int32})
    acc2_low  = zero(Vec{N, Int32})
    acc2_high = zero(Vec{N, Int32})
    acc3_low  = zero(Vec{N, Int32})
    acc3_high = zero(Vec{N, Int32})
    acc4_low  = zero(Vec{N, Int32})
    acc4_high = zero(Vec{N, Int32})
    
    n = length(x)
    limit_unrolled = n - CHUNK + 1
    i = 1
    
    # Mask to isolate the bottom 4 bits (00001111 in binary)
    mask = 0x0f 
    
    # --- PHASE 1: The Unrolled Loop ---
    @inbounds while i <= limit_unrolled
        # Chunk 1
        vx1 = vload(Vec{N, UInt8}, x, i)
        vy1 = vload(Vec{N, UInt8}, y, i)
        
        # Unpack lower nibbles (bits 0-3) and widen
        diff1_low = convert(Vec{N, Int32}, vx1 & mask) - convert(Vec{N, Int32}, vy1 & mask)
        acc1_low  = muladd(diff1_low, diff1_low, acc1_low)
        
        # Unpack upper nibbles (bits 4-7) by logical right-shift and widen
        diff1_high = convert(Vec{N, Int32}, vx1 >>> 4) - convert(Vec{N, Int32}, vy1 >>> 4)
        acc1_high  = muladd(diff1_high, diff1_high, acc1_high)

        # Chunk 2
        vx2 = vload(Vec{N, UInt8}, x, i + N)
        vy2 = vload(Vec{N, UInt8}, y, i + N)
        diff2_low  = convert(Vec{N, Int32}, vx2 & mask) - convert(Vec{N, Int32}, vy2 & mask)
        acc2_low   = muladd(diff2_low, diff2_low, acc2_low)
        diff2_high = convert(Vec{N, Int32}, vx2 >>> 4) - convert(Vec{N, Int32}, vy2 >>> 4)
        acc2_high  = muladd(diff2_high, diff2_high, acc2_high)

        # Chunk 3
        vx3 = vload(Vec{N, UInt8}, x, i + 2N)
        vy3 = vload(Vec{N, UInt8}, y, i + 2N)
        diff3_low  = convert(Vec{N, Int32}, vx3 & mask) - convert(Vec{N, Int32}, vy3 & mask)
        acc3_low   = muladd(diff3_low, diff3_low, acc3_low)
        diff3_high = convert(Vec{N, Int32}, vx3 >>> 4) - convert(Vec{N, Int32}, vy3 >>> 4)
        acc3_high  = muladd(diff3_high, diff3_high, acc3_high)

        # Chunk 4
        vx4 = vload(Vec{N, UInt8}, x, i + 3N)
        vy4 = vload(Vec{N, UInt8}, y, i + 3N)
        diff4_low  = convert(Vec{N, Int32}, vx4 & mask) - convert(Vec{N, Int32}, vy4 & mask)
        acc4_low   = muladd(diff4_low, diff4_low, acc4_low)
        diff4_high = convert(Vec{N, Int32}, vx4 >>> 4) - convert(Vec{N, Int32}, vy4 >>> 4)
        acc4_high  = muladd(diff4_high, diff4_high, acc4_high)

        i += CHUNK
    end
    
    # Combine all 8 accumulators into a single running total
    acc_total = acc1_low + acc1_high + 
                acc2_low + acc2_high + 
                acc3_low + acc3_high + 
                acc4_low + acc4_high
    
    # --- PHASE 2: Single SIMD Loop Cleanup ---
    limit_single = n - N + 1
    @inbounds while i <= limit_single
        vx = vload(Vec{N, UInt8}, x, i)
        vy = vload(Vec{N, UInt8}, y, i)
        
        diff_low  = convert(Vec{N, Int32}, vx & mask) - convert(Vec{N, Int32}, vy & mask)
        acc_total = muladd(diff_low, diff_low, acc_total)
        
        diff_high = convert(Vec{N, Int32}, vx >>> 4) - convert(Vec{N, Int32}, vy >>> 4)
        acc_total = muladd(diff_high, diff_high, acc_total)
        
        i += N
    end
    
    res = sum(acc_total)
    
    # --- PHASE 3: Scalar Tail Cleanup ---
    @inbounds while i <= n
        # Unpack the tail byte manually
        x_val, y_val = x[i], y[i]
        
        x_low, y_low   = Int32(x_val & mask), Int32(y_val & mask)
        x_high, y_high = Int32(x_val >>> 4), Int32(y_val >>> 4)
        
        diff_low  = x_low - y_low
        diff_high = x_high - y_high
        
        res += (diff_low * diff_low) + (diff_high * diff_high)
        i += 1
    end
    
    convert(Float32, res)
end