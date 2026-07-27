export sq_global_u8

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

!!! note
    The default `minmax=nothing` path estimates the range via `quantile` on a sample of
    `X`; as of this writing `ScalarQuant` does not import `quantile` (from `Statistics`
    or `StatsBase`), so calling `sq_global_u8` without an explicit `minmax` currently
    raises an `UndefVarError`. Passing `minmax` explicitly, as in the example above,
    avoids this code path.
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
