export sq_global_u8

function quant_global_u8!(vout, v, min::Float32, c::Float32)
    # c = 255f0 / (max - min)
    for j in eachindex(v)
        x = round((v[j] - min) * c; digits=0)
        vout[j] = clamp(x, 0, 255)
    end

    vout
end

function sq_global_u8(X::AbstractMatrix; qminmax=[0.01, 0.99])
    m, n = size(X)
    Q = Matrix{UInt8}(undef, m, n)
    min, max = let  V = vec(X),
                    n = length(V),
                    S = rand(V, ceil(Int, n^0.5))
        quantile(S, qminmax)
    end
    c = Float32(255 / (max - min + 1e-6))
    quant_global_u8!(vout, v, min, c)

    @batch per=thread minbatch=4 for i in 1:n
        quant_global_u8!(view(Q, :, i), view(X, :, i), min, c)
    end

    Q
end
