"""
    split_pathkey(spec; require_key=true) -> (path, key)

Splits `spec` on its last ':' into a file path and an internal HDF5/JLD2 key. Splitting on the
*last* ':' (rather than the first) keeps this robust to paths that might contain one themselves.
"""
function split_pathkey(spec::AbstractString; require_key::Bool=true)
    idx = findlast(==(':'), spec)
    if idx === nothing
        require_key && error("expected 'path:key' form, got '$spec' (no ':' found)")
        return String(spec), nothing
    end
    path = String(spec[1:idx-1])
    key = String(spec[idx+1:end])
    isempty(key) && error("empty key after ':' in '$spec'")
    path, key
end

_fileext(path::AbstractString) = lowercase(splitext(path)[2])

function read_key(path::AbstractString, key::AbstractString)
    ext = _fileext(path)
    if ext == ".h5"
        HDF5.h5open(path, "r") do f
            read(f[key])
        end
    elseif ext == ".jld2"
        JLD2.jldopen(path, "r") do f
            f[key]
        end
    else
        error("unsupported extension '$ext' for '$path' (expected .h5 or .jld2)")
    end
end

function has_key(path::AbstractString, key::AbstractString)
    ext = _fileext(path)
    if ext == ".h5"
        HDF5.h5open(path, "r") do f
            haskey(f, key)
        end
    elseif ext == ".jld2"
        JLD2.jldopen(path, "r") do f
            haskey(f, key)
        end
    else
        error("unsupported extension '$ext' for '$path' (expected .h5 or .jld2)")
    end
end

"""
    read_matrix(path, key) -> Matrix{Float32}

Reads a numeric matrix stored under `key` in an `.h5`/`.jld2` file. Columns are treated as the
individual object vectors (i.e. shape `(dim, nobjects)`), matching `MatrixDatabase`'s convention.
"""
read_matrix(path::AbstractString, key::AbstractString) = Matrix{Float32}(read_key(path, key))

"""
    write_results(path, ids, dists=nothing)

Writes a k-NN results file with fixed keys `"ids"` (`(k, nqueries)` `Int32` matrix) and,
if given, `"dists"` (`(k, nqueries)` `Float32` matrix). `path` must end in `.h5` or `.jld2`.
"""
function write_results(path::AbstractString, ids::AbstractMatrix{<:Integer},
                        dists::Union{Nothing,AbstractMatrix{<:AbstractFloat}}=nothing)
    ext = _fileext(path)
    I = Matrix{Int32}(ids)
    if ext == ".jld2"
        if dists === nothing
            JLD2.jldsave(path; ids=I)
        else
            JLD2.jldsave(path; ids=I, dists=Matrix{Float32}(dists))
        end
    elseif ext == ".h5"
        HDF5.h5open(path, "w") do f
            f["ids"] = I
            dists !== nothing && (f["dists"] = Matrix{Float32}(dists))
        end
    else
        error("results path must end in .h5 or .jld2, got '$path'")
    end
end

"""
    load_results_spec(spec) -> (ids::Matrix{Int32}, dists::Union{Nothing,Matrix{Float32}})

Loads a k-NN results file referenced by `spec`. Without a `:key` suffix, `spec` is treated as a
file written by `simsearch search` (fixed keys `"ids"`/`"dists"`). With a `:key` suffix, `key` is
treated as pointing directly at an externally-produced ids matrix; a sibling key `"\$(key)_dists"`
supplies distances if present, otherwise distance-distribution stats are skipped for that input.
"""
function load_results_spec(spec::AbstractString)
    path, key = split_pathkey(spec; require_key=false)
    idskey, distskey = key === nothing ? ("ids", "dists") : (key, key * "_dists")
    ids = Matrix{Int32}(read_key(path, idskey))
    dists = has_key(path, distskey) ? Matrix{Float32}(read_key(path, distskey)) : nothing
    ids, dists
end
