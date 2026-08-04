"""
    save_index(path, index)

Saves `index` to `path` (must end in `.jld2`). The dataset (`index.db`) is never embedded in the
saved file -- it is swapped for a tiny placeholder first -- so `--dataset` must always be given
to `load_index` to reattach the real data.
"""
function save_index(path::AbstractString, index)
    endswith(lowercase(path), ".jld2") || error("--save must end in .jld2, got '$path'")
    placeholder = MatrixDatabase(zeros(Float32, 0, 0))
    out = @reset index.db = placeholder
    JLD2.jldsave(path; index=out)
end

"""
    load_index(path, dataset_spec)

Loads an index saved by `save_index` from `path` and reattaches the real dataset read from
`dataset_spec` (a `"path:key"` spec), returning a fully usable index.
"""
function load_index(path::AbstractString, dataset_spec::AbstractString)
    idx = JLD2.load_object(path)
    p, k = split_pathkey(dataset_spec)
    @reset idx.db = MatrixDatabase(read_matrix(p, k))
end
