# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::AbstractSearchIndex, meta; store_db=true)

Saves the index into the given file using JLD2. It is recommended instead plain JLD2 since it simplifies
several typical operations and also allow that indexes perform pre-save and post-load operations
when indexes are saved and restored. 

# Arguments:
- `filename`: output file of the index.
- `index`: the index to be saved.
- `meta`: additional metadata to be saved with the index (it can be any object supported by JLD2).

# Keyword arguments:
- `store_db=true` controls if the daset should stored, when `store_db=false` you need to give the dataset on loading.

"""
function saveindex(file::JLD2.JLDFile, index::AbstractSearchIndex, meta, options::Dict; parent="/")
    file[joinpath(parent, "options")] = options 
    file[joinpath(parent, "meta")] = meta
    file[joinpath(parent, "index")] = index
end

function saveindex(filename::AbstractString, index::AbstractSearchIndex; meta=nothing, store_db=true, parent="/")
    jldopen(filename, "w") do f
        saveindex(f, index; meta, store_db, parent)
    end
end

function saveindex(file::JLD2.JLDFile, index::AbstractSearchIndex; meta=nothing, store_db=true, parent="/")
    uses_stride_matrix_database = database(index) isa StrideMatrixDatabase

    if store_db
        stores_database = true
    else
        stores_database = false
        index = copy(index; db=MatrixDatabase(zeros(Float32, 2, 2)))
    end

    options = Dict(
        "uses_stride_matrix_database" => uses_stride_matrix_database,
        "stores_database" => stores_database
    )

    saveindex(file, index, meta, options; parent)
end

"""
    restoreindex(index::AbstractSearchIndex, meta, options::Dict)

Called on loading to restore any value on the index that depends on the new instance (i.e., datasets, pointers, etc.)
"""
restoreindex(index::AbstractSearchIndex, meta, options::Dict; kwargs...) = index

"""
    loadindex(filename::AbstractString, db::Union{Nothing,AbstractDatabase}=nothing; kwargs...)

Loads an index from `filename`, optionally, if `db` is given it will replace the dataset on the loaded index, but it is
mandatory if the index was saved with `store_db=false`
"""
function loadindex(filename::AbstractString, db::Union{Nothing,AbstractDatabase}=nothing; kwargs...)
    jldopen(filename) do f
        loadindex(f, db; kwargs...)
    end
end

function loadindex(f::JLD2.JLDFile, db::Union{Nothing,AbstractDatabase}=nothing; parent="/", kwargs...)
    options = f[joinpath(parent, "options")]
    meta = f[joinpath(parent, "meta")]
    index = f[joinpath(parent, "index")]
    index = restoreindex(index, meta, options; kwargs...)

    if db === nothing
        if !options["stores_database"]
            @warn "the database was not stored and was not passed to loadindex"
        end
    else
        index = copy(index; db)
    end

    if options["uses_stride_matrix_database"]
        index = copy(index; db=StrideMatrixDatabase(database(index)))
    end

    index, meta
end
