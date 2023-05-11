# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::AbstractSearchIndex; meta=nothing, store_db=true, parent="/")
    saveindex(file::JLD2.JLDFile, index::AbstractSearchIndex; meta=nothing, store_db=true, parent="/")

Saves the index into the given file using JLD2. It is recommended instead plain JLD2 since it simplifies
several typical operations and also allow that indexes perform pre-save and post-load operations
when indexes are saved and restored. 

# Arguments:
- `filename` or `file`: output file of the index.
- `index`: the index to be saved.
- `meta=nothing`: additional metadata to be saved with the index (it can be any object supported by JLD2).
- `parent="/"`: parent group in the h5 file
- `store_db=true` controls if the daset should stored, when `store_db=false` you need to give the dataset on loading.

"""

function saveindex(filename::AbstractString, index::AbstractSearchIndex; meta=nothing, store_db=true, parent="/")
    jldopen(filename, "w") do f
        saveindex(f, index; meta, store_db, parent)
    end
end

function saveindex(file::JLD2.JLDFile, index::AbstractSearchIndex; meta=nothing, store_db=true, parent="/")
    db = database(index)
    uses_stride_matrix_database = database(index) isa StrideMatrixDatabase
    index = copy(index; db=MatrixDatabase(zeros(Float32, 2, 2))) # the database is replaced in any case to handle it here and serializeindex forget about it

    if store_db
        stores_database = true
        db = uses_stride_matrix_database ? Matrix(db) : db
    else
        stores_database = false
        db = nothing
    end
    
    options = Dict(
        "uses_stride_matrix_database" => uses_stride_matrix_database,
        "stores_database" => stores_database
    )

    file[joinpath(parent, "options")] = options 
    file[joinpath(parent, "meta")] = meta
    file[joinpath(parent, "db")] = db
    serializeindex(file, parent, index, meta, options)
    nothing
end

"""
    serializeindex(file, parent::String, index::AbstractSearchIndex, meta, options::Dict)

Stores the index in the h5 file. It can be specialized to make any special treatment of the index
"""
function serializeindex(file, parent::String, index::AbstractSearchIndex, meta, options::Dict)
    file[joinpath(parent, "index")] = index
end

"""
    restoreindex(file, index::AbstractSearchIndex, meta, options::Dict)

Called on loading to restore any value on the index that depends on the new instance (i.e., datasets, pointers, etc.)
"""
restoreindex(file, parent, index::AbstractSearchIndex, meta, options::Dict; kwargs...) = index

"""
    loadindex(filename::AbstractString, db::Union{Nothing,AbstractDatabase}=nothing; kwargs...)

Loads an index from `filename`. If `db` is given it will replace the dataset on the loaded index

# Arguments
- `db`: Replaces the index database after loading with that specified in `db`
- `parent="/"`: Parent group of the index
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
    index = restoreindex(f, parent, index, meta, options; kwargs...)

    if db === nothing
        if options["stores_database"]
            db = f[joinpath(parent, "db")]
            if options["uses_stride_matrix_database"]
                db = copy(index; db=StrideMatrixDatabase(database(index)))
            end

            index = copy(index; db)
        else
            @warn "the database was not stored and was not passed to loadindex"
        end
    else
        index = copy(index; db)
    end

    index, meta
end
