# This file is a part of SimilaritySearch.jl

"""
    saveindex(filename::AbstractString, index::AbstractSearchIndex; store_db=true)

Saves the index into the given file using JLD2. It is recommended instead plain JLD2 since it simplifies
several typical operations and also allow that indexes perform pre-save and post-load operations
when indexes are saved and restored. 

- `store_db=true` controls if the daset should stored, when `store_db=false` you need to give the dataset on loading.

"""
function saveindex(filename::AbstractString, index::AbstractSearchIndex, meta::Dict)
    jldsave(filename; index, meta)
end

function saveindex(filename::AbstractString, index::AbstractSearchIndex; store_db=true)
    uses_stride_matrix_database = database(index) isa StrideMatrixDatabase
    if store_db
        stores_database = true
    else
        stores_database = false
        index = copy(index; db=MatrixDatabase(zeros(Float32, 2, 2)))
    end

    meta = Dict(
        "uses_stride_matrix_database" => uses_stride_matrix_database,
        "stores_database" => stores_database
    )
    saveindex(filename::AbstractString, index::AbstractSearchIndex, meta)

    index
end

"""
    restoreindex(index::AbstractSearchIndex, meta::Dict, f)

Called on loading to restore any value on the index that depends on the new instance (i.e., datasets, pointers, etc.)
"""
function restoreindex(index::AbstractSearchIndex, meta::Dict, f; kwargs...)
    index, meta # nothing for most indexes
end

"""
    loadindex(filename::AbstractString, db::Union{Nothing,AbstractDatabase}=nothing; kwargs...)

Loads an index from `filename`, optionally, if `db` is given it will replace the dataset on the loaded index, but it is
mandatory if the index was saved with `store_db=false`
"""
function loadindex(filename::AbstractString, db::Union{Nothing,AbstractDatabase}=nothing; kwargs...)
    index, meta = jldopen(filename) do f
        # @info typeof(f), keys(f) # JLD2.JLDFile
        restoreindex(f["index"], f["meta"], f; kwargs...)
    end

    if db === nothing
        if !meta["stores_database"]
            @warn "the database was not stored for $filename and was not passed to loadindex"
        end
    else
        index = copy(index; db)
    end

    if meta["uses_stride_matrix_database"]
        index = copy(index; db=StrideMatrixDatabase(database(index)))
    end

    index
end
