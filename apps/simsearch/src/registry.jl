const INDEX_BUILDERS = Dict{String,Function}(
    "ExhaustiveSearch"         => (dist, db) -> SimilaritySearch.ExhaustiveSearch(dist, db),
    "ParallelExhaustiveSearch" => (dist, db) -> SimilaritySearch.Exact.ParallelExhaustiveSearch(dist, db),
    "SearchGraph"              => (dist, db) -> SimilaritySearch.SearchGraph(dist, db),
)

const DISTANCE_CONSTRUCTORS = Dict{String,Any}(
    "L1" => Dist.L1, "L2" => Dist.L2, "SqL2" => Dist.SqL2, "LInfty" => Dist.LInfty,
    "Cosine" => Dist.Cosine, "Angle" => Dist.Angle,
    "NormCosine" => Dist.NormCosine, "NormAngle" => Dist.NormAngle,
)

"""
    build_index(type, dist, db)

Constructs an (unindexed) `AbstractSearchIndex` of the given CLI `type` name over `db` with
distance `dist`. Raises on unknown `type`.
"""
function build_index(type::AbstractString, dist, db)
    ctor = get(INDEX_BUILDERS, type, nothing)
    ctor === nothing && error("unknown --type '$type'; supported: " *
                               join(sort(collect(keys(INDEX_BUILDERS))), ", "))
    ctor(dist, db)
end

"""
    parse_distance(spec)

Parses a CLI `--distance` string (e.g. `"Dist.SqL2"` or `"SqL2"`) into a distance object,
e.g. `Dist.SqL2()`. Only the zero-argument numeric-vector distance family is supported.
"""
function parse_distance(spec::AbstractString)
    name = startswith(spec, "Dist.") ? spec[6:end] : spec
    ctor = get(DISTANCE_CONSTRUCTORS, name, nothing)
    ctor === nothing && error("unknown --distance '$spec'; supported: " *
                               join(sort(collect(keys(DISTANCE_CONSTRUCTORS))), ", "))
    ctor()
end
