# This file is a part of SimilaritySearch.jl
# This file was created initially by Antigravity by ideas and inscructions of Eric S. Tellez

using Random: randperm

"""
    index!(idx::SearchGraph, ctx, ::Val{:prefixes};
            numrefs::Integer=max(16, ceil(Int, sqrt(length(db)))),
            k::Integer=4,
            sample_method::Symbol=:fft, #:random, :fft
            probfactor::Float64=0.9,
            max_pairs_per_node::Integer=32,
            verbose::Bool=true)

Fast non-incremental construction of a `SearchGraph` index using prefixes of a set of nearest references to warmup the graph.
You may want to use `rebuild` on this fast construction to improve the structure.

1. Selects a reference sample `refs` of size `numrefs` from `db` using `sample_method` (`:fft` or `:random`).
2. Projects `db` onto `refs` by finding the `k` nearest references for each item in `db` using `searchbatch` on `ExhaustiveSearch`.
3. Connects references to each other (reference backbone) and connects each item to its nearest reference center.
4. Connects items in `db` sharing reference prefixes of length `L` (from `k` down to `1`) with probability `P(L) = probfactor^(k - L)`.
5. Creates non-directed (undirected) reverse links.
6. Sets initial search `hints` to `ref_ids`.
"""
function index!(idx::SearchGraph, ::SearchGraphContext, ::Val{:prefixes};
    numrefs::Integer=max(16, ceil(Int, sqrt(length(database(idx))))),
    k::Integer=4,
    sample_method::Symbol=:fft, #:random, :fft
    probfactor::Float64=0.9,
    max_pairs_per_node::Integer=32,
    verbose::Bool=true
)
    db = database(idx)
    dist = distance(idx)
    n = length(db)
    length(idx) == 0 || throw(ArgumentError("This construction method accepts only not previously created graphs"))
    n > numrefs >= k > 1 || throw(ArgumentError("The following must be ensured: |db| > numrefs >= k > 1"))

    # 1. Select reference sample
    ref_ids = if sample_method === :fft
        fft_res = fft(dist, db, numrefs; verbose)
        fft_res.centers
    elseif sample_method === :random
        randperm(n)[1:numrefs]
    else
        throw(ArgumentError("Unknown sample_method: $sample_method. Use :fft or :random"))
    end

    refs_db = SubDatabase(db, ref_ids)

    # 2. Project DB onto references via searchbatch on ExhaustiveSearch
    seq = ExhaustiveSearch(dist, refs_db)
    ectx = GenericContext(KnnSorted)
    knns = searchbatch(seq, ectx, db, k) # size (k, n)

    # 3. Create adjacency sets for each node to build undirected graph
    adj_sets = [Set{UInt32}() for _ in 1:n]

    # 3b. Connect each item i to its nearest reference center(s)
    @inbounds for i in 1:n
        for l in 1:min(2, k)
            ref_global = UInt32(ref_ids[knns[l, i].id])
            if i != ref_global
                push!(adj_sets[i], ref_global)
                push!(adj_sets[ref_global], UInt32(i))
            end
        end
    end

    # 3c. Prefix matching via Dict hashmap
    for L in k:-1:1
        prob = probfactor^(k - L)
        groups = Dict{NTuple{L,UInt32},Vector{UInt32}}()
        @inbounds for i in 1:n
            prefix = ntuple(l -> knns[l, i].id, L)
            push!(get!(Vector{UInt32}, groups, prefix), UInt32(i))
        end

        # Connect items sharing prefix of length L with probability prob
        for bucket in values(groups)
            m = length(bucket)
            m <= 1 && continue

            if m <= max_pairs_per_node
                # Connect pairs in small buckets
                @inbounds for a in 1:m
                    u = bucket[a]
                    for b in (a+1):m
                        if rand() <= prob
                            v = bucket[b]
                            push!(adj_sets[u], v)
                            push!(adj_sets[v], u) # non-directed link
                        end
                    end
                end
            else
                # Sample neighbors in larger buckets to keep time bounded
                @inbounds for a in 1:m
                    u = bucket[a]
                    for _ in 1:max_pairs_per_node
                        if rand() <= prob
                            b = rand(1:m)
                            b == a && continue
                            v = bucket[b]
                            push!(adj_sets[u], v)
                            push!(adj_sets[v], u) # non-directed link
                        end
                    end
                end
            end
        end
    end

    resize!(idx.adj, n)
    for i in 1:n
        add!(idx.adj, i, collect(adj_sets[i]))
    end
    append!(idx.hints, ref_ids)
    idx.len[] = n
    idx
end

