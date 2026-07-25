# This file is a part of SimilaritySearch.jl

using Random: randperm
using Combinatorics: combinations

"""
    index!(idx::SearchGraph, ::SearchGraphContext, ::Val{:knr}, knr::Matrix{IdDist};
        subset_list::Vector{Int}=Int[2],
        bpow::Float64=0.8,
        hints_size::Int=100
    )

Fast non-incremental construction of a `SearchGraph` index using subsets of the nearest references to warmup the graph.
You may want to use `rebuild` on this fast construction to improve the structure.

"""
function index!(idx::SearchGraph, ::SearchGraphContext, ::Val{:knr}, knr::Matrix{IdDist};
    comb_list::Vector{Int}=Int[2],
    bpow::Float64=0.8,
    hints_size::Int=100
)
    k, n = size(knr)
    length(idx) == 0 || throw(ArgumentError("This construction method accepts only not previously created graphs"))
    n == length(database(idx)) || throw(ArgumentError("The given knr matrix doesn't correpond with the given index: sizes doesn't matches"))
    n > size(knr, 1) >= k >= 1 || throw(ArgumentError("The following must be ensured: |db| > numrefs >= k >= 1"))
    n > hints_size || throw(ArgumentError("hints_size cannot be larger than the dataset (delta * log n could be a good value)"))
    0 < bpow < 1 || throw(ArgumentError("bpow is between 0 and 1 and controls how clusters are linked"))
    length(comb_list) > 0 || throw(ArgumentError("comb_list must be provided"))

    adj_sets = [Set{UInt32}() for _ in 1:n]

    #=for c in eachcol(knr)
        sort!(c, by=p -> p.id)
    end=#

    function link_clusters(adj_sets, clusters, bpow)
        for cl in values(clusters)
            m = length(cl)
            m <= 1 && continue
            #=p = let b = binomial(m, 2)
                b^bpow / b
            end=#
            p = m^bpow / m

            for i in 1:m
                if i > 1 && rand() > p
                    continue
                end
                u = cl[i]
                for j in (i+1):m
                    #rand() > p && continue
                    v = cl[j]
                    push!(adj_sets[u], v)
                    push!(adj_sets[v], u) # non-directed link
                end
            end

            #=p = m^bpow / m

            for i in 1:m
                if i > 1 && rand() > p
                    continue
                end
                u = cl[i]
                for j in (i+1):m
                    v = cl[j]
                    push!(adj_sets[u], v)
                    push!(adj_sets[v], u) # non-directed link
                end

            end
            =#
        end
    end

    # create clusters based on common neighborhoods
    for l in comb_list
        clusters = Dict{NTuple{l,UInt32},Vector{UInt32}}()
        @time "comb" for L in combinations(1:k, l) # reverse order
            for objID in 1:n
                prefix = ntuple(i -> knr[L[i], objID].id, l)
                push!(get!(Vector{UInt32}, clusters, prefix), objID)
            end
        end
        #=
                @time "comb" for objID in 1:n
                    prefix = ntuple(i -> knr[i, objID].id, l)
                    push!(get!(Vector{UInt32}, clusters, prefix), objID)
                end
                =#
        @time "links" link_clusters(adj_sets, clusters, bpow)
    end

    let H = collect(zip(1:n, length.(adj_sets)))
        sort!(H, by=last, rev=true)
        for h in H
            push!(idx.hints, first(h))
            length(idx.hints) < hints_size || break
        end
    end

    resize!(idx.adj, n)
    for i in 1:n
        L = collect(adj_sets[i])
        add!(idx.adj, i, L)
    end

    idx.len[] = n
    idx
end

function index!(idx::SearchGraph, ctx::SearchGraphContext, kind::Val{:knr};
    numrefs::Integer=ceil(Int, sqrt(length(database(idx)))),
    k::Integer=4,
    sample_method::Symbol=:fft, #:random, :fft
    comb_list::Vector{Int}=Int[2],
    bpow::Float64=0.8,
    hints_size::Int=100,
    verbose::Bool=true
)
    dist = distance(idx)
    db = database(idx)
    n = length(db)

    ref_ids = if sample_method === :fft
        fft_res = fft(dist, db, numrefs; verbose)
        fft_res.centers
    elseif sample_method === :random
        randperm(n)[1:numrefs]
    else
        throw(ArgumentError("Unknown sample_method: $sample_method. Use :fft or :random"))
    end

    sort!(ref_ids)
    refs_db = SubDatabase(db, ref_ids)

    # 2. Project DB onto references via searchbatch on ExhaustiveSearch
    seq = ExhaustiveSearch(dist, refs_db)
    ectx = GenericContext(KnnSorted)
    @time "KNR" knr = searchbatch(seq, ectx, db, k) # size (k, n)
    index!(idx, ctx, kind, knr; bpow, comb_list, hints_size)
end
