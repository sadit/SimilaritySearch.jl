# run as:
# JULIA_PROJECT=. srun -N1 -xgeoint0 --pty julia -t64 -L ex.jl

using SimilaritySearch, MLDatasets, DataFrames, Serialization #, LoopVectorization, StrideArrays

#=
struct LvSqL2Distance <: SemiMetric end

function SimilaritySearch.evaluate(::LvSqL2Distance, u, v)
	d = zero(Float32)
	@turbo for i in eachindex(u)
		d += (u[i] - v[i])^2
	end

	d
end

function load_data_stridearray()
	train, test = MNIST(split=:train), MNIST(split=:test)
	(w, h, n), m = size(train.features), size(test.features, 3)
	M = reshape(train.features, w * h, n)
	db = MatrixDatabase(StrideArray(M, StaticInt.(size(M))))
	M = reshape(test.features, w * h, m)
	queries = MatrixDatabase(StrideArray(M, StaticInt.(size(M))))
	db, queries, LvSqL2Distance()
end

function load_data_vectors()
	train, test = MNIST(split=:train), MNIST(split=:test)
	(w, h, n), m = size(train.features), size(test.features, 3)
	db = VectorDatabase(reshape(train.features, w * h, n))
	queries = VectorDatabase(reshape(test.features, w * h, m))
	db, queries, SqL2Distance()
end

=#

function load_data_matrix_warmup()
	MatrixDatabase(rand(Float32, 2, 16)), MatrixDatabase(rand(Float32, 2, 16)), SqL2Distance()
end

function load_data_matrix()
	train, test = MNIST(split=:train), MNIST(split=:test)
	(w, h, n), m = size(train.features), size(test.features, 3)
	db = MatrixDatabase(reshape(train.features, w * h, n))
	queries = MatrixDatabase(reshape(test.features, w * h, m))
	db, queries, SqL2Distance()
end

function test_searchgraph(G; method, queries, k, mem, Eid, df, buildtime::Real, opttime::Real)
	searchtime = @elapsed searchbatch(G, queries, k; parallel=true)
	closestpairtime = @elapsed closestpair(G; parallel=true)
	allknntime = @elapsed Gid, _ = allknn(G, k; parallel=true)

	push!(df, (method, buildtime, opttime, searchtime, closestpairtime, allknntime, mem, macrorecall(Eid, Gid)))
end

function main(db, queries, dist, k, warming)
	df = DataFrame(method=String[], build=Float64[], optim=Float64[], searchqueries=Float64[], closestpair=Float64[], allknn=Float64[], mem=Float64[], recall=Float64[])
	!warming && @info "computing gold standard"
	E = ExhaustiveSearch(; dist, db)
	if !warming && isfile("E.bin")
		Ereg, Eid = deserialize("E.bin")
	else
		Esearchtime = @elapsed searchbatch(E, queries, k; parallel=true)
		Eclosestpairtime = @elapsed closestpair(E; parallel=true)
		Eallknntime = @elapsed Eid, _ = allknn(E, k; parallel=true)
		mem = sizeof(db.matrix) / 2^20
		Ereg = ("ExhaustiveSearch", 0.0, 0.0, Esearchtime, Eclosestpairtime, Eallknntime, mem, 1.0)
		!warming && serialize("E.bin", (Ereg, Eid))
	end

	push!(df, Ereg)
	G = SearchGraph(; dist, db, verbose=false)
	buildtime = @elapsed index!(G; parallel_block=512)
	serialize("G.bin", G)
	mem = filesize("G.bin") / 2^20
	test_searchgraph(G; method="ParetoRecall", queries, k, buildtime, mem, opttime=0.0, Eid, df)
	opttime = @elapsed optimize!(G, MinRecall(0.9))
	test_searchgraph(G; method="MinRecall(0.9)", queries, k, buildtime, mem, opttime, Eid, df)
	opttime = @elapsed optimize!(G, MinRecall(0.95))
	test_searchgraph(G; method="MinRecall(0.95)", queries, k, buildtime, mem, opttime, Eid, df)
	#opttime = @elapsed optimize!(G, MinRecall(0.99))
	#test_searchgraph(G; method="MinRecall(0.99)", queries, k, buildtime, mem, opttime, Eid, df)
	opttime = @elapsed optimize!(G, MinRecall(0.6))
	test_searchgraph(G; method="MinRecall(0.6)", queries, k, buildtime, mem, opttime, Eid, df)
	G, Eid, df
end

function main_mnist(k=32)
	db, queries, dist = load_data_matrix_warmup()
	main(db, queries, dist, 3, true)

	db, queries, dist = load_data_matrix()
	_, Eid, df = main(db, queries, dist, k, false)

	@info df
	db = VectorDatabase([(db[i] .>= 0.5).chunks for i in eachindex(db)])
	queries = VectorDatabase([(queries[i] .>= 0.5).chunks for i in eachindex(queries)])

    H = SearchGraph(; db, dist=BinaryHammingDistance())
    buildtime = @elapsed index!(H; parallel_block=512)
	opttime = @elapsed optimize!(H, MinRecall(0.9))
	serialize("H.bin", H)
	mem = filesize("H.bin") / 2^20

	test_searchgraph(H; method="Bin-Hamming", queries, k, buildtime, mem, opttime, Eid, df)
	@show Threads.nthreads()
	df
end