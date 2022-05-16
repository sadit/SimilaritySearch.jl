using SimilaritySearch, MLDatasets

function main_mnist(dist=SqL2Distance(), k=15)
	train, test = MNIST(split=:train), MNIST(split=:test)
	(w, h, n), m = size(train.features), size(test.features, 3)
	db = MatrixDatabase(reshape(train.features, w * h, n))
	queries = MatrixDatabase(reshape(test.features, w * h, m))

	G = SearchGraph(; dist, db)
	Gbuildtime = @elapsed index!(G; parallel_block=256)
	Gopttime = @elapsed optimize!(G, MinRecall(0.95))

	Gsearchtime = @elapsed searchbatch(G, queries, k; parallel=true)
	Gclosestpairtime = @elapsed closestpair(G; parallel=true)
	Gallknntime = @elapsed allknn(G, k; parallel=true)

	Gqueriespersecond = 1 / (Gsearchtime / m)


	E = ExhaustiveSearch(; dist, db)

	Esearchtime = @elapsed searchbatch(E, queries, k; parallel=true)
	Eclosestpairtime = @elapsed closestpair(E; parallel=true)
	Eallknntime = @elapsed allknn(E, k; parallel=true)

	Gqueriespersecond = 1 / (Esearchtime / m)


	@show Threads.nthreads()

	@info (; Equeriespersecond, Eclosestpairtime, Eallknntime)
	@info (; Gbuildtime, Gopttime, Gqueriespersecond, Gclosestpairtime, Gallknntime)
end


