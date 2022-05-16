# run as:
# JULIA_PROJECT=. srun -N1 -xgeoint0 --pty julia -t64 -L ex.jl

using SimilaritySearch, MLDatasets

function main_mnist(dist=SqL2Distance(), k=15)
	train, test = MNIST(split=:train), MNIST(split=:test)
	(w, h, n), m = size(train.features), size(test.features, 3)
	db = MatrixDatabase(reshape(train.features, w * h, n))
	queries = MatrixDatabase(reshape(test.features, w * h, m))

	

	G = SearchGraph(; dist, db)
	Gbuildtime = @elapsed index!(G; parallel_block=256)
	Gsearchtime = @elapsed searchbatch(G, queries, k; parallel=true)
	Gclosestpairtime = @elapsed closestpair(G; parallel=true)
	Gallknntime = @elapsed Gid, Gdist = allknn(G, k; parallel=true)

	Gqueriespersecond = 1 / (Gsearchtime / m)

	
	_Gopttime = @elapsed optimize!(G, MinRecall(0.95))
	_Gsearchtime = @elapsed searchbatch(G, queries, k; parallel=true)
	_Gclosestpairtime = @elapsed closestpair(G; parallel=true)
	_Gallknntime = @elapsed _Gid, _Gdist = allknn(G, k; parallel=true)

	_Gqueriespersecond = 1 / (_Gsearchtime / m)


	E = ExhaustiveSearch(; dist, db)

	Esearchtime = @elapsed searchbatch(E, queries, k; parallel=true)
	Eclosestpairtime = @elapsed closestpair(E; parallel=true)
	Eallknntime = @elapsed Eid, Edist = allknn(E, k; parallel=true)

	Gqueriespersecond = 1 / (Gsearchtime / m)
        Equeriespersecond = 1 / (Esearchtime / m)

	@show Threads.nthreads()

	@info (; Equeriespersecond, Eclosestpairtime, Eallknntime)
	@info (; Gbuildtime, Gqueriespersecond, Gclosestpairtime, Gallknntime, recall=macrorecall(Eid, Gid))
	@info (; Gbuildtime, _Gopttime, _Gqueriespersecond, _Gclosestpairtime, _Gallknntime, recall=macrorecall(Eid, _Gid))
end


