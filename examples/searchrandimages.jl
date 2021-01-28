### A Pluto.jl notebook ###
# v0.12.19

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c5af5d4a-455b-11eb-0b57-4d8d63615b85
begin
	using SimilaritySearch  # this is the only package needed
	using MLDatasets # this dataset allows access the dataset
	using Colors # allows to show images in the Pluto notebook
	using PlutoUI
end

# ╔═╡ 5cd87a9e-5506-11eb-2744-6f02144677ff
md"""
# Using SimilaritySearch with synthetic images


This example shows how to search on synthetic images


## Loading the required packages for the examples
"""

# ╔═╡ d8d27dbc-5507-11eb-20e9-0f16ddba080b
md"""
### Creating the dataset

We will create a $n$ random matrices (a.k.a. synthetic image)

"""

# ╔═╡ c6ca5d24-616c-11eb-1031-4b943ecc633c
function create_object(dim)
	s = randn(Float32, dim, dim)
	s .= max.(s, 0.0)
	s .= 1 .- s ./ maximum(s) # just to get many ~white pixels
	s
end

# ╔═╡ a23b0cae-455d-11eb-0e50-4dc31c050cc1
begin
	n = 10_000
	dim = 7
	X = [create_object(dim) for i in 1:n]
	length(X), size(X[1]), length(X[1])
	X[1]
end

# ╔═╡ a2984e1a-6165-11eb-20e0-6d16394dc05a
md"""
## Creating the index
Firstly, we need to create an index; please uncomment any index you want to try
"""

# ╔═╡ 1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
index = ExhaustiveSearch(L2Distance(), X);
#index = PivotedSearch(L2Distance(), X, 16);
#index = sss(L2Distance(), X);
#index = distant_tournament(L2Distance(), X, 16, 3)
#index = Kvp(L2Distance(), X);
#index = SearchGraph(SqL2Distance(), X)

# ╔═╡ e600bf26-6167-11eb-08b8-4904cd4d787f
md"""## Querying
Some indexes support changing the querying distance (ExhaustiveSearch & SearchGraph in the previous list), others will just work but without guarantee
"""

# ╔═╡ 5b743cbc-54fa-11eb-1be4-4b619e1070b2

begin
	qsel = @bind qid Slider(1:n, default=10, show_value=true)
	dsel = @bind distname Select([
			"L2" => "Euclidean",
			"L1" => "Manhattan",
			"LInf" => "Chevyshev",
			"Angle" => "Angle",
			"Cosine" => "Cosine",
			], default="L2")

	md"""
	select the query object: $(qsel)
	$(dsel)
	"""
end

# ╔═╡ def63abc-45e7-11eb-231d-11d94709acd3
begin
	dist = Dict(
		"L1" => L1Distance(),
		"L2" => L2Distance(),
		"LInf" => LInftyDistance(),
		"Angle" => AngleDistance(),
		"Cosine" => CosineDistance()
	)[distname]
	index_ = copy(index, dist=dist) # makes a copy of the index with a different distance function
	res = KnnResult(7)
	q = X[qid]
	sep = ones(Float32, dim, 2)
	R = [q, sep, sep]
	with_terminal() do
		@time search(index_, X[qid], res)

		for (i, p) in enumerate(res)
			push!(R, Float32.(X[p.id]))
			push!(R, sep)
			println("$(i)nn: $(p.id) => $(p.dist)")
		end
	end
	
	h = hcat(R...)
	
	md""" $(size(h))

Query Id: $(qid)
database size: $(length(X))
dist: $(distname)
	
# Result:
$(Gray.(h))


	"""
end

# ╔═╡ Cell order:
# ╠═5cd87a9e-5506-11eb-2744-6f02144677ff
# ╠═c5af5d4a-455b-11eb-0b57-4d8d63615b85
# ╠═d8d27dbc-5507-11eb-20e9-0f16ddba080b
# ╠═c6ca5d24-616c-11eb-1031-4b943ecc633c
# ╠═a23b0cae-455d-11eb-0e50-4dc31c050cc1
# ╠═a2984e1a-6165-11eb-20e0-6d16394dc05a
# ╠═1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
# ╠═e600bf26-6167-11eb-08b8-4904cd4d787f
# ╠═5b743cbc-54fa-11eb-1be4-4b619e1070b2
# ╠═def63abc-45e7-11eb-231d-11d94709acd3
