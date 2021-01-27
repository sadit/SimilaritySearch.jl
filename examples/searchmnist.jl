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
end

# ╔═╡ 5cd87a9e-5506-11eb-2744-6f02144677ff
md"""
# Using SimilaritySearch with MNIST


This example shows how to search on the MNIST dataset


## Loading the required packages for the examples
"""

# ╔═╡ d8d27dbc-5507-11eb-20e9-0f16ddba080b
md"""
### Loading the dataset
As first step, you must download the dataset (MNIST, FashionMNIST) before fetching the train data, e.g.

```MNIST.download()```

apparently, this doesn't work on Pluto and must be done using the terminal directly and accepting the terms and conditions of using the dataset.

You can also accept it from Pluto passing keyworkd argument `i_accept_the_terms_of_use=true`, e.g., `MNIST.download(i_accept_the_terms_of_use=true)`.


"""

# ╔═╡ a23b0cae-455d-11eb-0e50-4dc31c050cc1
begin
	T, y = MNIST.traindata()
	n = size(T, 3)
	X = [Float32.(view(T, :, :, i)) for i in 1:n]
	length(X), size(X[1]), length(X[1])
end

# ╔═╡ 1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
index = ExhaustiveSearch(SqL2Distance(), X);
#index = PivotedSearch(L2Distance(), X, 16);
##index = sss(L2Distance(), X);
#index = distant_tournament(L2Distance(), X, 16, 3)
#index = Kvp(L2Distance(), X);
#index = SearchGraph(SqL2Distance(), X)

# ╔═╡ 5b743cbc-54fa-11eb-1be4-4b619e1070b2

begin
	sel = @bind example_symbol html"<input type='range' min='1' max='60000' step='1'>"
	md"""
	select the query object using the bar: $(sel), $n
	"""
end

# ╔═╡ def63abc-45e7-11eb-231d-11d94709acd3
begin
	@time res = search(index, X[example_symbol], KnnResult(10))
	qinverted = 1 .- X[example_symbol]' # just to distinguish easily
	h = hcat(qinverted,  [X[p.id]' for p in res]...)
	
	md""" $(size(h))

Query Id: $(example_symbol)
Label: $(y[example_symbol])
database size: $(length(X))
	
	
# Result:
$(Gray.(h))


note: the symbol is the query object and its colors has been inverted
	"""
end

# ╔═╡ Cell order:
# ╠═5cd87a9e-5506-11eb-2744-6f02144677ff
# ╠═c5af5d4a-455b-11eb-0b57-4d8d63615b85
# ╠═d8d27dbc-5507-11eb-20e9-0f16ddba080b
# ╠═a23b0cae-455d-11eb-0e50-4dc31c050cc1
# ╠═1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
# ╠═5b743cbc-54fa-11eb-1be4-4b619e1070b2
# ╠═def63abc-45e7-11eb-231d-11d94709acd3
