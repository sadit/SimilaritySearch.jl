Our example can be run as follows:

```bash
JULIA_PROJECT=. julia -e 'using Pkg; Pkg.instantiate()'   # only the first time, it installs package's dependencies
JULIA_PROJECT=. julia -t auto -L ex.jl
```

This command will launch the Julia REPL, then call the example function
```julia
main_mnist()
```

The rest of the scripts will produce our comparison results.

```bash
python flatL2_allknn_mnist.py

python hnsw_allknn_mnist.py

python nndescent_allknn_mnist.py

python scann_index_allknn_mnist.py
```

These scripts produce a set of result files. To get the recalls go to the results/mnist directory and run 

```bash 
JULIA_PROJECT=../.. julia ../../recall.jl
```

