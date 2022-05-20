Our example can be run as follows:

```bash
JULIA_PROJECT=. julia -t auto -L ex.jl
```
inside the Julia REPL call the main function
```julia
main_mnist()
```

To run the scripts use 

```bash
python flatL2_allknn_mnist.py

python hnsw_allknn_mnist.py

python nndescent_allknn_mnist.py

python scann_index_allknn_mnist.py
```

These scripts will produce a set of result files. To get the recalls go to the results/mnist directory and run 

```bash 
julia ../../recall.jl
```

