# Plan: KnnSorted/KnnHeap — Matrices separadas de IDs y Distancias

## Motivación y objetivo

`KnnSorted` y `KnnHeap` almacenan actualmente sus elementos como `Vector{IdDist}` —
estructura AoS (*Array of Structs*) con `id::UInt32` y `dist::Float32` entrelazados.
El objetivo es migrar a SoA (*Struct of Arrays*): dos vectores separados
`ids::Vector{UInt32}` y `dists::Vector{Float32}`.

**Beneficios:**
- Mejor localidad de caché al operar solo sobre distancias (el caso más frecuente: el guard check `covradius`).
- Eliminación de `IdDist` como tipo de **almacenamiento interno** (sigue existiendo como par efímero devuelto por `nearest`/`frontier`/`pop_min!`/`pop_max!`).
- `searchbatch`/`allknn` devuelven `(Matrix{UInt32}, Matrix{Float32})` — más natural.
- El código de los callers que recibían `Matrix{IdDist}` pasa a recibir dos matrices.

**Decisiones de diseño tomadas:**
- `permindex.jl` y `basket-list.jl` se migran completamente.
- Ruptura de API es aceptable con documentación adecuada.
- `macrorecall(gold_ids, ids)` — solo matrices de ids (`UInt32`); el recall no depende de distancias.

---

## Proposed Changes

### Capa 1 — Núcleo `src/pqueue/`

#### [MODIFY] `src/pqueue/heap.jl`

Las primitivas actuales (`heapswap!`, `heapfix_up!`, `heapfix_down!`, `heapsort!`) operan
sobre un único array de `IdDist`. Pasan a operar sobre **dos arrays en paralelo**:
comparaciones solo sobre `dists`, swaps aplicados pareados. **No se usa `sortperm`.**

```julia
function heapswap!(ids, dists, i, j)
    @inbounds ids[i],   ids[j]   = ids[j],   ids[i]
    @inbounds dists[i], dists[j] = dists[j], dists[i]
end

function heapfix_up!(order, ids, dists, i)
    @inbounds while (p = heapparent(i)) > 0
        if lt(order, dists[p], dists[i])
            heapswap!(ids, dists, i, p); i = p
        else break end
    end
    i
end
# heapfix_down!, heapsort! análogos
```

#### [MODIFY] `src/pqueue/knnsorted.jl`

Nueva struct con dos vectores tipados:

```julia
mutable struct KnnSorted{IDS<:AbstractVector{UInt32},
                         DSTS<:AbstractVector{Float32}} <: AbstractKnn
    ids::IDS
    dists::DSTS
    sp::Int32
    ep::Int32
    maxlen::Int32
end
```

`sort_last_item!` hace insertion sort pareado (sin `sortperm`):

```julia
@inline function sort_last_item!(ids, dists, sp, ep)
    sp == ep && return nothing
    @inbounds item_id, item_dist = ids[ep], dists[ep]
    i = ep - 1
    @inbounds lt(DistOrder, dists[i], item_dist) && return nothing
    @inbounds while i >= sp
        if lt(DistOrder, item_dist, dists[i])
            ids[i+1] = ids[i]; dists[i+1] = dists[i]
        else
            ids[i+1] = item_id; dists[i+1] = item_dist
            return nothing
        end
        i -= 1
    end
    @inbounds ids[sp] = item_id; dists[sp] = item_dist
    nothing
end
```

`nearest`/`frontier` siguen devolviendo `IdDist` (tipo efímero, no de almacenamiento):

```julia
@inline nearest(res::KnnSorted)  = @inbounds IdDist(res.ids[res.sp], res.dists[res.sp])
@inline frontier(res::KnnSorted) = @inbounds IdDist(res.ids[res.ep], res.dists[res.ep])
```

#### [MODIFY] `src/pqueue/knnheap.jl`

```julia
mutable struct KnnHeap{IDS<:AbstractVector{UInt32},
                       DSTS<:AbstractVector{Float32}} <: AbstractKnn
    ids::IDS
    dists::DSTS
    min_id::UInt32
    min_dist::Float32
    len::Int32
    maxlen::Int32
end
```

`push_item!`, `pop_max!`, `reuse!` adaptados en consecuencia.

#### [MODIFY] `src/pqueue/pqueue.jl`

- `knnqueue(T, ids, dists)` — desde dos vectores ya asignados (para vistas de matrices).
- `knnqueue(T, k::Int)` — asigna `zeros(UInt32, k)` + `zeros(Float32, k)`.
- `IdView(knn::KnnSorted/KnnHeap)` y `DistView(...)` se simplifican: delegan directo a `ids`/`dists`.
- Se eliminan los métodos `convert` de/a `Matrix{IdDist}`.

---

### Capa 2 — API batch `src/SimilaritySearch.jl`

#### [MODIFY] `src/SimilaritySearch.jl`

```julia
# Nueva firma — devuelve tupla de dos matrices
function searchbatch(index, ctx, Q, k::Integer; sorted=true)
    ids   = zeros(UInt32,  k, length(Q))
    dists = zeros(Float32, k, length(Q))
    searchbatch!(index, ctx, Q, ids, dists; sorted)
end

# Nueva firma in-place — recibe dos matrices
function searchbatch!(index, ctx, Q,
                      ids::AbstractMatrix{UInt32},
                      dists::AbstractMatrix{Float32}; sorted=false)
    @BATCHES ... begin
    @LOOP for j in 1:length(Q)
        res = knnqueue(bctx, view(ids, :, j), view(dists, :, j))
        search(index, bctx, Q[j], res)
        sorted && sortitems!(res)
    end
    end
    ids, dists
end
```

---

### Capa 3 — Algoritmos de alto nivel

#### [MODIFY] `src/allknn.jl`
Devuelve `(ids, dists)`. Variable interna `knns` pasa a dos matrices. `knnqueue` recibe vistas pareadas.

#### [MODIFY] `src/closestpair.jl`
`knns = zeros(IdDist, min_k, @nbatches())` → dos matrices separadas.

#### [MODIFY] `src/neardup.jl`
`knns = Matrix{IdDist}(...)` → dos matrices. Accesos `knns[1, i]` → `(ids[1,i], dists[1,i])`.

#### [MODIFY] `src/hsp.jl`
Recibe `(ids, dists)` en lugar de `knns::Matrix{IdDist}`.

#### [MODIFY] `src/rerank.jl`
- Variante por columna: `rerank!(dist, db, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32})`.
- Variante batch: `rerank!(dist, db, queries, ids::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32})`.

#### [MODIFY] `src/perf.jl`
`macrorecall(gold_ids::AbstractMatrix{UInt32}, ids::AbstractMatrix{UInt32})` — solo matrices de ids.

#### [MODIFY] `src/opt.jl`
Adaptación menor a nuevas firmas.

---

### Capa 4 — `SearchGraph` y contextos

#### [MODIFY] `src/searchgraph/context.jl`
`beams::Matrix{IdDist}` → `beam_ids::Matrix{UInt32}` + `beam_dists::Matrix{Float32}`.
`getbeam(ctx, i)` devuelve `(view(beam_ids,:,i), view(beam_dists,:,i))`.

#### [MODIFY] `src/searchgraph/rebuild.jl`
`qcache::Matrix{IdDist}` → dos matrices; `knnqueue` recibe vistas pareadas.

#### [MODIFY] `src/searchgraph/insertions.jl`
Idem.

#### [MODIFY] `src/searchgraph/beamsearch.jl`
`c = IdDist(childID, d)` y `push_item!(res, c)` → `push_item!(res, childID, d)`.

#### [MODIFY] `src/searchgraph/staticindexing.jl`
`index!(... knr::Matrix{IdDist})` → `index!(... ids::Matrix{UInt32}, dists::Matrix{Float32})`.

---

### Capa 5 — Archivos adicionales

#### [MODIFY] `src/permindex.jl`
`res.items[i] = IdDist(p.π[x.id], x.dist)` → adaptado a la nueva struct (ids/dists separados).

#### [MODIFY] `src/exact/basket-list.jl`
`baskets::Vector{Vector{IdDist}}` → `Vector{Tuple{Vector{UInt32}, Vector{Float32}}}` o equivalente.

---

### Capa 6 — Tests

#### [MODIFY] `test/testresults.jl`
`gold = IdDist[]` → `gold_ids = UInt32[]; gold_dists = Float32[]`.

#### [MODIFY] `test/testsearchgraph.jl`
`zeros(IdDist, k, n)` → `(zeros(UInt32,k,n), zeros(Float32,k,n))`.
`macrorecall(gold, knns)` → `macrorecall(gold_ids, ids)`.

#### [MODIFY] Otros tests que usan `searchbatch` / `IdDist` directamente:
`testexact.jl`, `testallknn.jl`, `testhsp.jl`, `testinvertedfiles.jl`, `testindexingprefixes.jl`.

---

## Orden de ejecución

1. `src/pqueue/heap.jl` — primitivas pareadas
2. `src/pqueue/knnsorted.jl` — nueva struct + métodos
3. `src/pqueue/knnheap.jl` — nueva struct + métodos
4. `src/pqueue/pqueue.jl` — nuevos `knnqueue`, `IdView`/`DistView` simplificados
5. `src/SimilaritySearch.jl` — `searchbatch` / `searchbatch!`
6. `src/searchgraph/context.jl` — `beam_ids` + `beam_dists`
7. `src/searchgraph/{rebuild,insertions,beamsearch,staticindexing}.jl`
8. `src/{allknn,closestpair,neardup,hsp,rerank,opt,perf}.jl`
9. `src/permindex.jl`, `src/exact/basket-list.jl`
10. Tests

---

## Verification Plan

```sh
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

- `searchbatch` devuelve `(ids::Matrix{UInt32}, dists::Matrix{Float32})`.
- `allknn` devuelve `(ids, dists)`.
- Recall en `testsearchgraph.jl` ≥ 0.9.
- `macrorecall(gold_ids, ids)` funciona.
