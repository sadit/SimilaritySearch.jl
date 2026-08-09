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


**Invariante de `KnnSorted`**: el rango `ids[sp:ep]` / `dists[sp:ep]` está **siempre
ordenado de menor a mayor** por distancia. `sort_last_item!` es el único lugar que
inserta un elemento nuevo (en la posición `ep`) y es responsable de restaurar ese
invariante. Todos los demás métodos pueden asumir sin verificación que el array está
ordenado.

`sort_last_item!` localiza el punto de inserción con **búsqueda binaria** sobre `dists`
(aprovechando el invariante de orden) y luego desplaza el bloque de elementos con un
**shift vectorial** (`copyto!`). La búsqueda binaria recorre solo el array de `dists`
(Float32 contiguos, cache-friendly), y el shift es un movimiento de bloque que los
compiladores/CPUs pueden vectorizar:

```julia
@inline function sort_last_item!(ids, dists, sp, ep)
    sp == ep && return nothing
    @inbounds item_id, item_dist = ids[ep], dists[ep]
    @inbounds dists[ep-1] <= item_dist && return nothing  # ya está ordenado

    # Búsqueda binaria del punto de inserción en dists[sp:ep-1]
    lo, hi = sp, ep - 1
    @inbounds while lo < hi
        mid = (lo + hi) >>> 1
        if dists[mid] <= item_dist
            lo = mid + 1
        else
            hi = mid
        end
    end
    # lo == primer índice donde dists[lo] > item_dist → insertar aquí

    # Shift en bloque: mover ids[lo:ep-1] → ids[lo+1:ep]
    #                   y    dists[lo:ep-1] → dists[lo+1:ep]
    @inbounds copyto!(ids,   lo + 1, ids,   lo, ep - lo)
    @inbounds copyto!(dists, lo + 1, dists, lo, ep - lo)

    @inbounds ids[lo]   = item_id
    @inbounds dists[lo] = item_dist
    nothing
end
```

Ventajas respecto al insertion sort clásico:
- La búsqueda binaria toca solo `dists` (Float32 contiguos) — menor presión de caché.
- El `copyto!` emite un `memmove` vectorizado por el compilador (SIMD) en lugar de
  writes individuales intercalados entre ids y dists.
- Para k típicos (8–64) el costo de binary search es O(log k) comparaciones,
  versus O(k) en el caso peor del scan lineal.


`nearest`/`frontier` siguen devolviendo `IdDist` (tipo efímero, no de almacenamiento):

```julia
@inline nearest(res::KnnSorted)  = @inbounds IdDist(res.ids[res.sp], res.dists[res.sp])
@inline frontier(res::KnnSorted) = @inbounds IdDist(res.ids[res.ep], res.dists[res.ep])
```

`viewitems(res::KnnSorted)` ya no puede devolver un `view` directo del array de items (no existe). Debe devolver una vista de ambos arrays, representada como un iterador o una estructura par — ver nota abajo sobre `viewitems`.

> **Nota sobre `viewitems`**: Actualmente `viewitems(res)` devuelve un `AbstractVector{IdDist}`. Con SoA esto ya no es posible sin copia. Las opciones son:
> - **Opción A** (recomendada): `viewitems` devuelve un `StructOfArraysView` ligero que implementa indexado `[i] -> IdDist(ids[sp+i-1], dists[sp+i-1])` — cero copias, compatible con el iterador existente. Solo se necesita un tipo wrapper.
> - **Opción B**: `viewitems` desaparece de la API pública; los callers acceden directamente a `IdView(res)` y `DistView(res)`.
>
> `viewitems` se usa en `rerank!`, `hsp_proximal_neighborhood_filter!`, `hsp_distal_neighborhood_filter!`, y en los tests. La Opción A minimiza el impacto.

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

`push_item!(res::KnnHeap, item::IdDist)` → `push_item!(res::KnnHeap, id::UInt32, dist::Float32)`.
El `item::IdDist` overload puede seguir existiendo como wrapper delgado.

`pop_max!` devuelve `IdDist` (efímero):
```julia
@inline function pop_max!(res::KnnHeap)
    id, dist = res.ids[1], res.dists[1]
    len = res.len
    heapswap!(res.ids, res.dists, 1, len)
    len -= 1
    heapfix_down!(DistOrder, res.ids, res.dists, len)
    res.len = len
    IdDist(id, dist)
end
```

`reuse!` se simplifica: ya no hay `min::IdDist`, sino `min_id` y `min_dist`:
```julia
@inline function reuse!(res::KnnHeap, maxlen=length(res.ids))
    res.min_id   = zero(UInt32)
    res.min_dist = typemax(Float32)
    res.len      = 0
    res.maxlen   = maxlen
    res
end
```

#### [MODIFY] `src/pqueue/pqueue.jl`

- **`knnqueue(T, ids, dists)`** — desde dos vectores ya asignados (para vistas de matrices).
- **`knnqueue(T, k::Int)`** — asigna `zeros(UInt32, k)` + `zeros(Float32, k)`.
- **`IdView(knn::KnnSorted)`** → `view(knn.ids, knn.sp:knn.ep)`. Análogo para `KnnHeap`.
- **`DistView(knn::KnnSorted)`** → `view(knn.dists, knn.sp:knn.ep)`.
- Los `convert` de/a `Matrix{IdDist}` se eliminan; los callers pasan a usar `IdView`/`DistView` directamente sobre las matrices separadas.
- `knnqueue(ctx::AbstractContext, ids, dists)` — overload de contexto que delega.

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
    @BATCHES minbatch begin
    @BEGINBATCH
        bctx = @set ctx.batchid = @batchid()
    @LOOP for j in 1:length(Q)
        res = knnqueue(bctx, view(ids, :, j), view(dists, :, j))
        search(index, bctx, Q[j], res)
        sorted && sortitems!(res)
    end
    end
    ids, dists
end
```

> **Nota**: La variante `searchbatch!(index, ctx, Q, knns::AbstractVector{<:AbstractKnn})`
> (vector de AbstractKnn preasignados) se mantiene sin cambios — no usa `Matrix{IdDist}`.

---

### Capa 3 — Algoritmos de alto nivel

#### [MODIFY] `src/allknn.jl`

Devuelve `(ids, dists)`. Variable interna `knns` pasa a dos matrices. `knnqueue` recibe vistas pareadas:

```julia
function allknn(g, ctx, k; ...)
    n = length(g)
    ids   = zeros(UInt32,  k, n)
    dists = zeros(Float32, k, n)
    allknn(g, ctx, ids, dists; ...)
end

function allknn(g, ctx, ids, dists; sort=true, progress=nothing)
    @BATCHES minbatch begin
    @BEGINBATCH bctx = @set ctx.batchid = @batchid()
    @LOOP for j in 1:n
        res = knnqueue(bctx, view(ids, :, j), view(dists, :, j))
        allknn_single_search!(g, bctx, j, res)
        sort && sortitems!(res)
    end
    end
    ids, dists
end
```

#### [MODIFY] `src/closestpair.jl`

```julia
# En parallel_closestpair:
ids_cache   = zeros(UInt32,  min_k, @nbatches())
dists_cache = zeros(Float32, min_k, @nbatches())
# ...
r = knnqueue(KnnSorted, view(ids_cache, :, @batchid()), view(dists_cache, :, @batchid()))
```

#### [MODIFY] `src/neardup.jl`

`knns = Matrix{IdDist}(undef, k, blocksize)` → dos matrices:

```julia
ids_buf   = zeros(UInt32,  k, blocksize)
dists_buf = zeros(Float32, k, blocksize)
```

Los accesos actuales:
```julia
p = knns[1, i]  # => IdDist
# p.dist, p.id
```
Pasan a:
```julia
nn_id   = ids_buf[1, i]    # UInt32
nn_dist = dists_buf[1, i]  # Float32
```

`searchbatch!(idx, ctx, X[range], knns_; sorted=true)` → `searchbatch!(idx, ctx, X[range], ids_, dists_; sorted=true)`.
`fill!(knns, zero(IdDist))` → `fill!(ids_buf, zero(UInt32)); fill!(dists_buf, zero(Float32))`.

#### [MODIFY] `src/hsp.jl`

`hsp_queries` actualmente construye `matrix = zeros(IdDist, size(knns)...)` y crea `hsp = [knnqueue(KnnSorted, c) for c in eachcol(matrix)]`. Con SoA:

```julia
function hsp_queries(dist, X, Q, ids::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32})
    n = length(Q)
    k = size(ids, 1)
    out_ids   = zeros(UInt32,  k, n)
    out_dists = zeros(Float32, k, n)
    hsp = [knnqueue(KnnSorted, view(out_ids,:,i), view(out_dists,:,i)) for i in 1:n]
    # ...
    for p_id in IdView(@view ids[:, i])
        p_id == 0 && break
        p_dist = dists[p_idx, i]  # necesita índice local
        # ...
    end
    out_ids, out_dists, hsp
end
```

> **Nota**: el loop interno actualmente itera `for p in plist` donde `p` es un `IdDist`. Con SoA hay que iterar sobre pares `(ids[j,i], dists[j,i])`. La iteración puede hacerse con `zip(view(ids,:,i), view(dists,:,i))` y romper en `id==0`.

`hsp_proximal_neighborhood_filter!` y `hsp_distal_neighborhood_filter!` reciben `neighborhood::AbstractKnn` e iteran `for p in neighborhood[i]` → estas funciones acceden a `p.dist` y `p.id`. Deben adaptarse a iterar sobre el wrapper de `viewitems` (Opción A de arriba) o recibir directamente `(ids_col, dists_col)`.

#### [MODIFY] `src/rerank.jl`

- **Variante por vector** (columna de una matriz o vista):
  ```julia
  function rerank!(dist, db, q, ids::AbstractVector{UInt32}, dists::AbstractVector{Float32})
      m = 0
      for i in eachindex(ids)
          ids[i] == 0 && break
          m = i
          dists[i] = evaluate(dist, db[ids[i]], q)
      end
      # sort paired: insertion sort pareado en dists[1:m] / ids[1:m]
      # usar sort_last_item! en un loop, o una sort! con un comparador que mueva ambos
      _sort_paired!(ids, dists, 1, m)
      ids, dists
  end
  ```

  > **Nota importante**: el `sort!` actual sobre `Vector{IdDist}` usa un comparador natural. Con dos arrays hay que hacer un **sort pareado** sin `sortperm`. La función `_sort_paired!(ids, dists, sp, ep)` puede implementarse como una variante de insertion sort igual a `sort_last_item!` pero para ordenar todo el rango (no solo el último elemento).

- **Variante batch**:
  ```julia
  function rerank!(dist, db, queries, ids::AbstractMatrix{UInt32}, dists::AbstractMatrix{Float32})
      m = length(queries)
      @BATCHES getminbatch(m) for i in 1:m
          rerank!(dist, db, queries[i], view(ids,:,i), view(dists,:,i))
      end
      ids, dists
  end
  ```

- **Variante `AbstractKnn`**: sigue igual pero accede a `res.ids` y `res.dists` directamente.

#### [MODIFY] `src/perf.jl`

```julia
# macrorecall recibe solo ids (UInt32) — recall no depende de distancias
function macrorecall(gold_ids::AbstractMatrix{UInt32},
                     ids::AbstractMatrix{UInt32},
                     k::Integer=size(gold_ids, 1))::Float64
    n = size(gold_ids, 2)
    s = 0.0
    for i in 1:n
        s += recallscore(view(gold_ids, 1:k, i), view(ids, 1:k, i))
    end
    s / n
end

# recallscore para vectores de UInt32 ya existe:
# idset(a::AbstractVector{<:Integer}) = Set{UInt32}(a)  ← ya funciona
```

Se elimina:
```julia
# YA NO NECESARIO:
idset(a::AbstractVector{IdDist}) = Set{UInt32}(IdView(a))
```

La variante de listas `macrorecall(goldlist, reslist)` y `recallscore(gold, res)` con `AbstractKnn` se mantienen.

> **Benchmark**: `benchmarks/searchgraph.jl` llama `macrorecall(gold, knns)` donde `gold` y `knns` son actualmente `Matrix{IdDist}`. Pasará a `macrorecall(gold_ids, ids)` con la nueva API. Es el ejemplo más visible del cambio.

#### [MODIFY] `src/opt.jl`

`knnsmatrix = zeros(IdDist, ksearch, length(queries))` → dos matrices. El código que accede a las columnas de `knnsmatrix` para construir result sets pasa a usar vistas pareadas.

---

### Capa 4 — `SearchGraph` y contextos

#### [MODIFY] `src/searchgraph/context.jl`

`beams::Matrix{IdDist}` → `beam_ids::Matrix{UInt32}` + `beam_dists::Matrix{Float32}`:

```julia
# En SearchGraphContext:
beam_ids::Matrix{UInt32}    # era: beams::Matrix{IdDist}
beam_dists::Matrix{Float32}

# getbeam devuelve vistas pareadas
function getbeam(i::Integer, ctx::SearchGraphContext)
    view(ctx.beam_ids, :, i), view(ctx.beam_dists, :, i)
end
```

El constructor adapta la inicialización:
```julia
beam_ids   === nothing && (beam_ids   = zeros(UInt32,  32, maxbatches))
beam_dists === nothing && (beam_dists = zeros(Float32, 32, maxbatches))
```

`ConstructionBase.constructorof` ya definido — debe incluir los dos nuevos campos.

#### [MODIFY] `src/searchgraph/rebuild.jl`

```julia
qcache_ids   = zeros(UInt32,  ksearch, 2 * @nbatches())
qcache_dists = zeros(Float32, ksearch, 2 * @nbatches())
# ...
tmp = knnqueue(bctx, view(qcache_ids, :, 2*@batchid()-1), view(qcache_dists, :, 2*@batchid()-1))
N   = knnqueue(bctx, view(qcache_ids, :, 2*@batchid()),   view(qcache_dists, :, 2*@batchid()))
```

#### [MODIFY] `src/searchgraph/insertions.jl`

Igual que `rebuild.jl`: `zeros(IdDist, s, t)` → dos matrices `zeros(UInt32, s, t)` + `zeros(Float32, s, t)`.

#### [MODIFY] `src/searchgraph/beamsearch.jl`

```julia
# Antes:
c = IdDist(childID, d)
push_item!(beam, c)

# Después (ya soportado por el overload (i, d)):
push_item!(beam, childID, d)
```

`getbeam` ahora devuelve una tupla `(ids_view, dists_view)` — actualizar el callsite en `beamsearch.jl`.

#### [MODIFY] `src/searchgraph/staticindexing.jl`

```julia
# Antes:
function index!(idx, ctx, ::Val{:knr}, knr::Matrix{IdDist}; ...)

# Después:
function index!(idx, ctx, ::Val{:knr}, knr_ids::Matrix{UInt32}, knr_dists::Matrix{Float32}; ...)
```

---

### Capa 5 — Archivos adicionales

#### [MODIFY] `src/permindex.jl`

El único uso de `IdDist` es:
```julia
@inbounds for i in eachindex(res.items)
    x = res.items[i]
    res.items[i] = IdDist(p.π[x.id], x.dist)
end
```
Con SoA, `res` ya no tiene `.items`. El código pasa a:
```julia
sp, ep = res.sp, res.ep
@inbounds for i in sp:ep
    res.ids[i] = p.π[res.ids[i]]
    # dists[i] no cambia
end
```

#### [MODIFY] `src/exact/basket-list.jl`

`BasketList` usa `Vector{IdDist}` como tipo de sus baskets internos. El basket tiene semántica propia (primer elemento = header con centro+radio, resto = objetos+distancia al centro). Opciones:

- **Opción A** (recomendada): Migrar cada basket a un par `(ids::Vector{UInt32}, dists::Vector{Float32})`. El struct pasa a `baskets::Vector{Tuple{Vector{UInt32}, Vector{Float32}}}`.
- **Opción B**: Definir un struct `Basket` con `ids` y `dists` para mayor claridad.

Actualmente `center = L[1]` extrae el header como `IdDist`. Con SoA: `center_id = ids[1]; center_dist = dists[1]`.

El `push_item!(res, IdDist(item.id, d))` en `search` pasa a `push_item!(res, item_id, d)`.

---

### Capa 6 — Tests

#### [MODIFY] `test/testresults.jl`

El test construye `gold = IdDist[]` y verifica `collect(viewitems(R)) == gold`. Con SoA:

```julia
gold_ids   = UInt32[]
gold_dists = Float32[]
# verificar:
@test collect(IdView(R))   == gold_ids
@test collect(DistView(R)) ≈  gold_dists
```

`pop_min!` y `pop_max!` siguen devolviendo `IdDist` — la verificación `@test p == popfirst!(gold)` etc. se adapta comparando `.id` y `.dist` separadamente, o manteniendo un vector `gold::Vector{IdDist}` para las comparaciones con `pop_min!`/`pop_max!` (que siguen devolviendo `IdDist`).

#### [MODIFY] `test/testsearchgraph.jl`

```julia
# Antes:
knns = zeros(IdDist, B.ksearch, length(B.queries))
searchbatch!(G, ctx, B.queries, knns)
recall = macrorecall(gold, knns)

# Después:
ids   = zeros(UInt32,  B.ksearch, length(B.queries))
dists = zeros(Float32, B.ksearch, length(B.queries))
ids, dists = searchbatch!(G, ctx, B.queries, ids, dists)
recall = macrorecall(gold_ids, ids)
```

#### [MODIFY] Tests adicionales que usan `searchbatch` / `IdDist` directamente:
`testexact.jl`, `testallknn.jl`, `testhsp.jl`, `testinvertedfiles.jl`, `testindexingprefixes.jl`.

#### [MODIFY] `benchmarks/searchgraph.jl`

```julia
# Antes:
gold = searchbatch(seq, GenericContext(), queries, ksearch)   # => Matrix{IdDist}
knns = searchbatch(graph, ctx, queries, ksearch; sorted=false) # => Matrix{IdDist}
recall = macrorecall(gold, knns)

# Después:
gold_ids, _   = searchbatch(seq,   GenericContext(), queries, ksearch)
ids, _        = searchbatch(graph, ctx, queries, ksearch; sorted=false)
recall        = macrorecall(gold_ids, ids)
```

---

## Orden de ejecución

1. `src/pqueue/heap.jl` — primitivas pareadas
2. `src/pqueue/knnsorted.jl` — nueva struct + métodos + wrapper `viewitems`
3. `src/pqueue/knnheap.jl` — nueva struct + métodos
4. `src/pqueue/pqueue.jl` — nuevos `knnqueue`, `IdView`/`DistView` simplificados
5. `src/SimilaritySearch.jl` — `searchbatch` / `searchbatch!`
6. `src/searchgraph/context.jl` — `beam_ids` + `beam_dists`, `getbeam`
7. `src/searchgraph/beamsearch.jl`
8. `src/searchgraph/{rebuild,insertions,staticindexing}.jl`
9. `src/{allknn,closestpair,neardup,opt}.jl`
10. `src/perf.jl`, `src/rerank.jl`, `src/hsp.jl`
11. `src/permindex.jl`, `src/exact/basket-list.jl`
12. Tests + benchmarks

---

## Verification Plan

```sh
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
```

- `searchbatch` devuelve `(ids::Matrix{UInt32}, dists::Matrix{Float32})`.
- `allknn` devuelve `(ids, dists)`.
- Recall en `testsearchgraph.jl` ≥ 0.9.
- `macrorecall(gold_ids, ids)` funciona con matrices `UInt32`.
- `benchmarks/searchgraph.jl` corre correctamente de extremo a extremo.
