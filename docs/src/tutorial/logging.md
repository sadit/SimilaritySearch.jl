```@meta
CurrentModule = SimilaritySearch
```

# Logging and Observation Channels

In `SimilaritySearch.jl`, execution contexts maintain two separate channels to distinguish between human-readable reporting and programmatically actionable structural observation:

1. **Reporters (`ctx.reporters`)**: Receive diagnostic and progress messages via [`INFORM`](@ref). These format status lines for display on `stderr`, logs, or monitoring consoles.
2. **Observers (`ctx.observers`)**: Receive structural mutation events via [`OBSERVE`](@ref). These enable write-ahead logging, incremental checkpoints, and dynamic metric tracking.

Decoupling these channels ensures that suppressing console output does not disrupt data persistence or event hooks.

---

## Controlling Console Output: Reporters

### Silencing Progress Output

To disable all informational logging, set `reporters` to an empty array:

```julia
using SimilaritySearch

ctx = SearchGraphContext(; reporters=[])
```

When `reporters` is empty, log messages are skipped entirely with negligible computational overhead.

### Verbosity vs. Reporters

- `reporters`: Controls the destination sinks for informational messages.
- `verbose`: A boolean flag determining whether detailed per-iteration diagnostics (such as optimization trajectories and hint selections) are emitted. Defaults to `false`.

```julia
ctx = SearchGraphContext(; verbose=true)     # Emits detailed optimization diagnostics
ctx = SearchGraphContext(; reporters=[])     # Completely silences output regardless of verbose
```

---

## Configuring `InformativeLog`

The default reporter is [`InformativeLog`](@ref), which emits timestamped status lines containing index cardinality, live heap allocation, and resident memory (RSS). 

Messages are rate-limited to emit at most once every `dt` seconds:

```julia
# Log to stderr with a 0.5-second throttle and custom prefix
ctx = SearchGraphContext(; reporters=InformativeLog(; dt=0.5, prompt="[Build]"))

# Direct logs to a file stream
io = open("build.log", "a")
ctx = SearchGraphContext(; reporters=[InformativeLog(), InformativeLog(io)])

# Disable rate-limiting (dt=0 logs all events synchronously)
ctx = GenericContext(; reporters=InformativeLog(; dt=0))
```

---

## Structural Events: Observers and the `:add!` Contract

Mutating operations (`push_item!`, `append_items!`, `index!`) trigger structural notifications by calling:

$$\text{OBSERVE}(\text{ctx}, \text{:add!}, \text{index}, sp, ep)$$

where $sp:ep$ defines the contiguous range of object identifiers inserted into the index.

### Guarantees of the `:add!` Contract

1. **Single Emission**: When high-level functions delegate work (e.g., `append_items!` extending the database and calling `index!`), only the mutating layer emits the `:add!` event, preventing duplicate notifications.
2. **Scope Isolation**: Observers should be attached to a single index instance to ensure identifier ranges remain unambiguous.

### Implementing Observers

`SimilaritySearch.jl` provides [`CallbackLog`](@ref) for functional hooks, or users can subtype `SimilaritySearch.AbstractObserver`:

```julia
# Using CallbackLog
ranges = Tuple{Int, Int}[]
ctx = SearchGraphContext(; observers=CallbackLog((index, sp, ep) -> push!(ranges, (Int(sp), Int(ep)))))

# Custom AbstractObserver implementation
struct CountingObserver <: SimilaritySearch.AbstractObserver
    counts::Dict{Symbol, Int}
end
CountingObserver() = CountingObserver(Dict{Symbol, Int}())

function SimilaritySearch.OBSERVE(log::CountingObserver, event::Symbol, index, ctx, sp::Integer, ep::Integer)
    log.counts[event] = get(log.counts, event, 0) + (ep - sp + 1)
end

ctx = SearchGraphContext(; observers=CountingObserver())
```

---

## Context Inheritance Rules

When library algorithms instantiate internal auxiliary contexts (such as scratch indexes created during parameter tuning), these child contexts inherit the parent's `reporters` (preserving output configuration), but **do not** inherit `observers`. This prevents auxiliary operations from emitting events into the primary index's observation stream.

---

## Incremental Neighborhood Tracking

Because `add!(index.adj, objID, ...)` executes prior to the `:add!` event, an observer can inspect newly computed neighborhoods immediately upon insertion:

```julia
struct NeighborCaptureLog <: SimilaritySearch.AbstractObserver
    captured::Dict{Int, Vector{UInt32}}
end
NeighborCaptureLog() = NeighborCaptureLog(Dict{Int, Vector{UInt32}}())

function SimilaritySearch.OBSERVE(log::NeighborCaptureLog, event::Symbol, index::SearchGraph, ctx::SearchGraphContext, sp::Integer, ep::Integer)
    for i in sp:ep
        log.captured[i] = collect(neighbors(index.adj, i))
    end
end

X = MatrixDatabase(rand(Float32, 4, 500))
G = SearchGraph(Dist.L2(), X)
mylog = NeighborCaptureLog()
ctx = SearchGraphContext(; observers=mylog)
index!(G, ctx)

length(mylog.captured)   # 500 objects recorded at insertion time
```

### Forward Neighbors vs. Bidirectional Connectivity

In `SearchGraph`, edge creation includes:
1. **Forward links**: When object $j$ is inserted, it discovers nearest neighbors among $1, \dots, j-1$ and connects directed edges $j \to i$.
2. **Reverse links**: Object $j$ also connects reverse edges $i \to j$ to ensure graph bidirectionality.

Consequently, the neighborhood captured for object $i$ at its insertion moment contains only its initial forward connections. To retrieve the complete bidirectional neighborhood after construction completes, inspect the final adjacency structure:

```julia
final_neighbors = Dict(i => collect(neighbors(G.adj, i)) for i in 1:length(G))
```

---

In the next section, [Inverted Files and Posting List Intersections](invertedfiles.md), we explore inverted indexing for sparse vector search and set metrics.
